// Video decoder for mtmd on Apple platforms, on top of AVFoundation (see mtmd_helper_video_decoder).
// Built without ARC.

#include "rn-video.h"

#import <AVFoundation/AVFoundation.h>
#import <Accelerate/Accelerate.h>

#include <cmath>
#include <vector>

namespace {

struct apple_video {
    AVURLAsset *             asset        = nil;
    AVAssetTrack *           video_track  = nil;
    AVAssetReader *          reader       = nil;
    AVAssetReaderTrackOutput * output     = nil;
    NSString *               tmp_path     = nil; // the content of a video given without a path
    CMTimeRange              frames_range = kCMTimeRangeZero; // from the first frame to the end of the last one
    int                      quarter_turns = 0;  // clockwise, to display a decoded frame
    uint32_t                 width        = 0;   // displayed size
    uint32_t                 height       = 0;

    std::vector<uint8_t> scaled; // a frame at the requested size, before it is turned upright
    std::vector<uint8_t> upright;
    std::vector<float>   audio;

    ~apple_video() {
        [reader cancelReading];
        [output release];
        [reader release];
        [video_track release];
        [asset release];
        if (tmp_path) {
            [[NSFileManager defaultManager] removeItemAtPath:tmp_path error:nil];
            [tmp_path release];
        }
    }
};

// Compressed samples are counted without decoding them, the container has no exact frame count.
// range is the time span of the frames: the track's own range also covers an empty edit before
// the first frame (a video that starts later than its soundtrack), which has no frame to decode.
int32_t count_frames(AVAsset * asset, AVAssetTrack * track, CMTimeRange * range) {
    AVAssetReader * reader = [AVAssetReader assetReaderWithAsset:asset error:nil];
    AVAssetReaderTrackOutput * output = [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:track outputSettings:nil];
    output.alwaysCopiesSampleData = NO;
    if (!reader || ![reader canAddOutput:output]) {
        return -1;
    }
    [reader addOutput:output];
    if (![reader startReading]) {
        return -1;
    }
    int32_t n = 0;
    CMTime start = kCMTimePositiveInfinity;
    CMTime end   = kCMTimeNegativeInfinity;
    while (CMSampleBufferRef sample = [output copyNextSampleBuffer]) {
        const CMItemCount n_samples = CMSampleBufferGetNumSamples(sample);
        if (n_samples > 0) {
            const CMTime pts = CMSampleBufferGetOutputPresentationTimeStamp(sample);
            start = CMTimeMinimum(start, pts);
            end   = CMTimeMaximum(end, CMTimeAdd(pts, CMSampleBufferGetOutputDuration(sample)));
            n += (int32_t) n_samples;
        }
        CFRelease(sample);
    }
    if (reader.status != AVAssetReaderStatusCompleted || n == 0) {
        return -1;
    }
    *range = CMTimeRangeFromTimeToTime(start, end);
    return n;
}

void * video_open(void *, const char * path, const unsigned char * buf, size_t len, mtmd_helper_video_stream_info * info) {
    @autoreleasepool {
        auto * v = new apple_video();
        NSURL * url = nil;
        if (path) {
            url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        } else {
            // AVFoundation only reads from a URL
            NSString * name = [NSString stringWithFormat:@"rnllama-video-%@.mp4", [[NSUUID UUID] UUIDString]];
            v->tmp_path = [[NSTemporaryDirectory() stringByAppendingPathComponent:name] retain];
            if (![[NSData dataWithBytesNoCopy:(void *) buf length:len freeWhenDone:NO] writeToFile:v->tmp_path atomically:NO]) {
                delete v;
                return nullptr;
            }
            url = [NSURL fileURLWithPath:v->tmp_path];
        }

        v->asset = [[AVURLAsset alloc] initWithURL:url options:nil];
        v->video_track = [[[v->asset tracksWithMediaType:AVMediaTypeVideo] firstObject] retain];
        if (!v->video_track) {
            delete v;
            return nullptr;
        }

        // the track's transform is a rotation by a multiple of 90 degrees (the orientation of the camera)
        const CGAffineTransform t = v->video_track.preferredTransform;
        v->quarter_turns = ((int) std::lround(std::atan2(t.b, t.a) / M_PI_2) % 4 + 4) % 4;
        const CGSize size = v->video_track.naturalSize;
        const bool swap = v->quarter_turns % 2 == 1;
        v->width  = (uint32_t) std::lround(swap ? size.height : size.width);
        v->height = (uint32_t) std::lround(swap ? size.width : size.height);

        v->frames_range = v->video_track.timeRange;
        const int32_t n_frames = count_frames(v->asset, v->video_track, &v->frames_range);
        const double duration = CMTimeGetSeconds(v->frames_range.duration);

        v->reader = [[AVAssetReader alloc] initWithAsset:v->asset error:nil];
        v->output = [[AVAssetReaderTrackOutput alloc]
            initWithTrack:v->video_track
            outputSettings:@{(id) kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)}];
        v->output.alwaysCopiesSampleData = NO;
        if (!v->reader || ![v->reader canAddOutput:v->output]) {
            delete v;
            return nullptr;
        }
        [v->reader addOutput:v->output];
        v->reader.timeRange = v->frames_range; // an empty edit would be decoded as a black frame
        if (![v->reader startReading]) {
            delete v;
            return nullptr;
        }

        info->width    = v->width;
        info->height   = v->height;
        info->fps      = rnllama::video_stream_fps(v->video_track.nominalFrameRate, n_frames, duration);
        info->n_frames = n_frames;
        info->duration = duration > 0.0 ? (float) duration : -1.0f;
        return v;
    }
}

bool video_read_frame(void * state, unsigned char * rgb, uint32_t width, uint32_t height) {
    auto * v = static_cast<apple_video *>(state);
    @autoreleasepool {
        CMSampleBufferRef sample = nullptr;
        CVImageBufferRef image = nullptr;
        // a sample without an image only carries timing information
        while (!image) {
            if (sample) {
                CFRelease(sample);
            }
            sample = [v->output copyNextSampleBuffer];
            if (!sample) {
                return false;
            }
            image = CMSampleBufferGetImageBuffer(sample);
        }
        if (!rgb) {
            CFRelease(sample);
            return true;
        }

        CVPixelBufferLockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
        vImage_Buffer frame = {
            CVPixelBufferGetBaseAddress(image),
            CVPixelBufferGetHeight(image),
            CVPixelBufferGetWidth(image),
            CVPixelBufferGetBytesPerRow(image),
        };
        vImage_Error err = kvImageNoError;
        // scale first: the frame is then turned upright at the requested size, not at the decoded one
        const bool   swap     = v->quarter_turns % 2 == 1;
        const size_t scaled_w = swap ? height : width;
        const size_t scaled_h = swap ? width : height;
        if (frame.width != scaled_w || frame.height != scaled_h) {
            v->scaled.resize(scaled_w * scaled_h * 4);
            vImage_Buffer scaled = {v->scaled.data(), scaled_h, scaled_w, scaled_w * 4};
            err = vImageScale_ARGB8888(&frame, &scaled, nullptr, kvImageHighQualityResampling);
            frame = scaled;
        }
        if (err == kvImageNoError && v->quarter_turns != 0) {
            v->upright.resize((size_t) width * height * 4);
            vImage_Buffer upright = {v->upright.data(), height, width, (size_t) width * 4};
            const Pixel_8888 background = {0, 0, 0, 0};
            // vImage counts quarter turns counterclockwise
            err = vImageRotate90_ARGB8888(&frame, &upright, (uint8_t) (4 - v->quarter_turns), background, kvImageNoFlags);
            frame = upright;
        }
        if (err == kvImageNoError) {
            vImage_Buffer out = {rgb, height, width, (size_t) width * 3};
            err = vImageConvert_BGRA8888toRGB888(&frame, &out, kvImageNoFlags);
        }
        CVPixelBufferUnlockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
        CFRelease(sample);
        return err == kvImageNoError;
    }
}

// the whole soundtrack over the span of the video frames, AVFoundation mixes it down and resamples it
void load_audio(apple_video * v, int sample_rate) {
    AVAssetTrack * track = [[v->asset tracksWithMediaType:AVMediaTypeAudio] firstObject];
    if (!track) {
        return;
    }
    AVAssetReader * reader = [AVAssetReader assetReaderWithAsset:v->asset error:nil];
    AVAssetReaderTrackOutput * output = [AVAssetReaderTrackOutput
        assetReaderTrackOutputWithTrack:track
        outputSettings:@{
            AVFormatIDKey: @(kAudioFormatLinearPCM),
            AVSampleRateKey: @(sample_rate),
            AVNumberOfChannelsKey: @1,
            AVLinearPCMBitDepthKey: @32,
            AVLinearPCMIsFloatKey: @YES,
            AVLinearPCMIsBigEndianKey: @NO,
            AVLinearPCMIsNonInterleaved: @NO,
        }];
    if (!reader || ![reader canAddOutput:output]) {
        return;
    }
    [reader addOutput:output];
    reader.timeRange = v->frames_range;
    if (![reader startReading]) {
        return;
    }
    while (CMSampleBufferRef sample = [output copyNextSampleBuffer]) {
        CMBlockBufferRef block = CMSampleBufferGetDataBuffer(sample);
        const size_t n_bytes = block ? CMBlockBufferGetDataLength(block) : 0;
        const size_t pos = v->audio.size();
        v->audio.resize(pos + n_bytes / sizeof(float));
        if (n_bytes > 0) {
            CMBlockBufferCopyDataBytes(block, 0, n_bytes / sizeof(float) * sizeof(float), v->audio.data() + pos);
        }
        CFRelease(sample);
    }
}

const float * video_read_audio(void * state, int sample_rate, size_t * n_samples) {
    auto * v = static_cast<apple_video *>(state);
    v->audio.clear();
    @autoreleasepool {
        load_audio(v, sample_rate);
    }
    *n_samples = v->audio.size();
    return v->audio.empty() ? nullptr : v->audio.data();
}

void video_close(void * state) {
    @autoreleasepool {
        delete static_cast<apple_video *>(state);
    }
}

} // namespace

namespace rnllama {

const mtmd_helper_video_decoder * video_decoder() {
    static const mtmd_helper_video_decoder decoder = {
        video_open, video_read_frame, video_read_audio, video_close,
        /* user_data */ nullptr, /* start */ nullptr,
    };
    return &decoder;
}

} // namespace rnllama
