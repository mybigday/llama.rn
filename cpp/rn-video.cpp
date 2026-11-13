// Video decoder for mtmd on Android, on top of the NDK media APIs (see mtmd_helper_video_decoder).
// The Apple decoder is rn-video-apple.mm; the other platforms have none.

// AMediaDataSource (API 28) reads a video from memory. The library also runs on older Android
// versions, where a video must be given by its path: the NDK declares the symbols weak when this
// is defined before its first header, and their use is guarded at run time.
#define __ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__ 1

#include "rn-video.h"

#if defined(__ANDROID__)

#include <media/NdkMediaCodec.h>
#include <media/NdkMediaDataSource.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

// MediaCodecInfo.CodecCapabilities color formats
constexpr int32_t COLOR_FORMAT_YUV420_PLANAR        = 19;
constexpr int32_t COLOR_FORMAT_YUV420_PACKED_PLANAR = 20;
// MediaFormat constants
constexpr int32_t COLOR_STANDARD_BT709 = 1;
constexpr int32_t COLOR_RANGE_FULL     = 1;
constexpr int32_t PCM_ENCODING_FLOAT   = 4;

constexpr int64_t CODEC_TIMEOUT_US = 10000;
constexpr int     CODEC_MAX_STALLS = 500; // 5 seconds without a frame: the codec is stuck

// where the video is: a file, or its content
struct media_source {
    std::string          path;
    std::vector<uint8_t> data;
};

// one track of the source and its decoder
struct track_decoder {
    AMediaExtractor * extractor = nullptr;
    AMediaCodec *     codec     = nullptr;
    AMediaFormat *    track_format  = nullptr;
    AMediaFormat *    output_format = nullptr;
    AMediaDataSource * data_source  = nullptr;
    int               fd            = -1;
    bool              input_done    = false;
    bool              output_done   = false;

    track_decoder() = default;
    track_decoder(const track_decoder &) = delete;
    track_decoder & operator=(const track_decoder &) = delete;

    ~track_decoder() {
        if (codec) {
            AMediaCodec_stop(codec);
            AMediaCodec_delete(codec);
        }
        if (output_format) {
            AMediaFormat_delete(output_format);
        }
        if (track_format) {
            AMediaFormat_delete(track_format);
        }
        if (extractor) {
            AMediaExtractor_delete(extractor);
        }
        if (data_source) {
            if (__builtin_available(android 28, *)) {
                AMediaDataSource_delete(data_source);
            }
        }
        if (fd >= 0) {
            close(fd);
        }
    }

    // select the first track whose mime type starts with mime_prefix ("video/", "audio/")
    bool open(const media_source & source, const char * mime_prefix) {
        extractor = AMediaExtractor_new();
        if (!source.path.empty()) {
            struct stat st;
            fd = ::open(source.path.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0 || fstat(fd, &st) != 0 ||
                    AMediaExtractor_setDataSourceFd(extractor, fd, 0, st.st_size) != AMEDIA_OK) {
                return false;
            }
        } else if (__builtin_available(android 28, *)) {
            data_source = AMediaDataSource_new();
            AMediaDataSource_setUserdata(data_source, (void *) &source.data);
            AMediaDataSource_setReadAt(data_source, [](void * userdata, off64_t offset, void * buffer, size_t size) -> ssize_t {
                const auto & data = *static_cast<const std::vector<uint8_t> *>(userdata);
                if (offset < 0 || (size_t) offset >= data.size()) {
                    return 0; // end of the data (-1 is taken as a read error, and the file as broken)
                }
                const size_t n = std::min(size, data.size() - (size_t) offset);
                memcpy(buffer, data.data() + offset, n);
                return (ssize_t) n;
            });
            AMediaDataSource_setGetSize(data_source, [](void * userdata) -> ssize_t {
                return (ssize_t) static_cast<const std::vector<uint8_t> *>(userdata)->size();
            });
            AMediaDataSource_setClose(data_source, [](void *) {});
            if (AMediaExtractor_setDataSourceCustom(extractor, data_source) != AMEDIA_OK) {
                return false;
            }
        } else {
            return false; // before Android 9
        }

        const size_t n_tracks = AMediaExtractor_getTrackCount(extractor);
        for (size_t i = 0; i < n_tracks; i++) {
            AMediaFormat * format = AMediaExtractor_getTrackFormat(extractor, i);
            const char * mime = nullptr;
            if (AMediaFormat_getString(format, AMEDIAFORMAT_KEY_MIME, &mime) && strncmp(mime, mime_prefix, strlen(mime_prefix)) == 0) {
                AMediaExtractor_selectTrack(extractor, i);
                track_format = format;
                return true;
            }
            AMediaFormat_delete(format);
        }
        return false;
    }

    bool start() {
        const char * mime = nullptr;
        AMediaFormat_getString(track_format, AMEDIAFORMAT_KEY_MIME, &mime);
        codec = AMediaCodec_createDecoderByType(mime);
        return codec &&
               AMediaCodec_configure(codec, track_format, nullptr, nullptr, 0) == AMEDIA_OK &&
               AMediaCodec_start(codec) == AMEDIA_OK;
    }

    // the next decoded buffer, to be released by the caller; -1 at the end of the track or on error
    ssize_t next_output(AMediaCodecBufferInfo & info) {
        int stalls = 0;
        while (!output_done && stalls < CODEC_MAX_STALLS) {
            if (!input_done) {
                const ssize_t in = AMediaCodec_dequeueInputBuffer(codec, CODEC_TIMEOUT_US);
                if (in >= 0) {
                    size_t capacity = 0;
                    uint8_t * buf = AMediaCodec_getInputBuffer(codec, (size_t) in, &capacity);
                    const ssize_t n = AMediaExtractor_readSampleData(extractor, buf, capacity);
                    if (n < 0) {
                        AMediaCodec_queueInputBuffer(codec, (size_t) in, 0, 0, 0, AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM);
                        input_done = true;
                    } else {
                        AMediaCodec_queueInputBuffer(codec, (size_t) in, 0, (size_t) n, AMediaExtractor_getSampleTime(extractor), 0);
                        AMediaExtractor_advance(extractor);
                    }
                }
            }
            const ssize_t out = AMediaCodec_dequeueOutputBuffer(codec, &info, CODEC_TIMEOUT_US);
            if (out >= 0) {
                if (info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM) {
                    output_done = true;
                    if (info.size == 0) {
                        AMediaCodec_releaseOutputBuffer(codec, (size_t) out, false);
                        return -1;
                    }
                }
                return out;
            }
            if (out == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
                if (output_format) {
                    AMediaFormat_delete(output_format);
                }
                output_format = AMediaCodec_getOutputFormat(codec);
            } else if (out == AMEDIACODEC_INFO_TRY_AGAIN_LATER) {
                stalls++;
            } else if (out != AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) {
                break; // codec error
            }
        }
        output_done = true;
        return -1;
    }

    int32_t output_int(const char * key, int32_t fallback) const {
        int32_t value = fallback;
        if (!output_format || !AMediaFormat_getInt32(output_format, key, &value)) {
            AMediaFormat_getInt32(track_format, key, &value);
        }
        return value;
    }
};

struct android_video {
    media_source  source;
    track_decoder video;
    int           rotation = 0; // degrees clockwise, to display a decoded frame
    uint32_t      width    = 0; // displayed size
    uint32_t      height   = 0;
    int64_t       start_us = 0; // time of the first video frame

    // a frame as decoded, then at the requested size before it is turned upright
    std::vector<uint8_t> converted;
    std::vector<uint8_t> scaled;
    std::vector<float>   resize_tmp;
    std::vector<float>   audio;
};

uint8_t clamp_u8(int v) {
    return (uint8_t) std::clamp(v, 0, 255);
}

// where the visible pixels are in the codec's YUV 4:2:0 output buffer, and their colors
struct yuv_layout {
    int32_t stride, slice_h, crop_x, crop_y;
    int32_t w, h; // visible size
    bool    planar, bt709, full_range;

    explicit yuv_layout(const track_decoder & d) {
        const int32_t coded_w = d.output_int("width", 0);
        const int32_t coded_h = d.output_int("height", 0);
        stride  = std::max(d.output_int("stride", coded_w), coded_w);
        slice_h = std::max(d.output_int("slice-height", coded_h), coded_h);
        crop_x  = d.output_int("crop-left", 0);
        crop_y  = d.output_int("crop-top", 0);
        w       = d.output_int("crop-right", coded_w - 1) - crop_x + 1;
        h       = d.output_int("crop-bottom", coded_h - 1) - crop_y + 1;
        // anything but I420 is taken as NV12, as the vendor formats of the hardware decoders are
        const int32_t format = d.output_int("color-format", COLOR_FORMAT_YUV420_PLANAR);
        planar = format == COLOR_FORMAT_YUV420_PLANAR || format == COLOR_FORMAT_YUV420_PACKED_PLANAR;
        // unspecified colors: BT.709 for HD, BT.601 below, both limited range
        bt709      = d.output_int("color-standard", h >= 720 ? COLOR_STANDARD_BT709 : 0) == COLOR_STANDARD_BT709;
        full_range = d.output_int("color-range", 0) == COLOR_RANGE_FULL;
    }
};

// the visible pixels of the codec's output buffer -> RGB24, f.w x f.h
void convert_frame(const yuv_layout & f, const uint8_t * buf, uint8_t * rgb) {
    // 10-bit fixed point
    const int y_gain = f.full_range ? 1024 : 1192;
    const int y_bias = f.full_range ? 0 : 16;
    const int r_v    = f.full_range ? (f.bt709 ? 1613 : 1436) : (f.bt709 ? 1836 : 1634);
    const int g_u    = f.full_range ? (f.bt709 ? 192 : 352) : (f.bt709 ? 218 : 401);
    const int g_v    = f.full_range ? (f.bt709 ? 479 : 731) : (f.bt709 ? 546 : 833);
    const int b_u    = f.full_range ? (f.bt709 ? 1900 : 1815) : (f.bt709 ? 2163 : 2066);

    const uint8_t * u_plane  = buf + (size_t) f.stride * f.slice_h;
    const uint8_t * v_plane  = f.planar ? u_plane + (size_t) (f.stride / 2) * (f.slice_h / 2) : u_plane + 1;
    const int32_t   c_stride = f.planar ? f.stride / 2 : f.stride;
    const int32_t   c_step   = f.planar ? 1 : 2;

    for (int32_t y = 0; y < f.h; y++) {
        const uint8_t * y_row = buf + (size_t) (y + f.crop_y) * f.stride + f.crop_x;
        const size_t    c_row = (size_t) ((y + f.crop_y) / 2) * c_stride;
        for (int32_t x = 0; x < f.w; x++, rgb += 3) {
            const size_t c = c_row + (size_t) ((x + f.crop_x) / 2) * c_step;
            const int yy = (y_row[x] - y_bias) * y_gain;
            const int cb = u_plane[c] - 128;
            const int cr = v_plane[c] - 128;
            rgb[0] = clamp_u8((yy + r_v * cr + 512) >> 10);
            rgb[1] = clamp_u8((yy - g_u * cb - g_v * cr + 512) >> 10);
            rgb[2] = clamp_u8((yy + b_u * cb + 512) >> 10);
        }
    }
}

// turn a w x h frame clockwise by rotation degrees (90, 180 or 270)
void rotate_rgb(const uint8_t * src, int32_t w, int32_t h, uint8_t * dst, int rotation) {
    const int32_t dst_w = rotation == 180 ? w : h;
    for (int32_t y = 0; y < h; y++) {
        for (int32_t x = 0; x < w; x++, src += 3) {
            int32_t dx, dy;
            switch (rotation) {
                case 90:  dx = h - 1 - y; dy = x;         break;
                case 180: dx = w - 1 - x; dy = h - 1 - y; break;
                default:  dx = y;         dy = w - 1 - x; break;
            }
            memcpy(dst + ((size_t) dy * dst_w + dx) * 3, src, 3);
        }
    }
}

// the source pixels and weights of every output pixel along one axis: a triangle filter,
// widened by the scale when shrinking so that no source pixel is skipped
struct resize_axis {
    std::vector<int32_t> start;
    std::vector<int32_t> count;
    std::vector<float>   weights; // max_count per output pixel
    int32_t              max_count = 0;

    resize_axis(int32_t n_src, int32_t n_dst) : start(n_dst), count(n_dst) {
        const double scale   = (double) n_src / n_dst;
        const double support = std::max(scale, 1.0);
        max_count = (int32_t) std::ceil(support * 2.0) + 1;
        weights.assign((size_t) n_dst * max_count, 0.0f);
        for (int32_t i = 0; i < n_dst; i++) {
            const double center = (i + 0.5) * scale;
            const int32_t first = std::max(0, (int32_t) std::floor(center - support));
            const int32_t last  = std::min(n_src - 1, (int32_t) std::ceil(center + support) - 1);
            float * w = weights.data() + (size_t) i * max_count;
            float sum = 0.0f;
            for (int32_t j = first; j <= last; j++) {
                w[j - first] = (float) std::max(0.0, 1.0 - std::abs(j + 0.5 - center) / support);
                sum += w[j - first];
            }
            for (int32_t j = 0; j <= last - first; j++) {
                w[j] /= sum;
            }
            start[i] = first;
            count[i] = last - first + 1;
        }
    }
};

void resize_rgb(const uint8_t * src, int32_t sw, int32_t sh, uint8_t * dst, int32_t dw, int32_t dh, std::vector<float> & tmp) {
    const resize_axis ax(sw, dw);
    const resize_axis ay(sh, dh);
    tmp.resize((size_t) dw * sh * 3);
    for (int32_t y = 0; y < sh; y++) {
        for (int32_t x = 0; x < dw; x++) {
            const float *   w = ax.weights.data() + (size_t) x * ax.max_count;
            const uint8_t * p = src + ((size_t) y * sw + ax.start[x]) * 3;
            float acc[3] = {0.0f, 0.0f, 0.0f};
            for (int32_t k = 0; k < ax.count[x]; k++, p += 3) {
                acc[0] += w[k] * p[0];
                acc[1] += w[k] * p[1];
                acc[2] += w[k] * p[2];
            }
            memcpy(tmp.data() + ((size_t) y * dw + x) * 3, acc, sizeof(acc));
        }
    }
    for (int32_t y = 0; y < dh; y++) {
        const float * w = ay.weights.data() + (size_t) y * ay.max_count;
        for (int32_t x = 0; x < dw; x++) {
            float acc[3] = {0.0f, 0.0f, 0.0f};
            for (int32_t k = 0; k < ay.count[y]; k++) {
                const float * p = tmp.data() + ((size_t) (ay.start[y] + k) * dw + x) * 3;
                acc[0] += w[k] * p[0];
                acc[1] += w[k] * p[1];
                acc[2] += w[k] * p[2];
            }
            uint8_t * out = dst + ((size_t) y * dw + x) * 3;
            out[0] = clamp_u8((int) std::lround(acc[0]));
            out[1] = clamp_u8((int) std::lround(acc[1]));
            out[2] = clamp_u8((int) std::lround(acc[2]));
        }
    }
}

void * video_open(void *, const char * path, const unsigned char * buf, size_t len, mtmd_helper_video_stream_info * info) {
    auto * v = new android_video();
    if (path) {
        v->source.path = path;
    } else {
        v->source.data.assign(buf, buf + len);
    }
    track_decoder & d = v->video;
    if (!d.open(v->source, "video/")) {
        delete v;
        return nullptr;
    }

    int32_t w = 0, h = 0, frame_rate = 0;
    int64_t duration_us = 0;
    AMediaFormat_getInt32(d.track_format, AMEDIAFORMAT_KEY_WIDTH, &w);
    AMediaFormat_getInt32(d.track_format, AMEDIAFORMAT_KEY_HEIGHT, &h);
    AMediaFormat_getInt32(d.track_format, AMEDIAFORMAT_KEY_FRAME_RATE, &frame_rate);
    AMediaFormat_getInt64(d.track_format, AMEDIAFORMAT_KEY_DURATION, &duration_us);
    AMediaFormat_getInt32(d.track_format, "rotation-degrees", &v->rotation);
    v->rotation = (v->rotation % 360 + 360) % 360;
    const bool swap = v->rotation == 90 || v->rotation == 270;
    v->width  = (uint32_t) (swap ? h : w);
    v->height = (uint32_t) (swap ? w : h);

    // The container has no frame count: walk the samples, without decoding them. Their times also
    // give the span of the frames: the track's duration may cover an empty edit before the first
    // frame (a video that starts later than its soundtrack).
    int32_t n_frames = 0;
    int64_t first_us = INT64_MAX;
    int64_t last_us  = INT64_MIN;
    for (int64_t t; (t = AMediaExtractor_getSampleTime(d.extractor)) >= 0;) {
        n_frames++;
        first_us = std::min(first_us, t);
        last_us  = std::max(last_us, t);
        if (!AMediaExtractor_advance(d.extractor)) {
            break;
        }
    }
    AMediaExtractor_seekTo(d.extractor, 0, AMEDIAEXTRACTOR_SEEK_PREVIOUS_SYNC);

    double duration = duration_us / 1e6;
    if (n_frames > 1) {
        v->start_us = first_us;
        // the last frame lasts as long as the others do on average
        duration = (last_us - first_us) / 1e6 * n_frames / (n_frames - 1);
    }

    // Codec2 decoders convert to the requested layout, the older ones keep their own (see convert_frame)
    AMediaFormat_setInt32(d.track_format, AMEDIAFORMAT_KEY_COLOR_FORMAT, COLOR_FORMAT_YUV420_PLANAR);
    if (!d.start()) {
        delete v;
        return nullptr;
    }

    info->width    = v->width;
    info->height   = v->height;
    info->fps      = rnllama::video_stream_fps((float) frame_rate, n_frames, duration);
    info->n_frames = n_frames > 0 ? n_frames : -1;
    info->duration = duration > 0.0 ? (float) duration : -1.0f;
    return v;
}

bool video_read_frame(void * state, unsigned char * rgb, uint32_t width, uint32_t height) {
    auto * v = static_cast<android_video *>(state);
    AMediaCodecBufferInfo info;
    const ssize_t idx = v->video.next_output(info);
    if (idx < 0) {
        return false;
    }
    if (rgb) {
        size_t size = 0;
        const uint8_t * buf = AMediaCodec_getOutputBuffer(v->video.codec, (size_t) idx, &size);
        if (!buf) {
            AMediaCodec_releaseOutputBuffer(v->video.codec, (size_t) idx, false);
            return false;
        }
        // convert, scale, then turn upright: the frame is turned at the requested size, not at
        // the decoded one. A step that is not needed is skipped, the last one writes to rgb.
        const yuv_layout f(v->video);
        const bool    swap     = v->rotation == 90 || v->rotation == 270;
        const int32_t scaled_w = (int32_t) (swap ? height : width);
        const int32_t scaled_h = (int32_t) (swap ? width : height);
        const bool    scale    = f.w != scaled_w || f.h != scaled_h;
        const bool    rotate   = v->rotation != 0;

        uint8_t * frame = rgb;
        if (scale || rotate) {
            v->converted.resize((size_t) f.w * f.h * 3);
            frame = v->converted.data();
        }
        convert_frame(f, buf + info.offset, frame);
        if (scale) {
            uint8_t * out = rgb;
            if (rotate) {
                v->scaled.resize((size_t) scaled_w * scaled_h * 3);
                out = v->scaled.data();
            }
            resize_rgb(frame, f.w, f.h, out, scaled_w, scaled_h, v->resize_tmp);
            frame = out;
        }
        if (rotate) {
            rotate_rgb(frame, scaled_w, scaled_h, rgb, v->rotation);
        }
    }
    AMediaCodec_releaseOutputBuffer(v->video.codec, (size_t) idx, false);
    return true;
}

// the whole soundtrack from the first video frame on, mixed down to mono and resampled
void load_audio(android_video * v, int sample_rate) {
    track_decoder d;
    if (!d.open(v->source, "audio/") || !d.start()) {
        return;
    }
    std::vector<float> mono;
    int32_t rate     = 0;
    int32_t channels = 1;
    int64_t first_us = -1;
    AMediaCodecBufferInfo info;
    ssize_t idx;
    while ((idx = d.next_output(info)) >= 0) {
        size_t size = 0;
        const uint8_t * buf = AMediaCodec_getOutputBuffer(d.codec, (size_t) idx, &size);
        if (buf && info.size > 0) {
            rate     = d.output_int(AMEDIAFORMAT_KEY_SAMPLE_RATE, 0);
            channels = std::max(1, d.output_int(AMEDIAFORMAT_KEY_CHANNEL_COUNT, 1));
            if (first_us < 0) {
                first_us = info.presentationTimeUs;
            }
            const bool   is_float = d.output_int("pcm-encoding", 0) == PCM_ENCODING_FLOAT;
            const size_t n = (size_t) info.size / (is_float ? sizeof(float) : sizeof(int16_t)) / channels;
            const uint8_t * p = buf + info.offset;
            for (size_t i = 0; i < n; i++) {
                float sum = 0.0f;
                for (int32_t c = 0; c < channels; c++) {
                    if (is_float) {
                        float s;
                        memcpy(&s, p, sizeof(s));
                        p += sizeof(s);
                        sum += s;
                    } else {
                        int16_t s;
                        memcpy(&s, p, sizeof(s));
                        p += sizeof(s);
                        sum += s / 32768.0f;
                    }
                }
                mono.push_back(sum / channels);
            }
        }
        AMediaCodec_releaseOutputBuffer(d.codec, (size_t) idx, false);
    }
    if (mono.empty() || rate <= 0) {
        return;
    }
    // the soundtrack may start before the video does
    const size_t lead = (size_t) std::max<int64_t>(0, (v->start_us - first_us) * rate / 1000000);
    if (lead >= mono.size()) {
        return;
    }
    mtmd_helper::resampler_ptr resampler(mtmd_helper_resampler_init(rate, sample_rate));
    const float * out = nullptr;
    const int64_t n_out = resampler ? mtmd_helper_resampler_process(resampler.get(), mono.data() + lead, mono.size() - lead, &out) : -1;
    if (n_out > 0) {
        v->audio.assign(out, out + n_out);
    }
}

const float * video_read_audio(void * state, int sample_rate, size_t * n_samples) {
    auto * v = static_cast<android_video *>(state);
    v->audio.clear();
    load_audio(v, sample_rate);
    *n_samples = v->audio.size();
    return v->audio.empty() ? nullptr : v->audio.data();
}

void video_close(void * state) {
    delete static_cast<android_video *>(state);
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

#elif !defined(__APPLE__)

namespace rnllama {

const mtmd_helper_video_decoder * video_decoder() {
    return nullptr;
}

} // namespace rnllama

#endif
