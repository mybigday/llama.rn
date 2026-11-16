// Generates video.mp4, the fixture of video_test.cpp:  swift make_video.swift video.mp4
//
// 60 frames of 160x120 H.264 at 30 fps, stored rotated: the track's transform turns them a quarter
// clockwise, to 120x160. As displayed, frames 0-19 are red, 20-39 green and 40-59 blue, each with a
// white top left quarter and a black bottom right quarter. The soundtrack is a 440 Hz tone (AAC).
//
// video-offset.mp4 is the same video starting half a second after its soundtrack (an empty edit
// before the first frame):
//   ffmpeg -itsoffset 0.5 -i video.mp4 -i video.mp4 -map 0:v -map 1:a -c copy video-offset.mp4
import AVFoundation

let path = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "video.mp4"
let url = URL(fileURLWithPath: path)
try? FileManager.default.removeItem(at: url)

let width = 160, height = 120, fps: Int32 = 30, nFrames = 60
let writer = try! AVAssetWriter(outputURL: url, fileType: .mp4)

let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
    AVVideoCodecKey: AVVideoCodecType.h264,
    AVVideoWidthKey: width,
    AVVideoHeightKey: height,
])
video.transform = CGAffineTransform(rotationAngle: .pi / 2)
let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: video, sourcePixelBufferAttributes: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
    kCVPixelBufferWidthKey as String: width,
    kCVPixelBufferHeightKey as String: height,
])
writer.add(video)

let sampleRate = 44100.0
let audio = AVAssetWriterInput(mediaType: .audio, outputSettings: [
    AVFormatIDKey: kAudioFormatMPEG4AAC,
    AVSampleRateKey: sampleRate,
    AVNumberOfChannelsKey: 1,
    AVEncoderBitRateKey: 64000,
])
writer.add(audio)

writer.startWriting()
writer.startSession(atSourceTime: .zero)

var asbd = AudioStreamBasicDescription(
    mSampleRate: sampleRate, mFormatID: kAudioFormatLinearPCM,
    mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
    mBytesPerPacket: 2, mFramesPerPacket: 1, mBytesPerFrame: 2, mChannelsPerFrame: 1, mBitsPerChannel: 16, mReserved: 0)
var audioFormat: CMAudioFormatDescription?
CMAudioFormatDescriptionCreate(allocator: nil, asbd: &asbd, layoutSize: 0, layout: nil, magicCookieSize: 0,
                               magicCookie: nil, extensions: nil, formatDescriptionOut: &audioFormat)
let samplesPerFrame = Int(sampleRate) / Int(fps)

// BGRA
let colors: [[UInt8]] = [[0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]]

func videoFrame(_ i: Int) -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &buffer)
    CVPixelBufferLockBaseAddress(buffer!, [])
    let base = CVPixelBufferGetBaseAddress(buffer!)!.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(buffer!)
    for y in 0..<height {
        for x in 0..<width {
            // the displayed frame is this one turned a quarter clockwise
            var c = colors[i / 20]
            if x < width / 2 && y >= height / 2 { c = [255, 255, 255, 255] } // displayed top left
            if x >= width / 2 && y < height / 2 { c = [0, 0, 0, 255] }       // displayed bottom right
            for k in 0..<4 { base[y * stride + x * 4 + k] = c[k] }
        }
    }
    CVPixelBufferUnlockBaseAddress(buffer!, [])
    return buffer!
}

// the share of the soundtrack that plays during frame i
func audioChunk(_ i: Int) -> CMSampleBuffer {
    let first = i * samplesPerFrame
    var pcm = (first..<first + samplesPerFrame).map { Int16(12000 * sin(2 * .pi * 440 * Double($0) / sampleRate)) }
    var block: CMBlockBuffer?
    CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: pcm.count * 2, blockAllocator: nil,
                                       customBlockSource: nil, offsetToData: 0, dataLength: pcm.count * 2, flags: 0,
                                       blockBufferOut: &block)
    CMBlockBufferReplaceDataBytes(with: &pcm, blockBuffer: block!, offsetIntoDestination: 0, dataLength: pcm.count * 2)
    var sample: CMSampleBuffer?
    CMAudioSampleBufferCreateReadyWithPacketDescriptions(
        allocator: nil, dataBuffer: block!, formatDescription: audioFormat!, sampleCount: pcm.count,
        presentationTimeStamp: CMTime(value: CMTimeValue(first), timescale: Int32(sampleRate)),
        packetDescriptions: nil, sampleBufferOut: &sample)
    return sample!
}

// the writer interleaves the tracks, so each one is fed when the writer asks for it
let fed = DispatchGroup()
var videoIndex = 0
fed.enter()
video.requestMediaDataWhenReady(on: DispatchQueue(label: "video")) {
    while video.isReadyForMoreMediaData && videoIndex < nFrames {
        adaptor.append(videoFrame(videoIndex), withPresentationTime: CMTime(value: CMTimeValue(videoIndex), timescale: fps))
        videoIndex += 1
    }
    if videoIndex == nFrames {
        videoIndex += 1
        video.markAsFinished()
        fed.leave()
    }
}
var audioIndex = 0
fed.enter()
audio.requestMediaDataWhenReady(on: DispatchQueue(label: "audio")) {
    while audio.isReadyForMoreMediaData && audioIndex < nFrames {
        audio.append(audioChunk(audioIndex))
        audioIndex += 1
    }
    if audioIndex == nFrames {
        audioIndex += 1
        audio.markAsFinished()
        fed.leave()
    }
}
fed.wait()

let done = DispatchSemaphore(value: 0)
writer.finishWriting { done.signal() }
done.wait()
print(writer.status == .completed ? "wrote \(path)" : "failed: \(String(describing: writer.error))")
