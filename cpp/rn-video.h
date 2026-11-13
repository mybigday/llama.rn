#ifndef RN_VIDEO_H
#define RN_VIDEO_H

#include "mtmd-helper.h"

namespace rnllama {

// The platform's video decoder for mtmd: AVFoundation on Apple platforms, MediaCodec on Android.
// nullptr on the other platforms, where mtmd can only decode video with ffmpeg (MTMD_VIDEO).
const mtmd_helper_video_decoder * video_decoder();

// mtmd_helper_init_opt_default() that decodes video with video_decoder()
inline mtmd_helper_init_opt media_init_opt() {
    mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
    opt.video_params.decoder = video_decoder();
    return opt;
}

// The frame rate a decoder reports: the mean rate of the counted frames, which is also right for
// a variable frame rate; the container's nominal rate if the count or the duration is unknown.
inline float video_stream_fps(float nominal_fps, int32_t n_frames, double duration) {
    return n_frames > 0 && duration > 0.0 ? (float) (n_frames / duration) : nominal_fps;
}

} // namespace rnllama

#endif // RN_VIDEO_H
