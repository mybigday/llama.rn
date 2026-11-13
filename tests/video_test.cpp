// The platform video decoder behind video_url (cpp/rn-video.h), without a model.
//
// fixtures/video.mp4 (see fixtures/make_video.swift) is stored rotated and has a
// soundtrack, so the test covers what a phone recording needs: the displayed
// orientation, the exact frame count, scaling, skipped frames, a video given by
// its content instead of its path, and the soundtrack at the requested rate.
// fixtures/video-offset.mp4 is the same video starting half a second after its
// soundtrack: the time before the first frame is not part of the video.
// It also checks how mtmd picks frames by index for a decoder that has no fps filter.
//
// Usage:
//   ./video_test [video.mp4 [video-offset.mp4]]     # default: ../fixtures/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "rn-video.h"

#if defined(__ANDROID__)
#include <dlfcn.h>

// MediaCodec calls back over binder. An app has binder threads to take the calls,
// a bare executable started from adb shell has to start them itself (API 29).
static void start_binder_threads() {
    void * lib = dlopen("libbinder_ndk.so", RTLD_NOW);
    auto * start = lib ? (void (*)()) dlsym(lib, "ABinderProcess_startThreadPool") : nullptr;
    if (start) {
        start();
    }
}
#else
static void start_binder_threads() {}
#endif

static int n_failed = 0;

static void check(const std::string & name, bool ok) {
    std::cout << "TEST: " << name << " ... " << (ok ? "PASSED" : "FAILED") << std::endl;
    n_failed += ok ? 0 : 1;
}

struct rgb {
    int r, g, b;
};

static rgb pixel(const std::vector<unsigned char> & frame, uint32_t width, uint32_t x, uint32_t y) {
    const unsigned char * p = frame.data() + ((size_t) y * width + x) * 3;
    return {p[0], p[1], p[2]};
}

// lossy codec and color conversion: the fixture's colors are saturated, a wide margin is enough
static bool near(rgb a, rgb b) {
    return std::abs(a.r - b.r) < 60 && std::abs(a.g - b.g) < 60 && std::abs(a.b - b.b) < 60;
}

static const rgb WHITE = {255, 255, 255};
static const rgb BLACK = {0, 0, 0};
static const rgb BACKGROUND[3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}}; // frames 0-19, 20-39, 40-59

// the displayed layout of fixture frame idx, at any size
static bool is_fixture_frame(const std::vector<unsigned char> & frame, uint32_t w, uint32_t h, int idx) {
    return near(pixel(frame, w, w / 4, h / 4), WHITE) &&
           near(pixel(frame, w, w * 3 / 4, h * 3 / 4), BLACK) &&
           near(pixel(frame, w, w * 3 / 4, h / 4), BACKGROUND[idx / 20]) &&
           near(pixel(frame, w, w / 4, h * 3 / 4), BACKGROUND[idx / 20]);
}

struct fake_seq : mtmd_helper_video_seq {
    bool read_raw_frame() override { return true; }
};

static void test_sample_by_index() {
    mtmd_helper_video_src src;
    src.width    = 4;
    src.height   = 4;
    src.fps      = 30.0f;
    src.n_frames = 60;
    const mtmd_video_config no_preference = {MTMD_VIDEO_TEMPLATE_DEFAULT, 0.0f, 0, 0, 0};
    {
        fake_seq seq;
        seq.configure(no_preference, src, 4.0f);
        seq.sample_by_index(src);
        const std::vector<int32_t> expected = {0, 7, 15, 22, 30, 37, 45, 52};
        check("4 fps of a 2 s video at 30 fps: one frame every 7.5", seq.frame_indices == expected);
    }
    {
        fake_seq seq;
        seq.configure(no_preference, src, 60.0f);
        seq.sample_by_index(src);
        check("a target above the source fps repeats frames",
              seq.frame_indices.size() == 120 && seq.frame_indices[1] == 0 && seq.frame_indices[119] == 59);
    }
    {
        mtmd_helper_video_src unknown = src;
        unknown.n_frames = -1;
        fake_seq seq;
        seq.configure(no_preference, unknown, 4.0f);
        seq.sample_by_index(unknown);
        check("a video of unknown length keeps every frame", seq.frame_indices.empty() && seq.info.fps == 30.0f);
    }
}

static void test_decoder(const mtmd_helper_video_decoder * dec, const std::string & path) {
    mtmd_helper_video_stream_info info = {0, 0, 0.0f, -1, -1.0f};
    void * video = dec->open(dec->user_data, path.c_str(), nullptr, 0, &info);
    check("opens the video by its path", video != nullptr);
    if (!video) {
        return;
    }
    check("the size is the displayed one (rotated)", info.width == 120 && info.height == 160);
    check("the frame count is exact", info.n_frames == 60);
    check("fps and duration", std::fabs(info.fps - 30.0f) < 0.5f && std::fabs(info.duration - 2.0f) < 0.1f);

    std::vector<unsigned char> frame((size_t) info.width * info.height * 3);
    bool ok = dec->read_frame(video, frame.data(), info.width, info.height);
    check("frame 0 is upright", ok && is_fixture_frame(frame, info.width, info.height, 0));

    // skip frames 1-24, then read frame 25 at another size
    for (int i = 1; i < 25 && ok; i++) {
        ok = dec->read_frame(video, nullptr, info.width, info.height);
    }
    std::vector<unsigned char> small((size_t) 60 * 40 * 3);
    ok = ok && dec->read_frame(video, small.data(), 60, 40);
    check("frame 25 after skipped frames, scaled", ok && is_fixture_frame(small, 60, 40, 25));

    int n_read = 26;
    while (dec->read_frame(video, frame.data(), info.width, info.height)) {
        n_read++;
    }
    check("the stream ends after its last frame", n_read == 60 && is_fixture_frame(frame, info.width, info.height, 59));

    if (dec->read_audio) {
        size_t n = 0;
        const float * samples = dec->read_audio(video, 16000, &n);
        const std::vector<float> pcm(samples, samples + (samples ? n : 0));
        // a 440 Hz tone crosses zero 880 times a second
        int crossings = 0;
        for (size_t i = 1; i < pcm.size(); i++) {
            crossings += (pcm[i - 1] < 0.0f) != (pcm[i] < 0.0f);
        }
        const double seconds = pcm.size() / 16000.0;
        std::cout << "  soundtrack: " << pcm.size() << " samples, " << crossings << " zero crossings" << std::endl;
        check("the soundtrack spans the video at 16 kHz", std::fabs(seconds - 2.0) < 0.15);
        check("the soundtrack is the 440 Hz tone", seconds > 0.0 && std::fabs(crossings / seconds - 880.0) < 40.0);
    }
    dec->close(video);

    std::ifstream file(path, std::ios::binary);
    const std::vector<unsigned char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    info  = {0, 0, 0.0f, -1, -1.0f};
    video = dec->open(dec->user_data, nullptr, data.data(), data.size(), &info);
    check("opens the video by its content", video != nullptr && info.width == 120 && info.n_frames == 60);
    if (video) {
        ok = dec->read_frame(video, frame.data(), info.width, info.height);
        check("frame 0 of the content", ok && is_fixture_frame(frame, info.width, info.height, 0));
        dec->close(video);
    }

    const std::string garbage = "not a video";
    video = dec->open(dec->user_data, nullptr, (const unsigned char *) garbage.data(), garbage.size(), &info);
    check("rejects what is not a video", video == nullptr);
    if (video) {
        dec->close(video);
    }
}

// a video that starts later than its soundtrack: the decoder reports and reads the frames only,
// and the soundtrack from the first frame on
static void test_late_video(const mtmd_helper_video_decoder * dec, const std::string & path) {
    mtmd_helper_video_stream_info info = {0, 0, 0.0f, -1, -1.0f};
    void * video = dec->open(dec->user_data, path.c_str(), nullptr, 0, &info);
    check("opens the late video", video != nullptr);
    if (!video) {
        return;
    }
    check("the time before the first frame does not count",
          info.n_frames == 60 && std::fabs(info.fps - 30.0f) < 0.5f && std::fabs(info.duration - 2.0f) < 0.1f);

    std::vector<unsigned char> frame((size_t) info.width * info.height * 3);
    bool ok = dec->read_frame(video, frame.data(), info.width, info.height);
    check("the first frame read is frame 0", ok && is_fixture_frame(frame, info.width, info.height, 0));
    int n_read = 1;
    while (dec->read_frame(video, nullptr, info.width, info.height)) {
        n_read++;
    }
    check("no frame is made up for the time before", n_read == 60);

    if (dec->read_audio) {
        size_t n = 0;
        dec->read_audio(video, 16000, &n);
        std::cout << "  soundtrack: " << n << " samples" << std::endl;
        // the 2 s soundtrack ends 1.5 s after the first frame
        check("the soundtrack starts with the first frame", std::fabs(n / 16000.0 - 1.5) < 0.15);
    }
    dec->close(video);
}

int main(int argc, char ** argv) {
    const std::string path      = argc > 1 ? argv[1] : "../fixtures/video.mp4";
    const std::string late_path = argc > 2 ? argv[2] : "../fixtures/video-offset.mp4";

    start_binder_threads();
    test_sample_by_index();

    const mtmd_helper_video_decoder * dec = rnllama::video_decoder();
    if (dec) {
        test_decoder(dec, path);
        test_late_video(dec, late_path);
    } else {
        std::cout << "no platform video decoder here, decoder tests skipped" << std::endl;
    }

    std::cout << (n_failed == 0 ? "\nAll tests passed" : "\nSome tests FAILED") << std::endl;
    return n_failed == 0 ? 0 : 1;
}
