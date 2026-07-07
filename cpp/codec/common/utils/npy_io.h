#ifndef CODEC_EXAMPLES_UTILS_NPY_IO_H
#define CODEC_EXAMPLES_UTILS_NPY_IO_H

#include <cstdint>
#include <string>
#include <vector>

enum codec_example_npy_dtype {
    CODEC_EXAMPLE_NPY_DTYPE_UNKNOWN = 0,
    CODEC_EXAMPLE_NPY_DTYPE_I32 = 1,
    CODEC_EXAMPLE_NPY_DTYPE_F32 = 2,
};

struct codec_example_npy_array {
    enum codec_example_npy_dtype dtype = CODEC_EXAMPLE_NPY_DTYPE_UNKNOWN;
    std::vector<int32_t> shape;
    std::vector<int32_t> i32;
    std::vector<float> f32;
};

bool codec_example_load_npy(const char * path, codec_example_npy_array * out, std::string * err);
bool codec_example_load_npy_i32_2d_tq(const char * path, std::vector<int32_t> * out, int32_t * n_q, int32_t * n_frames, std::string * err);
bool codec_example_load_npy_f32_2d(const char * path, std::vector<float> * out, int32_t * n_rows, int32_t * n_cols, std::string * err);
bool codec_example_save_npy_i32_2d_qt(const char * path, const int32_t * data_tq, int32_t n_q, int32_t n_frames, std::string * err);
bool codec_example_save_npy_f32_1d(const char * path, const float * data, int32_t n, std::string * err);
bool codec_example_save_npy_i32_1d(const char * path, const int32_t * data, int32_t n, std::string * err);

#endif
