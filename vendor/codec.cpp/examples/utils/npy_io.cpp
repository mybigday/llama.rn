#include "npy_io.h"

#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static bool codec_example_parse_shape_dims(const std::string & header, std::vector<int32_t> * out_shape, std::string * err) {
    const size_t p0 = header.find('(');
    const size_t p1 = header.find(')', p0 == std::string::npos ? 0 : p0 + 1);
    if (p0 == std::string::npos || p1 == std::string::npos || p1 <= p0 + 1) {
        if (err != nullptr) {
            *err = "invalid npy shape";
        }
        return false;
    }

    out_shape->clear();
    const std::string inside = header.substr(p0 + 1, p1 - p0 - 1);

    size_t i = 0;
    while (i < inside.size()) {
        while (i < inside.size() && (std::isspace((unsigned char) inside[i]) || inside[i] == ',')) {
            ++i;
        }
        if (i >= inside.size()) {
            break;
        }

        size_t j = i;
        while (j < inside.size() && std::isdigit((unsigned char) inside[j])) {
            ++j;
        }
        if (j == i) {
            if (err != nullptr) {
                *err = "invalid npy shape contents";
            }
            return false;
        }

        const std::string token = inside.substr(i, j - i);
        char * end = nullptr;
        long v = std::strtol(token.c_str(), &end, 10);
        if (end == token.c_str() || *end != '\0' || v <= 0 || v > INT32_MAX) {
            if (err != nullptr) {
                *err = "invalid npy dimensions";
            }
            return false;
        }

        out_shape->push_back((int32_t) v);
        i = j;
    }

    if (out_shape->empty()) {
        if (err != nullptr) {
            *err = "npy shape is empty";
        }
        return false;
    }

    return true;
}

static bool codec_example_parse_npy_header(std::ifstream & ifs, enum codec_example_npy_dtype * dtype, std::vector<int32_t> * shape, std::string * err) {
    char magic[6] = { 0 };
    if (!ifs.read(magic, 6) || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        if (err != nullptr) {
            *err = "invalid npy magic";
        }
        return false;
    }

    uint8_t major = 0;
    uint8_t minor = 0;
    if (!ifs.read(reinterpret_cast<char *>(&major), 1) || !ifs.read(reinterpret_cast<char *>(&minor), 1)) {
        if (err != nullptr) {
            *err = "invalid npy version";
        }
        return false;
    }

    uint32_t hlen = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        if (!ifs.read(reinterpret_cast<char *>(&h16), 2)) {
            if (err != nullptr) {
                *err = "invalid npy header length";
            }
            return false;
        }
        hlen = h16;
    } else if (major == 2 || major == 3) {
        if (!ifs.read(reinterpret_cast<char *>(&hlen), 4)) {
            if (err != nullptr) {
                *err = "invalid npy header length";
            }
            return false;
        }
    } else {
        if (err != nullptr) {
            *err = "unsupported npy version";
        }
        return false;
    }

    std::string header(hlen, '\0');
    if (!ifs.read(header.data(), (std::streamsize) hlen)) {
        if (err != nullptr) {
            *err = "failed to read npy header";
        }
        return false;
    }

    if (header.find("fortran_order") == std::string::npos || header.find("False") == std::string::npos) {
        if (err != nullptr) {
            *err = "fortran-order npy is not supported";
        }
        return false;
    }

    if (header.find("'descr': '<i4'") != std::string::npos || header.find("\"descr\": \"<i4\"") != std::string::npos) {
        *dtype = CODEC_EXAMPLE_NPY_DTYPE_I32;
    } else if (header.find("'descr': '<f4'") != std::string::npos || header.find("\"descr\": \"<f4\"") != std::string::npos) {
        *dtype = CODEC_EXAMPLE_NPY_DTYPE_F32;
    } else {
        if (err != nullptr) {
            *err = "only little-endian int32/float32 npy is supported";
        }
        return false;
    }

    return codec_example_parse_shape_dims(header, shape, err);
}

bool codec_example_load_npy(const char * path, codec_example_npy_array * out, std::string * err) {
    if (path == nullptr || out == nullptr) {
        if (err != nullptr) {
            *err = "invalid npy load arguments";
        }
        return false;
    }

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        if (err != nullptr) {
            *err = "failed to open npy file";
        }
        return false;
    }

    out->dtype = CODEC_EXAMPLE_NPY_DTYPE_UNKNOWN;
    out->shape.clear();
    out->i32.clear();
    out->f32.clear();

    if (!codec_example_parse_npy_header(ifs, &out->dtype, &out->shape, err)) {
        return false;
    }

    size_t n_elem = 1;
    for (size_t i = 0; i < out->shape.size(); ++i) {
        if (out->shape[i] <= 0) {
            if (err != nullptr) {
                *err = "npy dimensions must be positive";
            }
            return false;
        }
        if (n_elem > SIZE_MAX / (size_t) out->shape[i]) {
            if (err != nullptr) {
                *err = "npy element count overflow";
            }
            return false;
        }
        n_elem *= (size_t) out->shape[i];
    }

    if (out->dtype == CODEC_EXAMPLE_NPY_DTYPE_I32) {
        out->i32.resize(n_elem);
        if (!ifs.read(reinterpret_cast<char *>(out->i32.data()), (std::streamsize) (n_elem * sizeof(int32_t)))) {
            if (err != nullptr) {
                *err = "failed to read npy int32 payload";
            }
            return false;
        }
    } else if (out->dtype == CODEC_EXAMPLE_NPY_DTYPE_F32) {
        out->f32.resize(n_elem);
        if (!ifs.read(reinterpret_cast<char *>(out->f32.data()), (std::streamsize) (n_elem * sizeof(float)))) {
            if (err != nullptr) {
                *err = "failed to read npy float32 payload";
            }
            return false;
        }
    } else {
        if (err != nullptr) {
            *err = "unsupported npy dtype";
        }
        return false;
    }

    return true;
}

bool codec_example_load_npy_i32_2d_tq(const char * path, std::vector<int32_t> * out, int32_t * n_q, int32_t * n_frames, std::string * err) {
    codec_example_npy_array arr;
    if (!codec_example_load_npy(path, &arr, err)) {
        return false;
    }
    if (arr.dtype != CODEC_EXAMPLE_NPY_DTYPE_I32 || arr.shape.size() != 2) {
        if (err != nullptr) {
            *err = "expected 2D int32 npy";
        }
        return false;
    }

    *n_q = arr.shape[0];
    *n_frames = arr.shape[1];
    out->assign(arr.i32.size(), 0);
    for (int32_t q = 0; q < *n_q; ++q) {
        for (int32_t t = 0; t < *n_frames; ++t) {
            (*out)[(size_t) t * (size_t) (*n_q) + (size_t) q] =
                arr.i32[(size_t) q * (size_t) (*n_frames) + (size_t) t];
        }
    }
    return true;
}

bool codec_example_load_npy_f32_2d(const char * path, std::vector<float> * out, int32_t * n_rows, int32_t * n_cols, std::string * err) {
    codec_example_npy_array arr;
    if (!codec_example_load_npy(path, &arr, err)) {
        return false;
    }
    if (arr.dtype != CODEC_EXAMPLE_NPY_DTYPE_F32 || arr.shape.size() != 2) {
        if (err != nullptr) {
            *err = "expected 2D float32 npy";
        }
        return false;
    }

    *n_rows = arr.shape[0];
    *n_cols = arr.shape[1];
    *out = std::move(arr.f32);
    return true;
}

bool codec_example_save_npy_i32_2d_qt(const char * path, const int32_t * data_tq, int32_t n_q, int32_t n_frames, std::string * err) {
    if (path == nullptr || data_tq == nullptr || n_q <= 0 || n_frames <= 0) {
        if (err != nullptr) {
            *err = "invalid npy save args";
        }
        return false;
    }

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        if (err != nullptr) {
            *err = "failed to open output npy file";
        }
        return false;
    }

    const char magic[] = "\x93NUMPY";
    ofs.write(magic, 6);
    const uint8_t major = 1;
    const uint8_t minor = 0;
    ofs.put((char) major);
    ofs.put((char) minor);

    char header[256];
    std::snprintf(
        header,
        sizeof(header),
        "{'descr': '<i4', 'fortran_order': False, 'shape': (%d, %d), }",
        n_q,
        n_frames);
    std::string hdr = header;
    const size_t preamble = 6 + 2 + 2;
    const size_t total = preamble + hdr.size();
    const size_t pad = (16 - (total % 16)) % 16;
    hdr.append(pad, ' ');
    hdr.push_back('\n');

    const uint16_t hlen = (uint16_t) hdr.size();
    ofs.write(reinterpret_cast<const char *>(&hlen), sizeof(hlen));
    ofs.write(hdr.data(), (std::streamsize) hdr.size());

    for (int32_t q = 0; q < n_q; ++q) {
        for (int32_t t = 0; t < n_frames; ++t) {
            const int32_t v = data_tq[(size_t) t * (size_t) n_q + (size_t) q];
            ofs.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
    }

    if (!ofs.good()) {
        if (err != nullptr) {
            *err = "failed to write npy data";
        }
        return false;
    }

    return true;
}

static bool codec_example_save_npy_1d(
    const char * path, const void * data, size_t elem_size,
    int32_t n, const char * descr, std::string * err) {

    if (path == nullptr || data == nullptr || n < 0) {
        if (err != nullptr) *err = "invalid npy save args";
        return false;
    }
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        if (err != nullptr) *err = "failed to open output npy file";
        return false;
    }

    const char magic[] = "\x93NUMPY";
    ofs.write(magic, 6);
    ofs.put((char) 1);  // major
    ofs.put((char) 0);  // minor

    char header[160];
    std::snprintf(header, sizeof(header),
                  "{'descr': '%s', 'fortran_order': False, 'shape': (%d,), }",
                  descr, n);
    std::string hdr = header;
    const size_t preamble = 6 + 2 + 2;
    const size_t total = preamble + hdr.size();
    const size_t pad = (16 - (total % 16)) % 16;
    hdr.append(pad, ' ');
    hdr.push_back('\n');

    const uint16_t hlen = (uint16_t) hdr.size();
    ofs.write(reinterpret_cast<const char *>(&hlen), sizeof(hlen));
    ofs.write(hdr.data(), (std::streamsize) hdr.size());
    if (n > 0) {
        ofs.write(reinterpret_cast<const char *>(data), (std::streamsize)((size_t) n * elem_size));
    }
    if (!ofs.good()) {
        if (err != nullptr) *err = "failed to write npy data";
        return false;
    }
    return true;
}

bool codec_example_save_npy_f32_1d(const char * path, const float * data, int32_t n, std::string * err) {
    return codec_example_save_npy_1d(path, data, sizeof(float), n, "<f4", err);
}

bool codec_example_save_npy_i32_1d(const char * path, const int32_t * data, int32_t n, std::string * err) {
    return codec_example_save_npy_1d(path, data, sizeof(int32_t), n, "<i4", err);
}
