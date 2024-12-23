#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include <algorithm>
#include <codecvt>
#include <cstring>
#include <filesystem>
#include <locale>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef LM_GGML_USE_CPU
#include "ggml-cpu.h"
#endif

#ifdef LM_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef LM_GGML_USE_METAL
#include <TargetConditionals.h>

#if !TARGET_OS_SIMULATOR
#include "ggml-metal.h"
#endif

#endif

#ifdef LM_GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#ifdef LM_GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef LM_GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif

#ifdef LM_GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#ifdef LM_GGML_USE_RPC
#include "ggml-rpc.h"
#endif

#ifdef LM_GGML_USE_CANN
#include "ggml-cann.h"
#endif

#ifdef LM_GGML_USE_KOMPUTE
#include "ggml-kompute.h"
#endif

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

struct dl_handle_deleter {
    void operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

static dl_handle * dl_load_library(const std::wstring & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.c_str());

    SetErrorMode(old_mode);

    return handle;
}

static dl_handle * dl_load_library(const std::string & path) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return dl_load_library(converter.from_bytes(path));
}

static void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    return p;
}

#else

using dl_handle = void;

struct dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

static void * dl_load_library(const std::string & path) {
    dl_handle * handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);

    return handle;
}

static void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

struct lm_ggml_backend_reg_entry {
    lm_ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

struct lm_ggml_backend_registry {
    std::vector<lm_ggml_backend_reg_entry> backends;
    std::vector<lm_ggml_backend_dev_t> devices;

    lm_ggml_backend_registry() {
#ifdef LM_GGML_USE_CUDA
        register_backend(lm_ggml_backend_cuda_reg());
#endif
#ifdef LM_GGML_USE_METAL

#if !TARGET_OS_SIMULATOR
        register_backend(lm_ggml_backend_metal_reg());
#endif

#endif
#ifdef LM_GGML_USE_SYCL
        register_backend(lm_ggml_backend_sycl_reg());
#endif
#ifdef LM_GGML_USE_VULKAN
        register_backend(lm_ggml_backend_vk_reg());
#endif
#ifdef LM_GGML_USE_OPENCL
        register_backend(lm_ggml_backend_opencl_reg());
#endif
#ifdef LM_GGML_USE_CANN
        register_backend(lm_ggml_backend_cann_reg());
#endif
#ifdef LM_GGML_USE_BLAS
        register_backend(lm_ggml_backend_blas_reg());
#endif
#ifdef LM_GGML_USE_RPC
        register_backend(lm_ggml_backend_rpc_reg());
#endif
#ifdef LM_GGML_USE_KOMPUTE
        register_backend(lm_ggml_backend_kompute_reg());
#endif
#ifdef LM_GGML_USE_CPU
        register_backend(lm_ggml_backend_cpu_reg());
#endif
    }

    ~lm_ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        for (auto & entry : backends) {
            if (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    void register_backend(lm_ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        if (!reg) {
            return;
        }

#ifndef NDEBUG
        LM_GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, lm_ggml_backend_reg_name(reg), lm_ggml_backend_reg_dev_count(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        for (size_t i = 0; i < lm_ggml_backend_reg_dev_count(reg); i++) {
            register_device(lm_ggml_backend_reg_dev_get(reg, i));
        }
    }

    void register_device(lm_ggml_backend_dev_t device) {
#ifndef NDEBUG
        LM_GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, lm_ggml_backend_dev_name(device), lm_ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    lm_ggml_backend_reg_t load_backend(const char * path, bool silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        if (!handle) {
            if (!silent) {
                LM_GGML_LOG_ERROR("%s: failed to load %s\n", __func__, path);
            }
            return nullptr;
        }

        auto score_fn = (lm_ggml_backend_score_t) dl_get_sym(handle.get(), "lm_ggml_backend_score");
        if (score_fn && score_fn() == 0) {
            if (!silent) {
                LM_GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path);
            }
            return nullptr;
        }

        auto backend_init_fn = (lm_ggml_backend_init_t) dl_get_sym(handle.get(), "lm_ggml_backend_init");
        if (!backend_init_fn) {
            if (!silent) {
                LM_GGML_LOG_ERROR("%s: failed to find lm_ggml_backend_init in %s\n", __func__, path);
            }
            return nullptr;
        }

        lm_ggml_backend_reg_t reg = backend_init_fn();
        if (!reg || reg->api_version != LM_GGML_BACKEND_API_VERSION) {
            if (!silent) {
                if (!reg) {
                    LM_GGML_LOG_ERROR("%s: failed to initialize backend from %s: lm_ggml_backend_init returned NULL\n", __func__, path);
                } else {
                    LM_GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path, reg->api_version, LM_GGML_BACKEND_API_VERSION);
                }
            }
            return nullptr;
        }

        LM_GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, lm_ggml_backend_reg_name(reg), path);

        register_backend(reg, std::move(handle));

        return reg;
    }

    void unload_backend(lm_ggml_backend_reg_t reg, bool silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](const lm_ggml_backend_reg_entry & entry) { return entry.reg == reg; });

        if (it == backends.end()) {
            if (!silent) {
                LM_GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            return;
        }

        if (!silent) {
            LM_GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, lm_ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](lm_ggml_backend_dev_t dev) { return lm_ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

static lm_ggml_backend_registry & get_reg() {
    static lm_ggml_backend_registry reg;
    return reg;
}

// Internal API
void lm_ggml_backend_register(lm_ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

void lm_ggml_backend_device_register(lm_ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
static bool striequals(const char * a, const char * b) {
    for (; *a && *b; a++, b++) {
        if (std::tolower(*a) != std::tolower(*b)) {
            return false;
        }
    }
    return *a == *b;
}

size_t lm_ggml_backend_reg_count() {
    return get_reg().backends.size();
}

lm_ggml_backend_reg_t lm_ggml_backend_reg_get(size_t index) {
    LM_GGML_ASSERT(index < lm_ggml_backend_reg_count());
    return get_reg().backends[index].reg;
}

lm_ggml_backend_reg_t lm_ggml_backend_reg_by_name(const char * name) {
    for (size_t i = 0; i < lm_ggml_backend_reg_count(); i++) {
        lm_ggml_backend_reg_t reg = lm_ggml_backend_reg_get(i);
        if (striequals(lm_ggml_backend_reg_name(reg), name)) {
            return reg;
        }
    }
    return nullptr;
}

// Device enumeration
size_t lm_ggml_backend_dev_count() {
    return get_reg().devices.size();
}

lm_ggml_backend_dev_t lm_ggml_backend_dev_get(size_t index) {
    LM_GGML_ASSERT(index < lm_ggml_backend_dev_count());
    return get_reg().devices[index];
}

lm_ggml_backend_dev_t lm_ggml_backend_dev_by_name(const char * name) {
    for (size_t i = 0; i < lm_ggml_backend_dev_count(); i++) {
        lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);
        if (striequals(lm_ggml_backend_dev_name(dev), name)) {
            return dev;
        }
    }
    return nullptr;
}

lm_ggml_backend_dev_t lm_ggml_backend_dev_by_type(enum lm_ggml_backend_dev_type type) {
    for (size_t i = 0; i < lm_ggml_backend_dev_count(); i++) {
        lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);
        if (lm_ggml_backend_dev_type(dev) == type) {
            return dev;
        }
    }
    return nullptr;
}

// Convenience functions
lm_ggml_backend_t lm_ggml_backend_init_by_name(const char * name, const char * params) {
    lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_by_name(name);
    if (!dev) {
        return nullptr;
    }
    return lm_ggml_backend_dev_init(dev, params);
}

lm_ggml_backend_t lm_ggml_backend_init_by_type(enum lm_ggml_backend_dev_type type, const char * params) {
    lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_by_type(type);
    if (!dev) {
        return nullptr;
    }
    return lm_ggml_backend_dev_init(dev, params);
}

lm_ggml_backend_t lm_ggml_backend_init_best(void) {
    lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_by_type(LM_GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        dev = lm_ggml_backend_dev_by_type(LM_GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        return nullptr;
    }
    return lm_ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
lm_ggml_backend_reg_t lm_ggml_backend_load(const char * path) {
    return get_reg().load_backend(path, false);
}

void lm_ggml_backend_unload(lm_ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, true);
}

static std::string get_executable_path() {
#if defined(__APPLE__)
    // get executable path
    std::vector<char> path;
    uint32_t size;
    while (true) {
        size = path.size();
        if (_NSGetExecutablePath(path.data(), &size) == 0) {
            break;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return base_path + "/";
#elif defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<char> path(1024);
    while (true) {
        // get executable path
#    if defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        if (len == -1) {
            break;
        }
        if (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            if (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            break;
        }
        path.resize(path.size() * 2);
    }

    return base_path + "/";
#elif defined(_WIN32)
    std::vector<char> path(MAX_PATH);
    DWORD len = GetModuleFileNameA(NULL, path.data(), path.size());
    if (len == 0) {
        return "";
    }
    std::string base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return base_path + "\\";
#endif
}

static std::string backend_filename_prefix() {
#ifdef _WIN32
    return "ggml-";
#else
    return "libggml-";
#endif
}

static std::string backend_filename_suffix() {
#ifdef _WIN32
    return ".dll";
#else
    return ".so";
#endif
}

static lm_ggml_backend_reg_t lm_ggml_backend_load_best(const char * name, bool silent, const char * user_search_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
     // TODO: search system paths
    std::string file_prefix = backend_filename_prefix() + name + "-";
    std::vector<std::string> search_paths;
    if (user_search_path == nullptr) {
        search_paths.push_back("./");
        search_paths.push_back(get_executable_path());
    } else {
#if defined(_WIN32)
        search_paths.push_back(std::string(user_search_path) + "\\");
#else
        search_paths.push_back(std::string(user_search_path) + "/");
#endif
    }

    int best_score = 0;
    std::string best_path;

    namespace fs = std::filesystem;
    for (const auto & search_path : search_paths) {
        if (!fs::exists(search_path)) {
            continue;
        }
        fs::directory_iterator dir_it(search_path, fs::directory_options::skip_permission_denied);
        for (const auto & entry : dir_it) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string ext = entry.path().extension().string();
                if (filename.find(file_prefix) == 0 && ext == backend_filename_suffix()) {
                    dl_handle_ptr handle { dl_load_library(entry.path().c_str()) };
                    if (!handle && !silent) {
                        LM_GGML_LOG_ERROR("%s: failed to load %s\n", __func__, entry.path().string().c_str());
                    }
                    if (handle) {
                        auto score_fn = (lm_ggml_backend_score_t) dl_get_sym(handle.get(), "lm_ggml_backend_score");
                        if (score_fn) {
                            int s = score_fn();
#ifndef NDEBUG
                            LM_GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, entry.path().string().c_str(), s);
#endif
                            if (s > best_score) {
                                best_score = s;
                                best_path = entry.path().string();
                            }
                        } else {
                            if (!silent) {
                                LM_GGML_LOG_INFO("%s: failed to find lm_ggml_backend_score in %s\n", __func__, entry.path().string().c_str());
                            }
                        }
                    }
                }
            }
        }
    }

    if (best_score == 0) {
        // try to load the base backend
        for (const auto & search_path : search_paths) {
            std::string path = search_path + backend_filename_prefix() + name + backend_filename_suffix();
            if (fs::exists(path)) {
                return get_reg().load_backend(path.c_str(), silent);
            }
        }
        return nullptr;
    }

    return get_reg().load_backend(best_path.c_str(), silent);
}

void lm_ggml_backend_load_all() {
    lm_ggml_backend_load_all_from_path(nullptr);
}

void lm_ggml_backend_load_all_from_path(const char * dir_path) {
#ifdef NDEBUG
    bool silent = true;
#else
    bool silent = false;
#endif

    lm_ggml_backend_load_best("blas", silent, dir_path);
    lm_ggml_backend_load_best("cann", silent, dir_path);
    lm_ggml_backend_load_best("cuda", silent, dir_path);
    lm_ggml_backend_load_best("hip", silent, dir_path);
    lm_ggml_backend_load_best("kompute", silent, dir_path);
    lm_ggml_backend_load_best("metal", silent, dir_path);
    lm_ggml_backend_load_best("rpc", silent, dir_path);
    lm_ggml_backend_load_best("sycl", silent, dir_path);
    lm_ggml_backend_load_best("vulkan", silent, dir_path);
    lm_ggml_backend_load_best("opencl", silent, dir_path);
    lm_ggml_backend_load_best("musa", silent, dir_path);
    lm_ggml_backend_load_best("cpu", silent, dir_path);
}
