require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))
base_ld_flags = "-framework Accelerate -framework Foundation -framework Metal -framework MetalKit"
base_compiler_flags = "-fno-objc-arc -fvisibility-inlines-hidden -DGGML_USE_CPU -DGGML_USE_ACCELERATE -DGGML_USE_BLAS -DGGML_BLAS_USE_ACCELERATE -DGGML_USE_CPU_REPACK -Wno-shorten-64-to-32"

if ENV["RNLLAMA_DISABLE_METAL"] != "1" then
  # GGMLMetalClass only exists when GGML_METAL_EMBED_LIBRARY is off; keep the
  # (process-global) Objective-C class name unique per library regardless.
  base_compiler_flags += " -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY=1 -DGGMLMetalClass=RNLlamaGGMLMetalClass" # -DGGML_METAL_NDEBUG
end

# Use base_optimizer_flags = "" for debug builds
# base_optimizer_flags = ""
base_optimizer_flags = "-O3 -DNDEBUG -funroll-loops"

if ENV["RNLLAMA_NATIVE_CPU"] == "1" then
  apple_cpu_flags = ENV["RNLLAMA_NATIVE_CPU_FLAGS"] || "-Xarch_arm64 -mcpu=apple-m2"
  base_optimizer_flags += " -U__ARM_FEATURE_SVE -U__ARM_FEATURE_SME #{apple_cpu_flags}"
end

Pod::Spec.new do |s|
  s.name         = "llama-rn"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]

  s.platforms    = { :ios => "13.0", :tvos => "13.0" }
  s.source       = { :git => "https://github.com/mybigday/llama.rn.git", :tag => "#{s.version}" }

  header_search_paths = ['$(inherited)']

  llama_cpp = "vendor/llama.cpp"
  codec_cpp = "vendor/codec.cpp"

  if ENV["RNLLAMA_BUILD_FROM_SOURCE"] == "1"
    # ios/*: not ios/**, or the prebuilt xcframework's flat Headers/ would join
    # the header map and shadow the vendored headers with the same basenames.
    s.source_files = "ios/*.{h,m,mm}", "cpp/**/*.{h,cpp,hpp,c,mm}",
      "#{llama_cpp}/include/*.h",
      "#{llama_cpp}/src/**/*.{h,cpp}",
      "#{llama_cpp}/ggml/include/*.h",
      "#{llama_cpp}/ggml/src/*.{h,c,cpp}",
      "#{llama_cpp}/ggml/src/ggml-cpu/**/*.{h,c,cpp}",
      "#{llama_cpp}/ggml/src/ggml-metal/*.{h,m,cpp,s}",
      "#{llama_cpp}/ggml/src/ggml-blas/*.{h,cpp}",
      "#{llama_cpp}/common/**/*.{h,cpp}",
      "#{llama_cpp}/tools/mtmd/**/*.{h,cpp}",
      "#{llama_cpp}/vendor/**/*.{h,hpp,c,cpp}",
      "#{codec_cpp}/{include,src,common}/**/*.{h,cpp}"
    # Exclude what no llama.rn build compiles (same set as the filters in
    # cmake/rnllama-sources.cmake): the parts of vendor/llama.cpp that are
    # only there for upstream's CMake project (see vendor/README.md), the
    # mtmd debug CLI, and codec's reference runners which conflict with
    # rn-tts. common/jinja/string.h must stay out of the pod's header map, or
    # every <string.h> in the target resolves to it; jinja itself finds it
    # next to value.h and via the common/ search path.
    s.exclude_files = "#{llama_cpp}/src/llama-quant.cpp", "#{llama_cpp}/ggml/src/ggml.cpp",
      "#{llama_cpp}/common/{arg,console,debug,download,hf-cache,imatrix-loader,llguidance,preset,subproc}.cpp",
      "#{llama_cpp}/ggml/src/ggml-cpu/hbm.cpp",
      "#{llama_cpp}/ggml/src/ggml-cpu/{kleidiai,llamafile,arch/wasm}/*",
      "#{llama_cpp}/vendor/{cpp-httplib,hash/sha1,hash/xxhash}/*",
      "#{llama_cpp}/tools/mtmd/debug/*.cpp", "#{codec_cpp}/common/tts_runner*.cpp",
      "#{llama_cpp}/common/jinja/string.h"
    base_compiler_flags += " -DRNLLAMA_BUILD_FROM_SOURCE"
    # Same order as cmake/rnllama-sources.cmake (basename collisions are
    # resolved by order: common/ before src/ and ggml/src/ggml-cpu).
    [
      "cpp",
      llama_cpp,
      "#{llama_cpp}/include",
      "#{llama_cpp}/ggml/include",
      "#{llama_cpp}/ggml/src",
      "#{llama_cpp}/common",
      "#{llama_cpp}/src",
      "#{llama_cpp}/ggml/src/ggml-cpu",
      "#{llama_cpp}/tools/mtmd",
      "#{llama_cpp}/vendor",
      "#{llama_cpp}/vendor/hash",
      "#{codec_cpp}/include",
      "#{codec_cpp}/common",
    ].each { |dir| header_search_paths << "\"$(PODS_TARGET_SRCROOT)/#{dir}\"" }
  else
    # JSI bindings always compiled from source (must match RN version)
    s.source_files = "ios/*.{h,m,mm}", "cpp/jsi/**/*.{h,cpp,mm}"
    s.vendored_frameworks = "ios/rnllama.xcframework"
    base_compiler_flags += " -DRNLLAMA_USE_FRAMEWORK_HEADERS"
    # Header-only JSON dependency needed by JSI when using the prebuilt xcframework
    header_search_paths << "\"$(PODS_TARGET_SRCROOT)/#{llama_cpp}/vendor\""
  end

  s.preserve_paths = "#{llama_cpp}/vendor/nlohmann/**/*.hpp"

  s.compiler_flags = base_compiler_flags
  pod_target_xcconfig = {
    "OTHER_LDFLAGS" => base_ld_flags,
    "OTHER_CFLAGS" => base_optimizer_flags,
    "OTHER_CPLUSPLUSFLAGS" => base_optimizer_flags + " -std=c++20",
    "HEADER_SEARCH_PATHS" => header_search_paths.join(" ")
  }
  s.pod_target_xcconfig = pod_target_xcconfig

  s.dependency "React-callinvoker"
  s.dependency "React"

  install_modules_dependencies(s)
end
