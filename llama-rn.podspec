require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))
base_ld_flags = "-framework Accelerate -framework Foundation -framework Metal -framework MetalKit"
base_compiler_flags = "-fno-objc-arc -DLM_GGML_USE_CPU -DLM_GGML_USE_ACCELERATE -DLM_GGML_USE_BLAS -DLM_GGML_BLAS_USE_ACCELERATE -Wno-shorten-64-to-32"

if ENV["RNLLAMA_DISABLE_METAL"] != "1" then
  base_compiler_flags += " -DLM_GGML_USE_METAL" # -DLM_GGML_METAL_NDEBUG
end

# Use base_optimizer_flags = "" for debug builds
# base_optimizer_flags = ""
base_optimizer_flags = "-O3 -DNDEBUG"

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

  if ENV["RNLLAMA_BUILD_FROM_SOURCE"] == "1"
    s.source_files = "ios/**/*.{h,m,mm}", "cpp/**/*.{h,cpp,hpp,c,m,mm}"
    s.exclude_files = "cpp/ggml-opencl/*.{c,cpp}", "cpp/ggml-hexagon/**/*.{c,cpp}"
    s.resources = "cpp/ggml-metal/ggml-metal.metal"
    base_compiler_flags += " -DRNLLAMA_BUILD_FROM_SOURCE"
    header_search_paths << '"$(PODS_TARGET_SRCROOT)/cpp"'
    header_search_paths << '"${PODS_TARGET_SRCROOT}/cpp/common"'
  else
    # JSI bindings always compiled from source (must match RN version)
    s.source_files = "ios/*.{h,m,mm}", "cpp/jsi/**/*.{h,cpp}"
    s.vendored_frameworks = "ios/rnllama.xcframework"
    base_compiler_flags += " -DRNLLAMA_USE_FRAMEWORK_HEADERS"
  end

  # Header-only JSON dependency needed by JSI when using the prebuilt xcframework
  s.preserve_paths = "cpp/nlohmann/**/*.{h,hpp}"

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
