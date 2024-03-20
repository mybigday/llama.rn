require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))
base_ld_flags = "-framework Accelerate -framework Foundation -framework Metal -framework MetalKit"
base_compiler_flags = "-fno-objc-arc -DLM_GGML_USE_ACCELERATE -Wno-shorten-64-to-32"
folly_compiler_flags = "-DFOLLY_NO_CONFIG -DFOLLY_MOBILE=1 -DFOLLY_USE_LIBCPP=1 -Wno-comma"

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

  s.platforms    = { :ios => "11.0", :tvos => "11.0" }
  s.source       = { :git => "https://github.com/mybigday/llama.rn.git", :tag => "#{s.version}" }

  s.source_files = "ios/**/*.{h,m,mm}", "cpp/**/*.{h,cpp,hpp,c,m,mm}"
  s.resources = "cpp/**/*.{metallib}"

  s.dependency "React-Core"

  s.compiler_flags = base_compiler_flags
  s.pod_target_xcconfig = {
    "OTHER_LDFLAGS" => base_ld_flags,
    "OTHER_CFLAGS" => base_optimizer_flags,
    "OTHER_CPLUSPLUSFLAGS" => base_optimizer_flags
  }

  # Don't install the dependencies when we run `pod install` in the old architecture.
  if ENV['RCT_NEW_ARCH_ENABLED'] == '1' then
    s.compiler_flags = base_compiler_flags + " " + folly_compiler_flags + " -DRCT_NEW_ARCH_ENABLED=1"
    new_arch_cpp_flags = "-DFOLLY_NO_CONFIG -DFOLLY_MOBILE=1 -DFOLLY_USE_LIBCPP=1"
    s.pod_target_xcconfig = {
      "CLANG_CXX_LANGUAGE_STANDARD" => "c++17",
      "HEADER_SEARCH_PATHS" => "\"$(PODS_ROOT)/boost\"",
      "OTHER_LDFLAGS" => "-framework Accelerate",
      "OTHER_CFLAGS" => base_optimizer_flags,
      "OTHER_CPLUSPLUSFLAGS" => new_arch_cpp_flags + " " + base_optimizer_flags
    }
    s.dependency "React-Codegen"
    s.dependency "RCT-Folly"
    s.dependency "RCTRequired"
    s.dependency "RCTTypeSafety"
    s.dependency "ReactCommon/turbomodule/core"
  end
end
