require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))
base_ld_flags = "-framework Accelerate -framework Foundation -framework Metal -framework MetalKit"
base_compiler_flags = "-fno-objc-arc -DLM_GGML_USE_CPU -DLM_GGML_USE_ACCELERATE -Wno-shorten-64-to-32"

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

  if ENV["RNLLAMA_BUILD_FROM_SOURCE"] == "1"
    s.source_files = "ios/**/*.{h,m,mm}", "cpp/**/*.{h,cpp,hpp,c,m,mm}"
    s.resources = "cpp/**/*.{metallib}"
    base_compiler_flags += " -DRNLLAMA_BUILD_FROM_SOURCE"
  else
    s.source_files = "ios/**/*.{h,m,mm}"
    s.vendored_frameworks = "ios/rnllama.xcframework"
  end

  s.dependency "React-Core"

  s.compiler_flags = base_compiler_flags
  s.pod_target_xcconfig = {
    "OTHER_LDFLAGS" => base_ld_flags,
    "OTHER_CFLAGS" => base_optimizer_flags,
    "OTHER_CPLUSPLUSFLAGS" => base_optimizer_flags + " -std=c++17"
  }

  # Don't install the dependencies when we run `pod install` in the old architecture.
  if ENV['RCT_NEW_ARCH_ENABLED'] == '1' then
    install_modules_dependencies(s)
  end
end
