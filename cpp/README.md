# llama.rn C++ sources

Only llama.rn's own native code lives here. llama.cpp, codec.cpp and the
other dependencies are vendored under `vendor/` in their upstream layout
(see `vendor/README.md`), and the builds compile them from there.

- `rn-llama.*`: context wrapper and lifecycle
- `rn-completion.*`: legacy completion flow
- `rn-slot.*`, `rn-slot-manager.*`: parallel decoding/queueing
- `rn-mtmd.hpp`: multimodal (vision/audio) helpers
- `rn-tts.*`: TTS/vocoder integration
- `rn-common.hpp`: shared helpers (tokenization, rerank formatting, etc.)
- `anyascii.*`: ASCII transliteration used by TTS
- `jsi/`: the JSI bindings that expose the above to JavaScript. Platform glue
  (`ios/RNLlama.mm`, `android/src/main/RNLlamaJSI.cpp`) installs them on app
  startup.

The source and include lists every CMake build uses are in
`cmake/rnllama-sources.cmake`; the CocoaPods equivalent is in `llama-rn.podspec`.

To change llama.cpp behavior, edit the vendored file and regenerate its patch
with `scripts/update-patch.sh` (see `vendor/README.md`).
