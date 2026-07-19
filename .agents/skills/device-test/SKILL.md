---
name: device-test
description: Build and run the llama.rn example app on real hardware to validate native (cpp/) changes end-to-end - iOS-on-Mac ("Designed for iPad" on Apple Silicon, real Metal GPU) and Android on a connected Snapdragon device with ggml-hexagon (HTP/NPU) enabled. Use this whenever the user wants to test on a real device or real GPU/NPU, says "run on my mac", "test on the phone/device", mentions hexagon, HTP, NPU, Metal GPU testing, or after C++ changes when unit tests pass but on-device verification is needed. Both flows compile the C++ from source, so cpp/ edits take effect.
---

# Real-device test process (example app)

Both example builds compile llama.rn's C++ from source (`RNLLAMA_BUILD_FROM_SOURCE=1`
in `example/ios/Podfile`, `rnllamaBuildFromSource=true` in
`example/android/gradle.properties`), so changes under `cpp/` are exercised for
real. First builds are long (iOS ~5-10 min, Android ~20 min including the
Hexagon HTP Docker build); incremental rebuilds are much faster. Run long
builds in the background and verify artifacts when they finish.

## Shared prerequisite: Metro

The Debug apps load JS from Metro. Start it detached and confirm it's up:

```bash
cd example && (yarn start > /tmp/metro.log 2>&1 &)
curl -s http://localhost:8081/status   # -> packager-status:running
```

## iOS on Mac (real Metal GPU)

"Designed for iPad" runs the arm64 **iphoneos** binary natively on Apple
Silicon - no simulator, real Metal GPU. The build itself is standard; the
install goes through an ipa because LaunchServices refuses to `open` a raw
iphoneos `.app` ("incorrect executable format") and direct exec dies silently.

### 1. Find the Mac destination id

```bash
cd example/ios
xcodebuild -showdestinations -workspace RNLlamaExample.xcworkspace \
  -scheme RNLlamaExample 2>/dev/null | grep platform:macOS
# { platform:macOS, arch:arm64, variant:Designed for [iPad,iPhone], id:<MAC_UDID>, name:My Mac }
```

Use the `id`. Do NOT pass the variant string in `-destination` - xcodebuild
splits destination specs on commas, so `Designed for [iPad,iPhone]` fails with
exit 64 ("unreadable input 'iPhone]'").

### 2. Build

```bash
cd example/ios
xcodebuild -workspace RNLlamaExample.xcworkspace -scheme RNLlamaExample \
  -configuration Debug -destination "platform=macOS,arch=arm64,id=<MAC_UDID>" \
  -quiet CC=clang CPLUSPLUS=clang++ LD=clang LDPLUSPLUS=clang++ \
  GCC_OPTIMIZATION_LEVEL=0 COMPILER_INDEX_STORE_ENABLE=NO build
```

Judge success by the exit code, not by grepping for "error:" - Obj-C selector
names like `stat:path error:` in deprecation warnings produce false matches.
The product lands in DerivedData under `Build/Products/Debug-iphoneos/`; locate
it with:

```bash
xcodebuild -showBuildSettings -workspace RNLlamaExample.xcworkspace \
  -scheme RNLlamaExample -configuration Debug \
  -destination "platform=macOS,arch=arm64,id=<MAC_UDID>" 2>/dev/null \
  | grep -m1 ' BUILT_PRODUCTS_DIR'
```

### 3. Install via ipa and launch

Every `open <ipa>` install creates a NEW numbered copy in /Applications
(`RNLlamaExample 2.app`, ` 3.app`, ...) - it never replaces an existing one,
even when the app is not running. The bundles are root-owned so `rm` fails;
Finder can trash them without an admin prompt. So: quit the app and trash
stale dev copies first, keeping any non-dev install (e.g. TestFlight) alone:

```bash
pkill -x RNLlamaExample 2>/dev/null
osascript -e 'tell application "Finder" to delete POSIX file "/Applications/RNLlamaExample 2.app"' 2>/dev/null
# repeat for any " 3", " 4", ... copies from earlier installs
```

Then package and install:

```bash
cd "$BUILT_PRODUCTS_DIR"  # or work in a scratch dir with the .app copied in
rm -rf Payload dev.ipa && mkdir Payload
cp -R RNLlamaExample.app Payload/
zip -qry dev.ipa Payload
open dev.ipa               # macOS installs it silently into /Applications
```

The install is async - poll until the new copy appears, confirm it is this
build by binary mtime, and launch it:

```bash
stat -f "%Sm %N" /Applications/RNLlamaExample*.app/Wrapper/RNLlamaExample.app/RNLlamaExample
open "/Applications/RNLlamaExample 2.app"   # the copy whose mtime matches the build
ps aux | grep "[R]NLlamaExample"            # process alive = launched
```

### 4. Verify GPU

In the app, download/load a model with GPU layers > 0; llama.rn logs Metal
device initialization on context load (`log stream --process RNLlamaExample`
shows native logs). The Parallel Decoding screen is the stress surface for
slots, state files, and multimodal.

## Android with ggml-hexagon (Snapdragon HTP/NPU)

### 1. Check device and prerequisites

```bash
adb devices -l                      # device connected?
adb shell getprop ro.soc.model      # e.g. SM8750 (Snapdragon 8 Elite) - needs a Snapdragon with HTP
ls ~/.hexagon-sdk/6.4.0.2           # Hexagon SDK (or HEXAGON_SDK_ROOT/HEXAGON_TOOLS_ROOT env)
docker info >/dev/null 2>&1 && echo docker-ok   # required on macOS for the HTP toolchain
```

Gradle auto-detects the SDK: the build log prints
`✅ Hexagon SDK detected — enabling DSP build` (or `🚫 ... building CPU-only`).
Prebuilt HTP DSP libs in `bin/arm64-v8a/libggml-htp-v{69,73,75,79,81}.so` are
synced into APK assets by the `prepareHTP` gradle task; when the SDK+Docker are
present they are rebuilt as part of the build.

### 2. Build

```bash
cd example/android && ./gradlew assembleDebug
# APK: example/android/app/build/outputs/apk/debug/app-debug.apk
```

### 3. Verify the APK actually contains hexagon

```bash
unzip -l app/build/outputs/apk/debug/app-debug.apk | grep -iE "hexagon|htp"
# expect: lib/arm64-v8a/librnllama_v8_2_dotprod_i8mm_hexagon_opencl.so (+ _jni_)
#         assets/ggml-hexagon/libggml-htp-v{69,73,75,79,81}.so
```

If these are missing the build silently fell back to CPU-only - fix the SDK /
Docker prerequisites before wasting device time.

### 4. Install, connect Metro, launch

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb reverse tcp:8081 tcp:8081       # Debug JS loads from the host's Metro
adb shell am start -n com.rnllamaexample/.MainActivity
```

### 5. Verify the Hexagon runtime is alive

```bash
adb shell pidof com.rnllamaexample
adb logcat -d | grep -iE "fastrpc|hexagon|htp" | tail
```

Healthy signs: `nativeloader` loading `librnllama_jni_..._hexagon_opencl.so`,
`fastrpc_apps_user_init done` with a DSP domain, `libcdsprpc.so loaded`, and
the HTP extraction dir `/data/user/0/com.rnllamaexample/app_rnllama-htp`
referenced. Then load a model in-app to exercise the NPU; watch
`adb logcat | grep -i htp` during context init and inference.

## Standard test pass

Run this sequence once the build is deployed. Steps 1-3 are the smoke baseline
for ANY native change; step 4 scopes deeper testing to the diff. Report each
step's evidence (a quoted log line or a screenshot), not just "works".

1. **Deploy fresh.** Quit/uninstall the previous instance, install this
   build, launch, confirm the process is alive and the JS console shows
   Metro-served activity. (Android: `adb logcat` shows
   `ReactNativeJS: Running "RNLlamaExample"`.)
2. **Load a model → accelerator init.** Open SimpleChat and initialize a
   model (see UI driving below). Any downloaded model is fine; small ones are
   fastest. Accelerator evidence differs by platform:
   - Android (definitive, in logcat at context init):
     `n_gpu_layers: 99, devices: [ 'HTP0', 'HTP1' ]`,
     `ggml-hex: Loading driver libcdsprpc.so`, `using device HTP0 (Hexagon)`.
     Absent = CPU-only fallback; stop and diagnose.
   - Mac: ggml's Metal logs are stderr, NOT os_log, so `log stream` shows
     nothing and Metro doesn't get them either - the device-init line is only
     visible in Xcode's console. Use the indirect proof instead: the RN dylib
     links Metal and exports the backend (below), and generation runs at
     GPU-consistent speed (step 3). To force the log line, run the app from
     Xcode (Product > Run) rather than a plain launch.
     ```bash
     APP="/Applications/RNLlamaExample 2.app/Wrapper/RNLlamaExample.app"
     otool -L "$APP/RNLlamaExample.debug.dylib" | grep Metal.framework
     nm -gU "$APP/RNLlamaExample.debug.dylib" | grep -c ggml_backend_metal   # > 0
     ```
3. **Completion smoke.** In SimpleChat send `Reply in one short sentence:
   what is 2+3?`. Expect a coherent reply that stops on its own (verified
   good runs: Android/HTP `2 + 3 = 5.` at ~11 tok/s on a 4B; Mac/Metal
   `The answer is 5.` at ~50 tok/s on a 1B - the speed itself is GPU
   evidence). Garbage tokens, an empty reply, or a never-ending stream are
   native-level failures worth bisecting first.
4. **Diff-scoped screens.** From the table below, run every row whose area
   the change touches. For each, capture the named evidence from the logs.
   Two rows deserve extra rigor because their failures are intermittent:
   - *ParallelDecoding*: send the example batch twice; every request must
     complete with a non-empty reply both times, and the second pass must
     log `cache_n > 0` in its `Timings:` objects (state/prefix reuse).
   - *Multimodal*: ask about the same image twice (expects cache reuse),
     then a different image with the same text (expects NO stale answer).

## Driving the UI

### Android (fully scriptable)

Never tap raw hardcoded coordinates for labeled controls - layouts shift and a
tap that lands wrong can long-press into an OS context menu or navigate away.
Instead locate elements by their visible text and tap the CENTER. `scripts/adb_ui.sh`
does this off `uiautomator dump` (no accessibility grant needed):

```bash
source .claude/skills/device-test/scripts/adb_ui.sh
adb logcat -c
ui_tap  "Simple Chat"                       # navigate by button label
ui_tap  "Initialize"                        # load the (already-downloaded) model
ui_wait "Type your message" 90              # block until the chat screen appears
ui_tap  "Type your message"                 # focus the input
ui_type "Reply in one short sentence: what is 2+3?"
```

Only genuinely unlabeled controls (the send paper-plane icon) need a
coordinate tap - read it off a screenshot rather than guessing:

```bash
adb exec-out screencap -p > /tmp/chat.png    # then Read the PNG to find the icon
adb shell input tap 1295 1607                # send icon (this device's layout)
```

Verify results the same way: `adb exec-out screencap -p` for the reply bubble
(shows the answer and a `tok/s` chip), `adb logcat -d | grep RNLlama` for the
`Timings`/`loadPrompt` lines. `ui_dump` prints all on-screen text when you need
to discover an anchor.

### iOS on Mac (scriptable via System Events coordinate clicks)

The RN UI renders as one Metal surface, so its AX tree is unlabeled nested
groups - System Events can't find buttons by name. But `click at {x,y}` and
`keystroke` DO work (verified), including on a secondary display with negative
coordinates. Read the target coordinate off a screenshot; there is no text
anchoring like Android, so this is coordinate-based and you re-screenshot after
each step to confirm.

Get the window origin/size once, capture it, and map screenshot fractions to
screen points (`screen = origin + fraction * size`):

```bash
osascript -e 'tell application "System Events" to tell process "RNLlamaExample" \
  to get {position, size} of window 1'          # e.g. -1073,443 900x839
screencapture -x -o -R-1073,443,900,839 -t png /tmp/mac.png   # then Read it
```

Click, and type (the process must be frontmost first or keystrokes are dropped):

```bash
osascript -e 'tell application "System Events" to click at {-623, 678}'  # a button
osascript <<'OSA'
tell application "System Events"
  set frontmost of process "RNLlamaExample" to true
  delay 0.3
  click at {-713, 1259}          # focus the text field
  delay 0.4
  keystroke "Reply in one short sentence: what is 2+3?"
end tell
OSA
osascript -e 'tell application "System Events" to click at {-201, 1257}'  # send icon
```

Re-screenshot to read the reply bubble and its `tok/s` chip. This works over
the user's live workspace, so if the app is on their active display, say what
you're about to drive first.
5. **Perf gate (perf-relevant changes only).** Bench screen: record
   tokens/sec for prompt and generation, compare against a pre-change run of
   the same model/device; flag regressions beyond run-to-run noise (~5%).

## Screen reference

The Home screen lists all screens; each downloads its own models on first use.
Three log surfaces carry the evidence while driving the app:

- **Android** (everything - JS console.log, native ggml, HTP/fastrpc):
  `adb logcat | grep -iE "rnllama|ReactNativeJS|htp|hexagon"`.
- **Mac**: native ggml logs go to stderr, which a plain launch does not
  capture (`log stream` and Metro both miss them) - use Xcode's console if you
  need them. Otherwise verify from the on-screen `tok/s` chip and screenshots.

| Changed area | Screen(s) | What to check |
|---|---|---|
| Completion, sampling, chat templates | SimpleChat, TextCompletion | streamed text is coherent, stop tokens honored, no truncated UTF-8 |
| Parallel decoding, slots, state files | ParallelDecoding | "Send N Examples" / "Send MM Examples"; in the JS console each completion logs `Timings: { cache_n, prompt_n, ... }` - resend the same prompts and expect `cache_n > 0` (state reuse) and no empty replies |
| Multimodal (mtmd, mmproj) | Multimodal, ParallelDecoding MM examples | image/audio answers are grounded (the dog is a dog); re-asking about the same image reuses cache; swapped images do not leak stale answers |
| Grammar / JSON schema | StructuredOutput | output validates against the schema |
| Tool calling / MCP | ToolCalling | tool_call parsed, arguments well-formed |
| Embeddings / rerank | Embeddings | similarity ranking is sane across the sample set |
| Speculative decoding (MTP) | MTPSpeculative | acceptance rate shown, output matches non-speculative quality |
| TTS (OuteTTS) | TTS | audio plays, no native crash |
| Performance regressions | Bench | tokens/sec prompt+gen vs the numbers before the change |
| Context shifting / long runs | StressTest | context-full handling, no position errors in native logs |
| Model loading / metadata | ModelInfo | GGUF metadata parses, capabilities detected |

Accelerator proof by platform: Android has the definitive logcat lines during
model load (`htp`/`fastrpc`/`Hexagon`); Mac relies on the linked-Metal-backend
check plus GPU-consistent tok/s, since ggml's Metal log isn't captured outside
Xcode. Capture the concrete evidence you used.

## Reporting

State plainly what was verified at which level: compile-only, installed,
launched, or model-inference-on-GPU/NPU. A successful build + launch does not
by itself prove the accelerator ran - the model-load log lines do.

If a level can't be reached (the phone is PIN-locked, the Mac app can't be
driven), say so with the concrete blocker and what remains, rather than
implying full coverage.
