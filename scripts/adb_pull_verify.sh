#!/usr/bin/env bash
# adb-based on-device TTS verification helper.
#
# Workflow (semi-interactive):
#   1. Run the example app on a connected Android device / emulator.
#   2. Pick a model, type or paste your test sentence, hit Generate.
#   3. When generation finishes, the app auto-dumps the last decoded audio to
#         /sdcard/Download/rnllama-tts-verify.wav
#      via example/src/utils/audioUtils.ts::dumpTtsWavToDisk.
#   4. Run this script with the expected sentence + language; it pulls the
#      WAV, runs faster-whisper locally, and reports CER.
#
# Usage:
#   scripts/adb_pull_verify.sh "expected sentence" [--lang zh|en] [--tag name]
#
# --tag NAME  copies the pulled WAV to .scratch/tts-verify/on-device/NAME.wav
#             so you can build up an on-device corpus alongside the host run.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 \"expected sentence\" [--lang zh|en] [--tag name]" >&2
    exit 2
fi

EXPECTED="$1"; shift
LANG="en"
TAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --lang) LANG="$2"; shift 2 ;;
        --tag)  TAG="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

REMOTE="/sdcard/Download/rnllama-tts-verify.wav"
OUT_DIR=".scratch/tts-verify/on-device"
mkdir -p "$OUT_DIR"

STAMP="$(python3 -c 'import time;print(int(time.time()))')"
LOCAL="$OUT_DIR/latest-$STAMP.wav"

echo "[adb] checking remote WAV: $REMOTE"
if ! adb shell "test -f $REMOTE"; then
    echo "[adb] no WAV at $REMOTE — generate speech in the app first" >&2
    exit 3
fi

echo "[adb] pulling → $LOCAL"
adb pull "$REMOTE" "$LOCAL" >/dev/null

if [ -n "$TAG" ]; then
    TAGGED="$OUT_DIR/$TAG.wav"
    cp "$LOCAL" "$TAGGED"
    echo "[adb] tagged copy: $TAGGED"
fi

VENV_PY=".scratch/venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "[verify] venv missing — expected .scratch/venv/bin/python" >&2
    exit 4
fi

"$VENV_PY" - "$LOCAL" "$LANG" "$EXPECTED" <<'PY'
import sys, wave, jiwer
from faster_whisper import WhisperModel

wav_path, lang, expected = sys.argv[1], sys.argv[2], sys.argv[3]

with wave.open(wav_path) as w:
    dur = w.getnframes() / float(w.getframerate())
    sr = w.getframerate()

model = WhisperModel("small", device="cpu", compute_type="int8")
segments, _info = model.transcribe(wav_path, language=lang, beam_size=1)
hyp = " ".join(s.text.strip() for s in segments).strip()

def norm(s, lang):
    s = s.strip()
    if lang == "zh":
        for ch in "，。！？、,.!?~・：；:; 　\n\t":
            s = s.replace(ch, "")
        return s
    return " ".join(s.lower().split())

ref_n = norm(expected, lang)
hyp_n = norm(hyp, lang)
cer = jiwer.cer(ref_n, hyp_n) if ref_n else 1.0

print()
print(f"  wav      : {wav_path}")
print(f"  duration : {dur:.2f}s @ {sr} Hz")
print(f"  expected : {expected}")
print(f"  heard    : {hyp}")
print(f"  CER      : {cer:.3f}")
PY
