#!/usr/bin/env bash
#
# Download GGUF models used by the KV-cache-reuse C++ tests.
#
# Each model exercises a different memory architecture so the tests can prove
# that prompt-cache reuse works (or is correctly bypassed) for every family:
#
#   smollm2   pure-attention        control: seq_rm always succeeds, checkpoint is a no-op
#   lfm2      hybrid (conv+attn)     LFM2.5, arch reports "lfm2"
#   granite4  hybrid (mamba2+attn)  different hybrid state layout
#   mamba     pure recurrent (SSM)  seq_rm always fails, no attention state
#   qwen35    hybrid + <think> strip canonical multi-turn divergence case
#   gemma4    SWA + vision           mmproj + assistant MTP draft gguf included
#
# Usage:
#   ./download.sh                 # download the "core" set (small, no vision, no 2B)
#   ./download.sh all             # download everything
#   ./download.sh smollm2 mamba   # download a specific subset
#
# Requires the HuggingFace CLI (`hf` or `huggingface-cli`) on PATH.

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)"

# Pick whichever HF CLI is installed.
if command -v hf >/dev/null 2>&1; then
  HF="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF="huggingface-cli"
else
  echo "error: neither 'hf' nor 'huggingface-cli' found on PATH" >&2
  echo "install with: pip install -U huggingface_hub" >&2
  exit 1
fi

QUANT="${QUANT:-Q8_0}"

# model spec = "local_name|repo|file [file2 ...] [repo2|file2 ...]"
# Bare extra entries download from the primary repo; "repo|file" entries bring
# their own repo.
# The local_name is the basename the tests look for (see kv_cache_reuse_test.cpp).
spec_for() {
  case "$1" in
    smollm2)  echo "smollm2|prithivMLmods/SmolLM2-135M-Instruct-GGUF|SmolLM2-135M-Instruct.${QUANT}.gguf" ;;
    # LFM2.5 declares general.architecture=lfm2 (same code path), newer + smaller
    # than LFM2-350M. Pinned to Q4_K_M (this repo has no Q8_0 for the 230M).
    lfm2)     echo "lfm2|LiquidAI/LFM2.5-230M-GGUF|LFM2.5-230M-Q4_K_M.gguf" ;;
    granite4) echo "granite4|unsloth/granite-4.0-h-350m-GGUF|granite-4.0-h-350m-${QUANT}.gguf" ;;
    # Pinned quant: the QuantFactory Q8_0 fails to load on the current
    # llama.cpp sync (unreadable tensor info); tensorblock's Q3_K_M loads.
    mamba)    echo "mamba|tensorblock/mamba-130m-hf-GGUF|mamba-130m-hf-Q3_K_M.gguf" ;;
    # Qwen3.5 is hybrid AND its chat template strips the empty <think></think>
    # the model emits, forcing a mid-sequence divergence -> full wipe every turn.
    # Its mmproj ships in the same repo, so it is also the hybrid + M-RoPE vision
    # model, and the only vision model with a native MTP draft head.
    qwen35)   echo "qwen35|bartowski/Qwen_Qwen3.5-2B-GGUF|Qwen_Qwen3.5-2B-Q4_0.gguf mmproj-Qwen_Qwen3.5-2B-f16.gguf" ;;
    # gemma4 MTP is assistant-type: the draft head lives in a separate
    # gemma4-assistant gguf, fetched here as gemma4.assistant.gguf. Pinned to
    # ggml-org's "mtp-*" conversion — most third-party assistant ggufs on HF
    # carry stale arch names and fail to load.
    gemma4)   echo "gemma4|bartowski/google_gemma-4-E2B-it-GGUF|google_gemma-4-E2B-it-Q4_K_M.gguf mmproj-google_gemma-4-E2B-it-f16.gguf ggml-org/gemma-4-E2B-it-GGUF|mtp-gemma-4-E2B-it-Q8_0.gguf" ;;
    # LFM2.5-VL-450M is a hybrid (LFM2 arch) vision model: small, and unlike the
    # SWA gemma4 its seq_rm genuinely fails on divergence, so it exercises the
    # multimodal checkpoint-restore branch end-to-end.
    lfm2vl)   echo "lfm2vl|LiquidAI/LFM2.5-VL-450M-GGUF|LFM2.5-VL-450M-Q8_0.gguf mmproj-LFM2.5-VL-450m-Q8_0.gguf" ;;
    # SmolVLM-500M: dense-attention (SmolLM2) vision model -- seq_rm reuses the
    # prefix for free (no state checkpoints), covering the non-recurrent path.
    smolvlm)  echo "smolvlm|ggml-org/SmolVLM-500M-Instruct-GGUF|SmolVLM-500M-Instruct-Q8_0.gguf mmproj-SmolVLM-500M-Instruct-Q8_0.gguf" ;;
    *) echo "" ;;
  esac
}

CORE=(smollm2 lfm2 granite4 mamba)
ALL=(smollm2 lfm2 granite4 mamba qwen35 gemma4 lfm2vl smolvlm)

if [ "$#" -eq 0 ]; then
  WANT=("${CORE[@]}")
elif [ "$1" = "all" ]; then
  WANT=("${ALL[@]}")
else
  WANT=("$@")
fi

download_file() {
  local repo="$1" file="$2" out="$3"
  if [ -f "$out" ]; then
    echo "  [skip] $(basename "$out") already present"
    return 0
  fi
  echo "  [get ] $repo :: $file"
  # hf download writes to a cache and prints the path; --local-dir places it directly.
  "$HF" download "$repo" "$file" --local-dir "$MODELS_DIR/.hf/$repo" >/dev/null
  mv "$MODELS_DIR/.hf/$repo/$file" "$out"
}

for key in "${WANT[@]}"; do
  [ "$key" = "images" ] && continue  # handled below
  spec="$(spec_for "$key")"
  if [ -z "$spec" ]; then
    echo "unknown model key: $key (valid: ${ALL[*]})" >&2
    exit 1
  fi
  local_name="${spec%%|*}"
  rest="${spec#*|}"
  repo="${rest%%|*}"
  files="${rest#*|}"

  echo "== $key ($repo) =="
  idx=0
  for entry in $files; do
    # entry is either "file" (from the primary repo) or "repo|file"
    case "$entry" in
      *\|*) erepo="${entry%%|*}"; f="${entry#*|}" ;;
      *)    erepo="$repo";        f="$entry" ;;
    esac
    if [ "$idx" -eq 0 ]; then
      # primary weights -> <local_name>.gguf
      download_file "$erepo" "$f" "$MODELS_DIR/${local_name}.gguf"
    else
      case "$f" in
        # MTP draft -> <local_name>.assistant.gguf, the name the tests auto-wire
        mtp-*|*assistant*) download_file "$erepo" "$f" "$MODELS_DIR/${local_name}.assistant.gguf" ;;
        # extra files (e.g. mmproj) -> keep original name prefixed with local_name
        *)                 download_file "$erepo" "$f" "$MODELS_DIR/${local_name}.${f}" ;;
      esac
    fi
    idx=$((idx + 1))
  done
done

# Fetch the vision test images (a clear dog and a clear cat) whenever a vision
# model is in the set. They are third-party sample photos, so we download rather
# than commit them. The vision tests are skipped if these are absent.
need_images=false
for key in "${WANT[@]}"; do
  case "$key" in qwen35|gemma4|lfm2vl|smolvlm|images) need_images=true ;; esac
done
if [ "$need_images" = true ]; then
  echo "== vision test images =="
  fetch_img() { # url out
    [ -f "$2" ] && { echo "  [skip] $(basename "$2")"; return 0; }
    echo "  [get ] $(basename "$2")"
    curl -fsSL -o "$2" "$1" || { echo "  [warn] failed to fetch $(basename "$2")"; return 0; }
    # Downscale to keep image-encode time modest (token count is set by the model
    # grid, not the input size, so this doesn't change the test). Best-effort:
    # skipped if Pillow isn't installed.
    python3 - "$2" <<'PY' 2>/dev/null || true
import sys
from PIL import Image
p = sys.argv[1]
im = Image.open(p).convert("RGB")
im.thumbnail((384, 384))
im.save(p, quality=90)
PY
  }
  fetch_img "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg" \
            "$MODELS_DIR/test_dog.jpg"
  fetch_img "https://raw.githubusercontent.com/pytorch/serve/master/examples/image_classifier/kitten.jpg" \
            "$MODELS_DIR/test_cat.jpg"
fi

rm -rf "$MODELS_DIR/.hf"
echo "done. models in: $MODELS_DIR"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || true
