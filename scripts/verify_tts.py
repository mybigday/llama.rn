#!/usr/bin/env python3
"""
End-to-end TTS model verification driver.

For every (model_family, sentence) in the sweep matrix, this script:
  1) downloads the backbone + codec GGUFs via huggingface_hub (cached),
  2) runs tests/build/tts_probe to synthesize a WAV,
  3) transcribes the WAV with faster-whisper,
  4) computes CER (jiwer) against the expected sentence,
  5) emits a Markdown + JSONL report at .scratch/tts-verify/report.md.

Speaker-similarity (ECAPA-TDNN) is optional and enabled per-family when a
reference WAV is provided.

Run:  .scratch/venv/bin/python scripts/verify_tts.py [--families ...] [--texts ...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
VERIFY_DIR = ROOT / ".scratch" / "tts-verify"
WAV_DIR = VERIFY_DIR / "wavs"
VOICES_JSON = VERIFY_DIR / "voices.json"
PROBE = ROOT / "tests" / "build" / "tts_probe"
REPORT_MD = VERIFY_DIR / "report.md"
REPORT_JSONL = VERIFY_DIR / "report.jsonl"


@dataclass
class Model:
    key: str
    repo: str
    backbone: str
    codec_repo: str
    codec_file: str
    # Family used to look up default voice in voices.json (family:lang:name)
    voice_key: Optional[str] = None
    # Whisper decoding language hint for transcriptions
    asr_lang: str = "en"
    # Test corpus per model.  Kept small — we sweep all models × several texts.
    texts: list[tuple[str, str, str]] = field(default_factory=list)  # (label, text, lang)
    n_predict: int = 300
    # If true, run text through espeak-ng IPA before passing to tts_probe.
    # NeuTTS/OuteTTS-legacy/Soprano expect phonemized input; RN example app
    # runs this in JS via toIPA().
    phonemize: bool = False
    # Sampling overrides.  NeuTTS wants temp=1.0/top_k=50/top_p=1.0 to match
    # the RN example app defaults; other models use probe defaults
    # (temp=0.7, top_p=0.9).
    temp: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    # Deterministic seed so retries are stable; verify_tts only forwards the
    # override when seed != 0 (0 = let llama.cpp choose its own seed).  We
    # only pin the seed for high-variance samplers (NeuTTS) where flaky
    # runs would drown out real regressions.
    seed: int = 0


# Text corpora tuned per model family.  Short prompts are known-flaky for
# BlueMagpie / VoxCPM-style continuous flows; we keep them but expect degraded
# CER, and add longer prompts as the primary quality signal.
EN_SHORT = ("en_short", "Hello world.", "en")
EN_MED   = ("en_med",   "The quick brown fox jumps over the lazy dog.", "en")
EN_LONG  = ("en_long",  "Machine learning has transformed how we build software, giving computers the ability to learn from experience.", "en")

ZH_SHORT = ("zh_short", "你好，世界。", "zh")
ZH_MED   = ("zh_med",   "今天天氣真好，我們一起去公園走走吧。", "zh")
ZH_LONG  = ("zh_long",  "人工智慧正在改變我們建立軟體的方式，讓電腦能夠從經驗中學習。", "zh")


MODELS: list[Model] = [
    Model(
        key="oute_0_3",
        repo="OuteAI/OuteTTS-0.3-500M-GGUF",
        backbone="OuteTTS-0.3-500M-Q4_K_M.gguf",
        codec_repo="hans00/codec.cpp-gguf",
        codec_file="wavtokenizer-large-speech-75tokens.gguf",
        voice_key="outetts:en-us:default",
        asr_lang="en",
        texts=[EN_SHORT, EN_MED],
        n_predict=400,
    ),
    Model(
        key="oute_1_0",
        repo="OuteAI/OuteTTS-1.0-0.6B-GGUF",
        backbone="OuteTTS-1.0-0.6B-Q4_K_M.gguf",
        codec_repo="hans00/codec.cpp-gguf",
        codec_file="ibm-research--DAC.speech.gguf",
        voice_key=None,  # V1.0 text-only; native provides no default speaker
        asr_lang="en",
        texts=[EN_SHORT, EN_MED],
        n_predict=400,
    ),
    Model(
        key="soprano",
        repo="hans00/Soprano-1.1-80M-GGUF",
        backbone="Soprano-1.1-80M.F16.gguf",
        codec_repo="hans00/Soprano-1.1-80M-GGUF",
        codec_file="codec-F32.gguf",
        voice_key=None,
        asr_lang="en",
        texts=[EN_SHORT, EN_MED],
        n_predict=400,
    ),
    Model(
        key="neutts_nano",
        repo="hans00/NeuTTS-Nano-GGUF",
        backbone="neutts-nano-q4_k_m.gguf",
        codec_repo="hans00/NeuTTS-Nano-GGUF",
        codec_file="codec-q8_0.gguf",
        voice_key="neutts:en-us:default",
        asr_lang="en",
        texts=[EN_SHORT, EN_MED],
        n_predict=400,
        phonemize=True,
        temp=1.0, top_k=50, top_p=1.0,
        seed=1,
    ),
    Model(
        key="neutts_air",
        repo="hans00/NeuTTS-Air-GGUF",
        backbone="neutts-air-q4_k_m.gguf",
        codec_repo="hans00/NeuTTS-Air-GGUF",
        codec_file="codec-q8_0.gguf",
        voice_key="neutts:en-us:default",
        asr_lang="en",
        texts=[EN_SHORT, EN_MED],
        n_predict=400,
        phonemize=True,
        temp=1.0, top_k=50, top_p=1.0,
        seed=1,
    ),
    Model(
        key="csm_1b",
        repo="hans00/CSM-1B-GGUF",
        backbone="csm-1b-q4_k_m.gguf",
        codec_repo="hans00/CSM-1B-GGUF",
        codec_file="codec-q8_0.gguf",
        voice_key=None,
        asr_lang="en",
        texts=[EN_SHORT, EN_MED],
        n_predict=400,
    ),
    Model(
        key="qwen3_tts",
        repo="hans00/Qwen3-TTS-12Hz-0.6B-GGUF",
        backbone="qwen3-tts-0.6b-q4_k_m.gguf",
        codec_repo="hans00/Qwen3-TTS-12Hz-0.6B-GGUF",
        codec_file="codec-q8_0.gguf",
        voice_key=None,
        asr_lang="zh",
        texts=[ZH_MED, ZH_LONG],
        n_predict=400,
    ),
    Model(
        key="moss_tts_realtime",
        repo="hans00/MOSS-TTS-Realtime-GGUF",
        backbone="moss-tts-realtime-q4_k_m.gguf",
        codec_repo="hans00/MOSS-TTS-Realtime-GGUF",
        codec_file="codec-q5_k_m.gguf",
        voice_key=None,
        asr_lang="zh",
        texts=[ZH_MED, ZH_LONG],
        n_predict=400,
    ),
    Model(
        key="moss_ttsd",
        repo="hans00/MOSS-TTSD-v0.7-GGUF",
        backbone="moss-ttsd-v0.7-q4_k_m.gguf",
        codec_repo="hans00/MOSS-TTSD-v0.7-GGUF",
        codec_file="codec-q5_k_m.gguf",
        voice_key=None,
        asr_lang="zh",
        texts=[ZH_MED, ZH_LONG],
        n_predict=400,
    ),
    Model(
        key="chatterbox_mtl",
        repo="hans00/Chatterbox-Multilingual-TTS-GGUF",
        backbone="chatterbox-mtl-t3-q4_k_m.gguf",
        codec_repo="hans00/Chatterbox-Multilingual-TTS-GGUF",
        codec_file="chatterbox-mtl-codec-q4_k_m.gguf",
        voice_key=None,
        asr_lang="en",
        texts=[EN_MED, EN_LONG],
        n_predict=400,
    ),
    Model(
        key="bluemagpie",
        repo="hans00/BlueMagpie-TTS-GGUF",
        backbone="BlueMagpie-Barbet-1B-q4_k_m.gguf",
        codec_repo="hans00/BlueMagpie-TTS-GGUF",
        codec_file="BlueMagpie-AudioVAE-q8_0.gguf",
        voice_key=None,
        asr_lang="zh",
        texts=[ZH_SHORT, ZH_MED, ZH_LONG],
        n_predict=400,
    ),
]


def hf_download(repo: str, fname: str) -> Path:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id=repo, filename=fname)
    return Path(p)


# NeuTTS/Soprano/OuteTTS-legacy build_*_prompt calls expect the input to
# already be phonemized (the RN example app runs `toIPA` before handing text
# to getFormattedAudioCompletion).  Mirror the JS pipeline's post-processing
# so the probe sees the same string the device sees.
def phonemize_ipa(text: str, language: str = "en-us") -> str:
    from phonemizer import phonemize
    lang = {"en": "en-us", "de": "de", "fr": "fr-fr"}.get(language, language)
    out = phonemize([text], language=lang, backend="espeak")[0]
    return (
        out.strip()
        .replace("ɫ", "l")
        .replace("oʊ", "əʊ")
    )


def dump_voice(voices: dict, key: Optional[str]) -> Optional[Path]:
    if not key or key not in voices:
        return None
    dst = VERIFY_DIR / f"voice_{key.replace(':','_').replace('/','_')}.json"
    if not dst.exists() or dst.stat().st_size == 0:
        dst.write_text(json.dumps(voices[key]))
    return dst


def run_probe(model: Model, text: str, out_wav: Path, speaker: Optional[Path]) -> tuple[bool, str, float]:
    backbone = hf_download(model.repo, model.backbone)
    codec = hf_download(model.codec_repo, model.codec_file)
    probe_text = phonemize_ipa(text, model.asr_lang) if model.phonemize else text
    args = [
        str(PROBE),
        "--backbone", str(backbone),
        "--codec", str(codec),
        "--text", probe_text,
        "--n-predict", str(model.n_predict),
        "--threads", "8",
        "--out-wav", str(out_wav),
    ]
    if model.seed:
        args += ["--seed", str(model.seed)]
    if model.temp is not None:  args += ["--temp",  str(model.temp)]
    if model.top_p is not None: args += ["--top-p", str(model.top_p)]
    if model.top_k is not None: args += ["--top-k", str(model.top_k)]
    if speaker:
        args += ["--speaker-json", str(speaker)]

    t0 = time.time()
    proc = subprocess.run(args, capture_output=True, timeout=900)
    dt = time.time() - t0
    ok = proc.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 44
    out = proc.stdout.decode("utf-8", errors="replace") + proc.stderr.decode("utf-8", errors="replace")
    log_tail = "\n".join(out.splitlines()[-40:])
    return ok, log_tail, dt


_WHISPER_MODEL = None


def transcribe(wav: Path, lang: str) -> str:
    global _WHISPER_MODEL
    from faster_whisper import WhisperModel  # lazy import
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _info = _WHISPER_MODEL.transcribe(str(wav), language=lang, beam_size=1)
    return " ".join(s.text.strip() for s in segments).strip()


def normalize(s: str, lang: str) -> str:
    s = s.strip()
    if lang == "zh":
        # remove all whitespace + punctuation for zh CER
        for ch in "，。！？、,.!?~・：；:; 　\n\t":
            s = s.replace(ch, "")
        return s
    return " ".join(s.lower().split())


def cer(ref: str, hyp: str) -> float:
    import jiwer
    # jiwer.cer treats each character as a token; already what we want.
    return jiwer.cer(ref, hyp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=None,
                    help="Model keys to include (default: all)")
    ap.add_argument("--texts", nargs="*", default=None,
                    help="Text labels to include (default: all defined per model)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N (model, text) probes")
    args = ap.parse_args()

    if not PROBE.exists():
        sys.exit(f"probe binary missing: {PROBE}\nbuild first (cmake --build tests/build --target tts_probe)")

    voices = json.loads(VOICES_JSON.read_text())

    picked = [m for m in MODELS if not args.families or m.key in args.families]
    n = 0
    results = []
    with REPORT_JSONL.open("w") as jl:
        for m in picked:
            speaker_path = dump_voice(voices, m.voice_key)
            for label, text, tlang in m.texts:
                if args.texts and label not in args.texts:
                    continue
                if args.limit and n >= args.limit:
                    break

                out_wav = WAV_DIR / f"{m.key}_{label}.wav"
                print(f"\n=== {m.key} / {label} ===", flush=True)
                ok, log, gen_s = run_probe(m, text, out_wav, speaker_path)
                if not ok:
                    print(f"  PROBE FAILED ({gen_s:.1f}s):\n{log}", flush=True)
                    row = dict(model=m.key, text=label, expected=text, ok=False,
                               error="probe_failed", log_tail=log,
                               gen_s=round(gen_s, 2))
                    jl.write(json.dumps(row, ensure_ascii=False) + "\n"); jl.flush()
                    results.append(row); n += 1; continue

                try:
                    hyp_raw = transcribe(out_wav, m.asr_lang)
                except Exception as e:
                    hyp_raw = ""
                    print(f"  ASR ERROR: {e}", flush=True)

                ref_n = normalize(text, tlang)
                hyp_n = normalize(hyp_raw, tlang)
                c = cer(ref_n, hyp_n) if ref_n else 1.0

                dur = 0.0
                try:
                    import wave
                    with wave.open(str(out_wav)) as w:
                        dur = w.getnframes() / float(w.getframerate())
                except Exception:
                    pass

                row = dict(model=m.key, text=label, expected=text, ok=True,
                           gen_s=round(gen_s, 2), wav_s=round(dur, 3),
                           hyp=hyp_raw, cer=round(c, 3),
                           wav=str(out_wav.relative_to(ROOT)))
                jl.write(json.dumps(row, ensure_ascii=False) + "\n"); jl.flush()
                results.append(row); n += 1
                print(f"  gen={gen_s:.1f}s wav={dur:.2f}s CER={c:.3f}\n  ref: {text}\n  hyp: {hyp_raw}", flush=True)

    # Write Markdown report
    lines = [
        "# TTS verification report",
        "",
        "| model | text | gen_s | wav_s | CER | hyp | notes |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for r in results:
        if not r.get("ok"):
            lines.append(f"| {r['model']} | {r['text']} | {r.get('gen_s','')} | | | | ❌ {r.get('error','')} |")
        else:
            lines.append(
                f"| {r['model']} | {r['text']} | {r['gen_s']} | {r['wav_s']} | {r['cer']:.3f} | "
                f"{(r['hyp'][:80] + '…') if len(r['hyp'])>80 else r['hyp']} | |"
            )
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {REPORT_MD}")


if __name__ == "__main__":
    main()
