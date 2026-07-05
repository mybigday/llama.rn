#!/usr/bin/env python3
"""Standalone Barbet (Open-Formosa R2 / BlueMagpie TSLM) → llama.cpp GGUF converter.

Barbet is a Mamba2 + attention hybrid (motif: global, sliding, sliding, mamba2).
It is the text-semantic backbone of BlueMagpie-TTS; the acoustic adaptor + AudioVAE
run in codec.cpp, this backbone runs in llama.cpp (arch "barbet").

The Megatron-style Mamba2 mixer stores 5 separate in-projections and 3 separate
depthwise convs; this script fuses them into the single ssm_in / ssm_conv1d
tensors llama.cpp's build_mamba2_layer expects:
  ssm_in      = concat([in_proj_z, in_proj_x, in_proj_b, in_proj_c, in_proj_dt])  (order [z,x,B,C,dt])
  ssm_conv1d  = concat([conv_x, conv_b, conv_c]) as (channels, d_conv)
  ssm_a       = -exp(A_log)                                                       (per-head)
  ssm_norm    = norm reshaped (n_group, group_size)
Attention layers carry per-head q/k RMSNorm (qk_norm). Layer types are written as
a per-layer schedule; mamba layers are marked with head_count_kv = 0 (nemotron-h
convention).  Tied embeddings (no separate lm_head); MTP heads dropped.

Usage:
  python convert_barbet_to_gguf.py --src <pytorch_model.bin> --config <config.json> --out barbet.gguf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import gguf

ARCH = "barbet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="pytorch_model.bin with base_lm.* tensors")
    ap.add_argument("--config", required=True, help="BlueMagpie config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--outtype", default="f16", choices=["f16", "f32"])
    ap.add_argument("--tokenizer-json", help="HF tokenizer.json (bakes the BPE vocab into the GGUF so llama.cpp can tokenize text natively; omit for the codec.cpp external-tokenization mode)")
    ap.add_argument("--tokenizer-config", help="HF tokenizer_config.json (used to pull bos/eos ids)")
    args = ap.parse_args()

    cfg = json.load(open(args.config))["barbet_config"]
    sd = torch.load(args.src, map_location="cpu", weights_only=True)
    P = "base_lm.backbone."
    bb = {k[len(P):]: v for k, v in sd.items() if k.startswith(P)}

    n_layer   = cfg["num_hidden_layers"]
    n_embd    = cfg["hidden_size"]
    head_dim  = cfg["head_dim"]
    n_head    = cfg["num_attention_heads"]
    n_head_kv = cfg["num_key_value_heads"]
    n_ff      = cfg["intermediate_size"]
    eps       = cfg["rms_norm_eps"]
    rope_theta = cfg["rope_theta"]
    swa       = cfg["sliding_window_size"]
    globals_  = set(cfg["global_attention_layers"])
    mambas    = set(cfg["mamba_layers"])
    d_state   = cfg["mamba_d_state"]
    d_conv    = cfg["mamba_d_conv"]
    d_inner   = n_embd * cfg["mamba_expand"]
    n_group   = n_head_kv                                   # Megatron Mamba2: groups = kv heads
    n_ssm_head = d_inner // head_dim                         # 3072/128 = 24

    np_dtype = np.float16 if args.outtype == "f16" else np.float32

    def layer_type(i: int) -> str:
        if i in globals_: return "global"
        if i in mambas:   return "mamba"
        return "sliding"

    w = gguf.GGUFWriter(args.out, ARCH)
    w.add_name("Barbet")

    # Tokenizer path: BlueMagpie's tokenizer.json is a plain BPE (GPT2-family
    # with a Sequence pre-tokenizer). Baking it in lets llama.cpp tokenize text
    # in-process — matches how OuteTTS / NeuTTS ship their vocabs.  Fall back to
    # the codec.cpp `tokenizer_model="none"` path when tokenizer.json isn't
    # provided (that's the codec.cpp-driven flow where a Python PangolinTokenizer
    # runs upstream and feeds token IDs directly).
    if args.tokenizer_json:
        tj = json.load(open(args.tokenizer_json))
        added = tj.get("added_tokens") or []
        model = tj.get("model") or {}
        vocab = model.get("vocab") or {}
        merges = model.get("merges") or []
        # Compose id → (token, type) with added_tokens overriding vocab.
        n_vocab = int(cfg["vocab_size"])
        id_to_tok = [""] * n_vocab
        id_to_type = [gguf.TokenType.NORMAL] * n_vocab
        for tok, tid in vocab.items():
            if 0 <= tid < n_vocab:
                id_to_tok[tid] = tok
        for at in added:
            tid = at.get("id")
            if tid is None or not (0 <= tid < n_vocab):
                continue
            id_to_tok[tid] = at.get("content", id_to_tok[tid])
            id_to_type[tid] = (gguf.TokenType.CONTROL if at.get("special")
                               else gguf.TokenType.USER_DEFINED)
        # BlueMagpie's runtime (bluemagpie/config.py::resolve_barbet_config)
        # auto-allocates its TTS control tokens in the *Megatron padding region*
        # — the ids in [EFFECTIVE_VOCAB_SIZE, vocab_size) that the BPE tokenizer
        # never emits.  These ids (spk=E+4, audio_start=E+0, ...) are the ones
        # the model was actually conditioned on; the `<|speaker|>`/`<|audio_start|>`
        # entries baked into tokenizer.json (114674 / 114666) are a DIFFERENT,
        # unused set of rows.  We must expose the padding-region ids as real
        # control tokens so the runtime prompt (`<|bm_spk|>` + text +
        # `<|bm_audio_start|>`) tokenizes onto the trained embedding rows.  The
        # allocation order below mirrors resolve_barbet_config exactly.
        effective_vocab = int(cfg.get("barbet_effective_vocab_size") or 0)
        if effective_vocab <= 0:
            # Pangolin R2 contract: EFFECTIVE_VOCAB_SIZE = 114822 for the padded
            # 114944 vocab.  Fall back to that when the config doesn't override.
            effective_vocab = 114822 if n_vocab == 114944 else n_vocab
        # (config field, canonical control-token string) in allocation order.
        bm_special = [
            ("audio_start_token",     "<|bm_audio_start|>"),
            ("audio_end_token",       "<|bm_audio_end|>"),
            ("ref_audio_start_token", "<|bm_ref_audio_start|>"),
            ("ref_audio_end_token",   "<|bm_ref_audio_end|>"),
            ("spk_token",             "<|bm_spk|>"),
        ]
        next_id = effective_vocab
        for field, text in bm_special:
            requested = cfg.get(field, -1)
            tid = requested if (requested is not None and requested >= 0) else next_id
            if requested is None or requested < 0:
                next_id += 1
            if 0 <= tid < n_vocab:
                id_to_tok[tid] = text
                id_to_type[tid] = gguf.TokenType.CONTROL
        # Any gaps become byte-fallback padding tokens; they're never emitted
        # by the trained head so labels don't matter — just need something.
        for i in range(n_vocab):
            if not id_to_tok[i]:
                id_to_tok[i] = f"<|unused_{i}|>"
                id_to_type[i] = gguf.TokenType.UNUSED
        # gguf writer expects list[str] + list[int].  Merges are already the
        # native "a b" strings gpt2 uses.
        w.add_tokenizer_model("gpt2")
        w.add_tokenizer_pre("default")
        w.add_token_list(id_to_tok)
        w.add_token_types([int(t) for t in id_to_type])
        if merges:
            w.add_token_merges([" ".join(m) if isinstance(m, list) else m for m in merges])
        # Pull bos/eos ids from tokenizer_config if provided; fall back to the
        # first control-token index in the vocab that matches "<s>" / "</s>".
        def _find_id_by_text(text):
            for i, t in enumerate(id_to_tok):
                if t == text:
                    return i
            return None
        tc = json.load(open(args.tokenizer_config)) if args.tokenizer_config else {}
        bos_text = tc.get("bos_token") or "<s>"
        eos_text = tc.get("eos_token") or "</s>"
        bos_id = _find_id_by_text(bos_text if isinstance(bos_text, str) else bos_text.get("content", "<s>"))
        eos_id = _find_id_by_text(eos_text if isinstance(eos_text, str) else eos_text.get("content", "</s>"))
        if bos_id is not None: w.add_bos_token_id(bos_id)
        if eos_id is not None: w.add_eos_token_id(eos_id)
    else:
        # External tokenization (HF PangolinTokenizer in the codec.cpp TTS
        # driver) → no-op vocab; token IDs are fed directly.
        w.add_tokenizer_model("none")
    w.add_vocab_size(cfg["vocab_size"])
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    # Per-layer head_count_kv: 0 marks a recurrent (mamba2) layer, so llama.cpp's
    # `is_recr = n_head_kv(i)==0` detection works automatically (nemotron-h/granite
    # convention).  Attention layers carry the real kv-head count.
    w.add_head_count_kv([0 if i in mambas else n_head_kv for i in range(n_layer)])
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)
    w.add_layer_norm_rms_eps(eps)
    w.add_rope_freq_base(rope_theta)
    w.add_rope_dimension_count(head_dim)
    w.add_ssm_conv_kernel(d_conv)
    w.add_ssm_inner_size(d_inner)
    w.add_ssm_state_size(d_state)
    w.add_ssm_time_step_rank(n_ssm_head)
    w.add_ssm_group_count(n_group)
    # NOTE: Barbet uses sliding-window attention (window=8192) on non-global
    # attention layers, but full causal attention is numerically identical for
    # any sequence <= 8192 tokens — which covers all TTS use.  We therefore run
    # full causal attention and do not emit a per-layer SWA schedule.  qk_norm is
    # always present in Barbet, so the q/k norm tensors are loaded unconditionally.
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if args.outtype == "f16" else gguf.LlamaFileType.ALL_F32)

    def t(name: str) -> np.ndarray:
        return bb[name].to(torch.float32).numpy()

    def add(name: str, arr: np.ndarray, force_f32: bool = False):
        a = arr.astype(np.float32 if force_f32 else np_dtype)
        w.add_tensor(name, a)

    # embeddings + final norm (tied: no separate output.weight)
    add("token_embd.weight", t("embed_tokens.weight"))
    add("output_norm.weight", t("norm.weight"), force_f32=True)

    for i in range(n_layer):
        L = f"layers.{i}."
        b = f"blk.{i}."
        lt = layer_type(i)
        add(b + "attn_norm.weight", t(L + "input_layernorm.weight"), force_f32=True)
        add(b + "ffn_norm.weight", t(L + "post_attention_layernorm.weight"), force_f32=True)
        add(b + "ffn_gate.weight", t(L + "mlp.gate_proj.weight"))
        add(b + "ffn_up.weight",   t(L + "mlp.up_proj.weight"))
        add(b + "ffn_down.weight", t(L + "mlp.down_proj.weight"))

        if lt == "mamba":
            M = L + "mixer."
            ssm_in = np.concatenate([t(M + "in_proj_z.weight"), t(M + "in_proj_x.weight"),
                                     t(M + "in_proj_b.weight"), t(M + "in_proj_c.weight"),
                                     t(M + "in_proj_dt.weight")], axis=0)   # (d_in_proj, n_embd)
            add(b + "ssm_in.weight", ssm_in)
            conv_w = np.concatenate([t(M + "conv_x.weight"), t(M + "conv_b.weight"),
                                     t(M + "conv_c.weight")], axis=0).squeeze(1)  # (channels, d_conv)
            add(b + "ssm_conv1d.weight", conv_w, force_f32=True)  # ggml_ssm_conv requires f32 weight
            conv_b = np.concatenate([t(M + "conv_x.bias"), t(M + "conv_b.bias"), t(M + "conv_c.bias")], axis=0)
            add(b + "ssm_conv1d.bias", conv_b, force_f32=True)
            add(b + "ssm_dt.bias", t(M + "dt_bias"), force_f32=True)
            A = -np.exp(t(M + "A_log").astype(np.float64)).astype(np.float32).reshape(n_ssm_head, 1)
            add(b + "ssm_a", A, force_f32=True)
            add(b + "ssm_d", t(M + "D").reshape(n_ssm_head, 1), force_f32=True)
            add(b + "ssm_norm.weight", t(M + "norm.weight").reshape(n_group, d_inner // n_group), force_f32=True)
            add(b + "ssm_out.weight", t(M + "out_proj.weight"))
        else:
            M = L + "mixer."
            add(b + "attn_q.weight", t(M + "q_proj.weight"))
            add(b + "attn_k.weight", t(M + "k_proj.weight"))
            add(b + "attn_v.weight", t(M + "v_proj.weight"))
            add(b + "attn_output.weight", t(M + "o_proj.weight"))
            add(b + "attn_q_norm.weight", t(M + "q_norm.weight"), force_f32=True)
            add(b + "attn_k_norm.weight", t(M + "k_norm.weight"), force_f32=True)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
