// Chatterbox T3 host-orchestration helpers.
//
// T3's transformer is a generic embd-driven Llama backbone (the host owns
// the llama.cpp decode loop).  The T3-specific pieces — text tokenizer,
// prompt-embed assembly (cond_enc + text_emb + speech BOS), CFG uncond
// lane, and per-step speech embeds — live here on the codec.cpp side and
// are surfaced through the `codec_lm_chatterbox_*` public API.
//
// Reference: `.model-src/chatterbox/src/chatterbox/models/t3/t3.py`
// (`inference` / `prepare_input_embeds`) and `tts.py` (`generate`,
// `punc_norm`, `EnTokenizer`).

#include "speaker_chatterbox.h"

#include "lm_internal.h"
#include "../codec_internal.h"

#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------
// Lazy per-lm chatterbox state (info + dequanted tables + tokenizer).
// Cached on codec_lm::impl-adjacent storage via a static map keyed by lm
// pointer would leak; instead we stash it inside a heap struct pointed to
// by a thread-safe-enough lazy singleton per lm.  Simpler: recompute the
// cheap parts, cache the expensive table dequant in function-local
// statics keyed by the tensor pointer.
// ---------------------------------------------------------------------

struct BpeTokenizer {
    bool loaded = false;
    std::vector<std::string>                 id_to_tok;   // id-ordered
    std::unordered_map<std::string, int32_t> tok_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;  // "a b" -> rank
    // Added tokens (special atoms) matched greedily before BPE.
    std::vector<std::pair<std::string, int32_t>> added;   // content, id
    int32_t unk_id = 1;
    std::string space_tok = "[SPACE]";
    int32_t     space_id  = -1;
};

struct CbxState {
    codec_lm_chatterbox_info info = {};
    bool                     have_info = false;

    // Dequanted tables (lazy).
    std::vector<float> text_emb;       // [text_vocab * hidden]
    std::vector<float> text_pos_emb;   // [max_text+2 * hidden]
    std::vector<float> speech_emb;     // [speech_vocab * hidden]
    std::vector<float> speech_pos_emb; // [max_speech+4 * hidden]
    int32_t text_pos_rows = 0;
    int32_t speech_pos_rows = 0;

    // Builtin conds.
    bool               have_builtin = false;
    std::vector<float> builtin_speaker_emb;
    std::vector<int32_t> builtin_cond_tokens;
    float              builtin_emotion = 0.5f;

    BpeTokenizer tok;
};

// One state per codec_lm.  Small leak-free registry.
std::unordered_map<codec_lm *, CbxState *> g_states;

CbxState * get_state(codec_lm * lm) {
    auto it = g_states.find(lm);
    if (it != g_states.end()) return it->second;
    return nullptr;
}

bool is_chatterbox(codec_lm * lm) {
    if (lm == nullptr || lm->codec == nullptr || lm->codec->gguf == nullptr) return false;
    return lm_gguf_find_key(lm->codec->gguf, "codec.lm.chatterbox.start_speech_token") >= 0;
}

// ─── tokenizer helpers ───────────────────────────────────────────────

// punc_norm from tts.py — deterministic English text cleanup.
std::string punc_norm(const std::string & in) {
    if (in.empty()) return "You need to add some text for me to talk.";
    std::string text = in;

    // Capitalise first letter if lowercase ASCII.
    if (!text.empty() && text[0] >= 'a' && text[0] <= 'z') {
        text[0] = (char) std::toupper((unsigned char) text[0]);
    }

    // Collapse whitespace runs to single spaces (" ".join(text.split())).
    {
        std::istringstream iss(text);
        std::string word, joined;
        bool first = true;
        while (iss >> word) {
            if (!first) joined += ' ';
            joined += word;
            first = false;
        }
        text = joined;
    }

    // Replace uncommon / LLM punctuation (order matters — matches tts.py).
    struct Rep { const char * from; const char * to; };
    static const Rep reps[] = {
        {"...", ", "}, {"\xE2\x80\xA6", ", "},          // …
        {":", ","}, {" - ", ", "}, {";", ", "},
        {"\xE2\x80\x94", "-"}, {"\xE2\x80\x93", "-"},    // — –
        {" ,", ","},
        {"\xE2\x80\x9C", "\""}, {"\xE2\x80\x9D", "\""},  // “ ”
        {"\xE2\x80\x98", "'"}, {"\xE2\x80\x99", "'"},    // ‘ ’
    };
    for (const auto & r : reps) {
        std::string from = r.from, to = r.to;
        size_t pos = 0;
        while ((pos = text.find(from, pos)) != std::string::npos) {
            text.replace(pos, from.size(), to);
            pos += to.size();
        }
    }

    // rstrip spaces.
    while (!text.empty() && text.back() == ' ') text.pop_back();

    // Add a full stop if it doesn't end with sentence-ending punctuation.
    if (!text.empty()) {
        char c = text.back();
        if (c != '.' && c != '!' && c != '?' && c != '-' && c != ',') text += '.';
    } else {
        text += '.';
    }
    return text;
}

void split_lines(const std::string & s, std::vector<std::string> * out) {
    out->clear();
    size_t start = 0;
    while (start <= s.size()) {
        size_t nl = s.find('\n', start);
        if (nl == std::string::npos) {
            out->push_back(s.substr(start));
            break;
        }
        out->push_back(s.substr(start, nl - start));
        start = nl + 1;
    }
}

void load_tokenizer(codec_lm * lm, BpeTokenizer * tk) {
    lm_gguf_context * gf = lm->codec->gguf;
    if (lm_gguf_find_key(gf, "codec.lm.chatterbox.tokenizer.tokens") < 0) return;

    std::string tokens_blob = codec_read_str_kv(gf, "codec.lm.chatterbox.tokenizer.tokens", "");
    std::string merges_blob = codec_read_str_kv(gf, "codec.lm.chatterbox.tokenizer.merges", "");
    std::string added_blob  = codec_read_str_kv(gf, "codec.lm.chatterbox.tokenizer.added", "");
    std::string unk         = codec_read_str_kv(gf, "codec.lm.chatterbox.tokenizer.unk_token", "[UNK]");

    split_lines(tokens_blob, &tk->id_to_tok);
    // Last split may be a trailing empty if the blob ended with '\n'; the
    // vocab has no trailing newline so this is fine.
    tk->tok_to_id.reserve(tk->id_to_tok.size() * 2);
    for (int32_t i = 0; i < (int32_t) tk->id_to_tok.size(); ++i) {
        tk->tok_to_id[tk->id_to_tok[i]] = i;
    }

    std::vector<std::string> merge_lines;
    split_lines(merges_blob, &merge_lines);
    int32_t rank = 0;
    for (const auto & m : merge_lines) {
        if (m.empty()) continue;
        tk->merge_rank[m] = rank++;
    }

    std::vector<std::string> added_lines;
    split_lines(added_blob, &added_lines);
    for (const auto & a : added_lines) {
        size_t tab = a.find('\t');
        if (tab == std::string::npos) continue;
        std::string content = a.substr(0, tab);
        int32_t id = 0;
        try { id = std::stoi(a.substr(tab + 1)); } catch (...) { continue; }
        tk->added.emplace_back(content, id);
    }
    // Longer added-token strings first so greedy matching prefers them.
    std::sort(tk->added.begin(), tk->added.end(),
              [](const auto & x, const auto & y) { return x.first.size() > y.first.size(); });

    auto uit = tk->tok_to_id.find(unk);
    tk->unk_id = (uit != tk->tok_to_id.end()) ? uit->second : 1;
    auto sit = tk->tok_to_id.find(tk->space_tok);
    tk->space_id = (sit != tk->tok_to_id.end()) ? sit->second : -1;
    tk->loaded = true;
}

// Apply greedy rank-based BPE merges to a sequence of symbol strings.
void bpe_merge(const BpeTokenizer & tk, std::vector<std::string> * syms) {
    if (syms->size() < 2) return;
    while (true) {
        int32_t best_rank = INT32_MAX;
        int32_t best_i = -1;
        for (int32_t i = 0; i + 1 < (int32_t) syms->size(); ++i) {
            std::string pair = (*syms)[i] + " " + (*syms)[i + 1];
            auto it = tk.merge_rank.find(pair);
            if (it != tk.merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        (*syms)[best_i] = (*syms)[best_i] + (*syms)[best_i + 1];
        syms->erase(syms->begin() + best_i + 1);
    }
}

bool is_word_char(unsigned char c) {
    // Approximate the HF Whitespace pre-tokenizer \w (ASCII letters,
    // digits, underscore).  The Chatterbox EN vocab is ASCII-only.
    return std::isalnum(c) || c == '_';
}

// Encode a plain chunk (already free of added tokens) into token ids by
// splitting on the Whitespace regex (\w+|[^\w\s]+) then BPE-merging each
// piece; each character becomes a base symbol (byte-wise; ASCII here).
void encode_chunk(const BpeTokenizer & tk, const std::string & chunk,
                  std::vector<int32_t> * out) {
    size_t i = 0;
    const size_t n = chunk.size();
    while (i < n) {
        unsigned char c = (unsigned char) chunk[i];
        if (std::isspace(c)) { ++i; continue; }
        size_t j = i;
        bool word = is_word_char(c);
        while (j < n) {
            unsigned char cj = (unsigned char) chunk[j];
            if (std::isspace(cj)) break;
            if (is_word_char(cj) != word) break;
            ++j;
        }
        std::string piece = chunk.substr(i, j - i);
        i = j;

        // base symbols = single characters
        std::vector<std::string> syms;
        syms.reserve(piece.size());
        for (char ch : piece) syms.emplace_back(1, ch);
        bpe_merge(tk, &syms);
        for (const auto & s : syms) {
            auto it = tk.tok_to_id.find(s);
            out->push_back(it != tk.tok_to_id.end() ? it->second : tk.unk_id);
        }
    }
}

void tokenize_bpe(const BpeTokenizer & tk, const std::string & text_in,
                  std::vector<int32_t> * out) {
    out->clear();
    // EnTokenizer.encode: replace ' ' with the [SPACE] added token, then
    // the HF tokenizer extracts added tokens (incl [SPACE]) as atomic and
    // BPE-encodes the rest.
    std::string text = text_in;
    {
        std::string tmp;
        for (char c : text) {
            if (c == ' ') tmp += tk.space_tok;
            else tmp += c;
        }
        text = tmp;
    }

    // Greedily scan for added tokens; emit their ids, BPE the gaps.
    size_t pos = 0;
    const size_t n = text.size();
    std::string pending;
    auto flush = [&]() {
        if (!pending.empty()) { encode_chunk(tk, pending, out); pending.clear(); }
    };
    while (pos < n) {
        bool matched = false;
        for (const auto & a : tk.added) {
            const std::string & s = a.first;
            if (!s.empty() && text.compare(pos, s.size(), s) == 0) {
                flush();
                out->push_back(a.second);
                pos += s.size();
                matched = true;
                break;
            }
        }
        if (!matched) { pending += text[pos]; ++pos; }
    }
    flush();
}

// ─── table dequant helpers ───────────────────────────────────────────

bool dequant_table(codec_lm * lm, const char * name, std::vector<float> * out,
                   int32_t * out_rows, int32_t hidden) {
    lm_ggml_tensor * t = lm_ggml_get_tensor(lm->codec->weights, name);
    if (t == nullptr) return false;
    if (!codec_tensor_as_vec_f32(t, out)) return false;
    if (hidden > 0 && (out->size() % (size_t) hidden) != 0) return false;
    if (out_rows) *out_rows = hidden > 0 ? (int32_t) (out->size() / (size_t) hidden) : 0;
    return true;
}

// ─── state init ──────────────────────────────────────────────────────

CbxState * ensure_state(codec_lm * lm) {
    if (!is_chatterbox(lm)) return nullptr;
    CbxState * st = get_state(lm);
    if (st != nullptr) return st;

    st = new (std::nothrow) CbxState();
    if (st == nullptr) return nullptr;

    lm_gguf_context * gf = lm->codec->gguf;
    codec_lm_chatterbox_info & ci = st->info;
    ci.hidden_dim         = codec_read_i32_kv(gf, "codec.lm.hidden_dim", 1024);
    ci.text_vocab_size    = codec_read_i32_kv(gf, "codec.lm.chatterbox.text_vocab_size", 704);
    {
        std::vector<int32_t> cbs;
        codec_read_i32_array_kv_vec(gf, "codec.lm.codebook_sizes", &cbs);
        ci.speech_vocab_size = cbs.empty() ? 8194 : cbs[0];
    }
    ci.start_text_token   = codec_read_i32_kv(gf, "codec.lm.chatterbox.start_text_token", 255);
    ci.stop_text_token    = codec_read_i32_kv(gf, "codec.lm.chatterbox.stop_text_token", 0);
    ci.start_speech_token = codec_read_i32_kv(gf, "codec.lm.chatterbox.start_speech_token", 6561);
    ci.stop_speech_token  = codec_read_i32_kv(gf, "codec.lm.chatterbox.stop_speech_token", 6562);
    ci.cond_rows          = lm->has_speaker_encoder ? lm->speaker_info.n_rows : 34;
    ci.has_tokenizer      = lm_gguf_find_key(gf, "codec.lm.chatterbox.tokenizer.tokens") >= 0 ? 1 : 0;
    ci.has_builtin_conds  = codec_read_bool_kv(gf, "codec.lm.chatterbox.has_builtin_conds", false) ? 1 : 0;
    ci.is_multilingual    = codec_read_bool_kv(gf, "codec.lm.chatterbox.is_multilingual", false) ? 1 : 0;
    st->have_info = true;

    if (ci.has_tokenizer) load_tokenizer(lm, &st->tok);

    if (ci.has_builtin_conds) {
        int key = lm_gguf_find_key(gf, "codec.lm.chatterbox.builtin.speaker_emb");
        if (key >= 0) {
            size_t n = lm_gguf_get_arr_n(gf, key);
            st->builtin_speaker_emb.assign(n, 0.0f);
            codec_read_f32_array_kv(gf, "codec.lm.chatterbox.builtin.speaker_emb",
                                    st->builtin_speaker_emb.data(), (int32_t) n);
        }
        codec_read_i32_array_kv_vec(gf, "codec.lm.chatterbox.builtin.cond_prompt_speech_tokens",
                                    &st->builtin_cond_tokens);
        st->builtin_emotion = codec_read_f32_kv(gf, "codec.lm.chatterbox.builtin.emotion_adv", 0.5f);
        st->have_builtin = !st->builtin_speaker_emb.empty() && !st->builtin_cond_tokens.empty();
    }

    g_states[lm] = st;
    return st;
}

bool ensure_tables(codec_lm * lm, CbxState * st) {
    const int32_t hidden = st->info.hidden_dim;
    if (st->text_emb.empty() &&
        !dequant_table(lm, "lm.chatterbox.text_emb.weight", &st->text_emb, nullptr, hidden)) {
        lm->last_error = "chatterbox: text_emb.weight missing";
        return false;
    }
    if (st->text_pos_emb.empty() &&
        !dequant_table(lm, "lm.chatterbox.text_pos_emb.weight", &st->text_pos_emb,
                       &st->text_pos_rows, hidden)) {
        lm->last_error = "chatterbox: text_pos_emb.weight missing";
        return false;
    }
    if (st->speech_emb.empty() &&
        !dequant_table(lm, "lm.audio_embd_0.weight", &st->speech_emb, nullptr, hidden)) {
        lm->last_error = "chatterbox: audio_embd_0.weight missing";
        return false;
    }
    if (st->speech_pos_emb.empty() &&
        !dequant_table(lm, "lm.chatterbox.speech_pos_emb.weight", &st->speech_pos_emb,
                       &st->speech_pos_rows, hidden)) {
        lm->last_error = "chatterbox: speech_pos_emb.weight missing";
        return false;
    }
    return true;
}

}  // namespace

// =====================================================================
// Public API
// =====================================================================

void codec_lm_chatterbox_free_state(struct codec_lm * lm) {
    auto it = g_states.find(lm);
    if (it != g_states.end()) {
        delete it->second;
        g_states.erase(it);
    }
}

const struct codec_lm_chatterbox_info *
codec_lm_chatterbox_get_info(struct codec_lm * lm) {
    CbxState * st = ensure_state(lm);
    if (st == nullptr || !st->have_info) return nullptr;
    return &st->info;
}

enum codec_status codec_lm_chatterbox_tokenize(
    struct codec_lm * lm, const char * text,
    int32_t * out_ids, int32_t cap, int32_t * n_out) {
    if (lm == nullptr || text == nullptr || out_ids == nullptr || n_out == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    CbxState * st = ensure_state(lm);
    if (st == nullptr || !st->info.has_tokenizer || !st->tok.loaded) {
        if (lm) lm->last_error = "chatterbox: no tokenizer baked into GGUF";
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    std::string normed = punc_norm(text);
    std::vector<int32_t> ids;
    tokenize_bpe(st->tok, normed, &ids);
    if ((int32_t) ids.size() > cap) {
        lm->last_error = "chatterbox: tokenize output exceeds capacity";
        return CODEC_STATUS_INVALID_ARG;
    }
    std::memcpy(out_ids, ids.data(), ids.size() * sizeof(int32_t));
    *n_out = (int32_t) ids.size();
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_lm_chatterbox_build_prompt(
    struct codec_lm * lm,
    const int32_t * text_ids, int32_t n_text, float cfg_weight,
    const float * speaker_emb, int32_t speaker_emb_dim,
    const int32_t * ref_speech_tokens, int32_t n_ref_speech_tokens,
    const float * emotion,
    const float * ref_pcm, int32_t ref_n_samples, int32_t ref_sample_rate,
    float * out_embeds, int32_t out_cap_rows,
    int32_t * out_seq_len, int32_t * out_n_seq) {
    if (lm == nullptr || out_embeds == nullptr || out_seq_len == nullptr || out_n_seq == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (n_text < 0 || (n_text > 0 && text_ids == nullptr)) return CODEC_STATUS_INVALID_ARG;
    CbxState * st = ensure_state(lm);
    if (st == nullptr) return CODEC_STATUS_NOT_SUPPORTED;
    if (!ensure_tables(lm, st)) return CODEC_STATUS_INTERNAL_ERROR;

    const int32_t hidden    = st->info.hidden_dim;
    const int32_t cond_rows = st->info.cond_rows;

    // ── cond_emb block (spkr + perceiver + emotion) ─────────────────
    const float *   spk_ptr   = speaker_emb;
    int32_t         spk_dim   = speaker_emb_dim;
    const int32_t * ref_ptr   = ref_speech_tokens;
    int32_t         ref_n     = n_ref_speech_tokens;
    float           emo_val   = emotion ? *emotion : st->builtin_emotion;
    if (spk_ptr == nullptr) {
        if (!st->have_builtin) {
            lm->last_error = "chatterbox: no speaker_emb and no builtin conds";
            return CODEC_STATUS_INVALID_ARG;
        }
        spk_ptr = st->builtin_speaker_emb.data();
        spk_dim = (int32_t) st->builtin_speaker_emb.size();
    }
    if (ref_ptr == nullptr) {
        ref_ptr = st->builtin_cond_tokens.data();
        ref_n   = (int32_t) st->builtin_cond_tokens.size();
    }

    std::vector<float> cond_emb((size_t) cond_rows * hidden, 0.0f);
    if (ref_pcm != nullptr && ref_n_samples > 0) {
        // Reference-audio path: run VE (mel + LSTM) + cond_enc to derive
        // the speaker prefix from the ref waveform.  The cond-prompt speech
        // tokens fall back to the builtin prompt (we don't tokenize the ref
        // audio through the S3 tokenizer here).
        codec_audio audio = {};
        audio.data        = ref_pcm;
        audio.n_samples   = ref_n_samples;
        audio.sample_rate = ref_sample_rate > 0 ? ref_sample_rate : 16000;
        audio.n_channels  = 1;
        audio.pcm_type    = CODEC_PCM_TYPE_F32;
        enum codec_status rc = chatterbox_speaker_encode(
            lm, &audio, ref_ptr, ref_n, emo_val,
            cond_emb.data(), (int32_t) cond_emb.size());
        if (rc != CODEC_STATUS_SUCCESS) return rc;
    } else {
        enum codec_status rc = chatterbox_speaker_encode_from_emb(
            lm, spk_ptr, ref_ptr, ref_n, emo_val,
            cond_emb.data(), (int32_t) cond_emb.size());
        if (rc != CODEC_STATUS_SUCCESS) return rc;
    }

    // ── text-wrapped sequence: [sot | text_ids | eot] ────────────────
    std::vector<int32_t> wrapped;
    wrapped.reserve((size_t) n_text + 2);
    wrapped.push_back(st->info.start_text_token);
    for (int32_t i = 0; i < n_text; ++i) wrapped.push_back(text_ids[i]);
    wrapped.push_back(st->info.stop_text_token);
    const int32_t n_wrapped = (int32_t) wrapped.size();

    // Layout per sequence: cond_rows + n_wrapped + 2 (speech_emb(6561)@0
    // from prepare_input_embeds AND the separately-appended BOS(6561)@0 —
    // reference emits both; see t3.py inference).
    const int32_t seq_len = cond_rows + n_wrapped + 2;
    const int32_t n_seq   = (cfg_weight > 0.0f) ? 2 : 1;
    if ((int64_t) seq_len * n_seq > out_cap_rows) {
        lm->last_error = "chatterbox: build_prompt output capacity too small";
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t text_pos_rows   = st->text_pos_rows;
    const int32_t speech_pos_rows = st->speech_pos_rows;
    const int32_t text_vocab      = st->info.text_vocab_size;
    const int32_t speech_vocab    = st->info.speech_vocab_size;
    const int32_t bos             = st->info.start_speech_token;

    for (int32_t s = 0; s < n_seq; ++s) {
        const bool uncond = (s == 1);
        float * base = out_embeds + (size_t) s * seq_len * hidden;
        int32_t row = 0;

        // cond block (identical across CFG lanes)
        std::memcpy(base + (size_t) row * hidden, cond_emb.data(),
                    (size_t) cond_rows * hidden * sizeof(float));
        row += cond_rows;

        // text block: text_emb[tok] + text_pos_emb[pos]; uncond zeros the
        // token content but keeps the positional embedding.
        for (int32_t p = 0; p < n_wrapped; ++p) {
            float * dst = base + (size_t) row * hidden;
            int32_t tok = wrapped[p];
            if (!uncond && tok >= 0 && tok < text_vocab) {
                const float * te = st->text_emb.data() + (size_t) tok * hidden;
                std::memcpy(dst, te, (size_t) hidden * sizeof(float));
            } else {
                std::memset(dst, 0, (size_t) hidden * sizeof(float));
            }
            if (p < text_pos_rows) {
                const float * pe = st->text_pos_emb.data() + (size_t) p * hidden;
                for (int32_t k = 0; k < hidden; ++k) dst[k] += pe[k];
            }
            ++row;
        }

        // speech_emb(bos)@pos0  (from prepare_input_embeds)
        // + separately appended BOS(bos)@pos0.  Two identical rows.
        for (int32_t rep = 0; rep < 2; ++rep) {
            float * dst = base + (size_t) row * hidden;
            if (bos >= 0 && bos < speech_vocab) {
                const float * se = st->speech_emb.data() + (size_t) bos * hidden;
                std::memcpy(dst, se, (size_t) hidden * sizeof(float));
            } else {
                std::memset(dst, 0, (size_t) hidden * sizeof(float));
            }
            if (speech_pos_rows > 0) {
                const float * pe = st->speech_pos_emb.data();  // row 0
                for (int32_t k = 0; k < hidden; ++k) dst[k] += pe[k];
            }
            ++row;
        }
    }

    *out_seq_len = seq_len;
    *out_n_seq   = n_seq;
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_lm_chatterbox_compose_speech_embd(
    struct codec_lm * lm, int32_t code, int32_t pos,
    float * out, int32_t out_cap) {
    if (lm == nullptr || out == nullptr) return CODEC_STATUS_INVALID_ARG;
    CbxState * st = ensure_state(lm);
    if (st == nullptr) return CODEC_STATUS_NOT_SUPPORTED;
    if (!ensure_tables(lm, st)) return CODEC_STATUS_INTERNAL_ERROR;
    const int32_t hidden = st->info.hidden_dim;
    if (out_cap < hidden) return CODEC_STATUS_INVALID_ARG;
    if (code < 0 || code >= st->info.speech_vocab_size) return CODEC_STATUS_INVALID_ARG;

    const float * se = st->speech_emb.data() + (size_t) code * hidden;
    std::memcpy(out, se, (size_t) hidden * sizeof(float));
    if (pos >= 0 && pos < st->speech_pos_rows) {
        const float * pe = st->speech_pos_emb.data() + (size_t) pos * hidden;
        for (int32_t k = 0; k < hidden; ++k) out[k] += pe[k];
    }
    return CODEC_STATUS_SUCCESS;
}
