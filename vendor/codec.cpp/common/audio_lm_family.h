#ifndef CODEC_AUDIO_LM_FAMILY_H
#define CODEC_AUDIO_LM_FAMILY_H

// Internal: the model families behind include/codec_audio_lm.h.  Each family
// turns a request into a completion description (prompt chunks, sampler,
// constraints, host flags) and interprets every AR step; the public C API
// (audio_lm_c.cpp) only forwards.  Nothing here calls llama.cpp.

#include "codec_audio_lm.h"
#include "codec_common.h"

#include <memory>
#include <string>
#include <vector>

namespace codec_common {

// The host's backbone vocabulary (codec_audio_lm_vocab) with C++ helpers.
struct alm_vocab {
    const codec_audio_lm_vocab * v = nullptr;

    bool    valid() const { return v != nullptr && v->tokenize != nullptr && v->token_to_piece != nullptr; }
    int32_t n_tokens() const { return v ? v->n_tokens : 0; }
    int32_t eos() const { return v ? v->eos : -1; }
    bool    is_eog(int32_t t) const { return v && (v->is_eog ? v->is_eog(v->user_data, t) : t == v->eos); }
    std::vector<int32_t> tokenize(const std::string & text, bool add_special, bool parse_special) const;
    std::string          piece(int32_t token) const;
    // Id of the token whose piece is exactly `text`, or -1.  Linear scan.
    int32_t              find_piece(const std::string & text) const;
};

// A prompt chunk the family owns (codec_audio_lm_chunk points into it).
struct alm_chunk {
    codec_audio_lm_chunk_type type = CODEC_AUDIO_LM_CHUNK_TEXT;
    std::string          text;
    bool                 add_special   = false;
    bool                 parse_special = false;
    std::vector<int32_t> tokens;
    std::vector<float>   embd;   // rows x n_embd
    int32_t              n_rows = 0;

    static alm_chunk of_text(std::string t, bool add_special, bool parse_special);
    static alm_chunk of_tokens(std::vector<int32_t> t);
    static alm_chunk of_embd(std::vector<float> e, int32_t n_rows);
};

// A group of codec-side heads the host samples with one sampler
// (codec_audio_lm_head_group).
struct alm_head_group {
    bool                 penalties = true;
    std::vector<int32_t> history;
    std::vector<int32_t> penalty_exempt;
};

struct alm_completion {
    codec_audio_lm_flow flow = CODEC_AUDIO_LM_FLOW_TOKENS;
    int32_t n_embd = 0;
    std::vector<std::vector<alm_chunk>> seqs;   // one chunk list per sequence
    // Suggested constraints of the host's backbone sampler.
    std::vector<int32_t> penalty_exempt;
    std::string grammar;
    std::vector<codec_audio_lm_token_range> allowed;
    bool    needs_embeddings    = false;
    bool    needs_prompt_hidden = false;
    bool    sample_first        = true;
    bool    streamable          = false;
    int32_t max_frames          = 0;
    std::vector<alm_head_group> head_groups;   // codec-side heads (SAMPLE_HEAD)
};

struct alm_tts_request {
    std::string text;
    std::string instruct;
    const codec_speaker * voice = nullptr;   // resolved: the request's, else the model's default (may be null)
    // false: a build without the request's text (the host kept only the
    // prompt); format then skips what depends on the text (its grammar,
    // length estimates) and must not fail on the empty text.
    bool text_known = true;
    int32_t min_frames = -1;         // < 0 = the model's own minimum
    int32_t max_frames = 0;          // <= 0 = model default
    float   lm_guidance = -1.0f;     // < 0 = the model's (likewise below)
    float   flow_guidance = -1.0f;
    int32_t flow_steps = -1;

};

// A formatted prompt (codec_audio_lm_tts_format) and what the completion of
// its build carries apart from the chunks.
struct alm_format {
    std::string    prompt;
    int32_t        n_seq = 1;
    alm_completion desc;   // seqs empty
};

// The marker of a model row / boundary in a prompt: <__codec:NAME__>.
std::string alm_codec_marker(const std::string & name);

// A prompt split at its markers, read front to back by a family's build.
class alm_prompt {
public:
    // false (err set) on a malformed marker.
    bool parse(const std::string & prompt, std::string * err);
    bool done() const { return pos_ >= pieces_.size(); }
    // The next piece when it is text (consumed), else "" (nothing consumed).
    std::string text();
    // Consume a <__voice__> / <__codec:NAME__> marker when it is next.
    bool voice();
    bool codec(const char * name);
    // The next piece's marker name ("voice" / "codec:NAME"), or "" for text / the end.
    std::string next_marker() const;
    // Text of a markerless prompt: the remaining pieces, false when a marker remains.
    bool rest_text(std::string * out);

private:
    struct piece { bool marker; std::string s; };   // s: the text, or "voice" / "codec:NAME"
    std::vector<piece> pieces_;
    size_t pos_ = 0;
};

// The seed of a model-owned random stream: `seed`, or a random one for
// CODEC_AUDIO_LM_DEFAULT_SEED.
uint32_t alm_resolve_seed(uint32_t seed);

// GBNF for a minimum length on backbone-sampled tokens: `min_tokens` tokens
// other than `end` before `end` (an end-of-generation token or not) may follow.  `body` is the grammar of one
// token (empty: any token but `end`); returns "" when min_tokens <= 0.
std::string alm_min_length_grammar(int32_t end, int32_t min_tokens, const std::string & body = "");

// A generation's decoder input (codec_audio_lm_generation): codec quantizer
// codes or frame-major latents, [n_frames, width].
struct alm_generation {
    bool                 latent   = false;
    int32_t              n_frames = 0;
    int32_t              width    = 0;
    std::vector<int32_t> codes;
    std::vector<float>   latents;
};

// One model family.  Lifetime: owned by the codec_audio_lm context; `alm`
// (codec + codec_lm) outlives it.
class alm_family {
public:
    explicit alm_family(audio_lm_context * alm) : alm_(alm) {}
    std::string backbone_path;   // set by the factory (models that read its embedding table)
    virtual ~alm_family() = default;
    // Takes over the read-only data `base` (the same model's family) loaded or
    // will load, instead of a copy (codec_audio_lm_init_shared).
    virtual void share_from(alm_family & /*base*/) {}

    virtual const char * name() const = 0;
    virtual void caps(codec_audio_lm_caps * out) const;
    virtual bool backbone_requirements(const alm_vocab & vocab, int32_t max_frames,
                                       codec_audio_lm_backbone_req * out);
    // A voice cloned from reference audio (codec_audio_lm_create_voice):
    // the codec's speaker_from_audio computes every slot the model reads.
    virtual codec_speaker * create_voice(const float * pcm, int32_t n_samples, int32_t sample_rate,
                                         const char * ref_text);
    // Segmentation (codec_audio_lm_tts_split) and joining (tts_join): one
    // general behaviour every model gets, tuned by the family's parameters;
    // a model that needs a different method picks another mode (or, in the
    // last resort, overrides split / join).
    enum class split_mode {
        clause,          // tts_split_text: sentences / clauses by UTF-8 bytes, any script
        sentence_pack,   // kittentts2 text.py: English sentences packed to max, merged up to min
    };
    struct split_params {
        split_mode mode      = split_mode::clause;
        int32_t    max_bytes = 0;     // 0: the whole text unless the host asks for a size
        int32_t    min_bytes = 0;     // sentence_pack: merge shorter chunks into a neighbour
    };
    struct join_params {
        bool    trim    = true;       // drop leading / trailing silence
        int32_t fade_ms = 8;          // fade both ends
        int32_t gap_ms  = 160;        // silence between segments
    };
    virtual split_params segment_split() const { return {}; }
    virtual join_params  segment_join() const { return {}; }
    virtual std::vector<std::string> split(const std::string & text, int32_t max_bytes);
    virtual int32_t join(std::vector<float> & pcm, int32_t sample_rate, int32_t i, int32_t n);
    // Text to speech: the prompt as text with markers, plus the completion's
    // constraints and flags (alm_format.desc), without generation state; then
    // its build: `out` arrives as the format's desc and gets the chunks, while
    // the family sets up the generation.  `prompt` must be read to its end.
    virtual bool tts_format(const alm_tts_request & req, const alm_vocab & vocab, alm_format * out) = 0;
    virtual bool tts_build(const alm_tts_request & req, alm_prompt & prompt, const alm_vocab & vocab,
                           alm_completion * out) = 0;
    virtual bool prefill(const float * hiddens, int32_t n_pos);
    // codec_audio_lm_set_seed: the model's own noise (flow / CFM heads).
    virtual void set_seed(uint32_t) {}
    virtual bool observe(const int32_t * tokens, const float * hiddens, codec_audio_lm_verdict * v) = 0;
    // After a SAMPLE_HEAD verdict: the head's logits, and the id the host drew.
    virtual const float * head_logits() const { return nullptr; }
    virtual bool head_token(int32_t, codec_audio_lm_verdict *) { return fail("this model has no codec-side heads"); }
    // Frames generated so far that decode() would turn into audio.
    virtual int32_t decodable_frames() const { return audio_lm_decodable_frames(alm_); }
    virtual const float * next_embd(int32_t seq) const;
    // Set by observe() on STOP; "max_frames" until then.
    const std::string & stop_reason() const { return stop_reason_; }
    virtual bool generate_self_contained();
    // The decoder input of the current generation.  Default: the codes /
    // latents accumulated in the audio_lm context, mapped to codec input.
    virtual bool generation(alm_generation * out);
    // Decode a generation (this one or an earlier one) with the decoder
    // speaker, without touching the current one.  Default: codec_decode of
    // the codes, or the codec's latent decode of the frame-major latents.
    virtual bool decode_generation(const alm_generation & g, audio_lm_audio_output * out);
    bool decode(audio_lm_audio_output * out) {
        alm_generation g;
        return generation(&g) && decode_generation(g, out);
    }

    const std::string & error() const { return error_; }

protected:
    bool fail(const std::string & e) { error_ = e; return false; }
    bool fail_alm(const char * what);   // e + audio_lm_last_error
    bool generation_from_codes(const std::vector<int32_t> & codes, alm_generation * out);

    audio_lm_context * alm_;
    std::string        error_;
    std::string        stop_reason_ = "max_frames";
};

// Wrap an audio_lm context the caller keeps owning (tts_runner while its
// families move behind the public API).  nullptr when no family fits.
struct codec_audio_lm * alm_wrap_borrowed(audio_lm_context * alm, const std::string & backbone_path,
                                         std::string * err);

// The family for a loaded codec GGUF, or nullptr (err set) when the codec is
// not the audio side of a supported model.
std::unique_ptr<alm_family> alm_make_family(audio_lm_context * alm, const std::string & backbone_path,
                                            std::string * err);

// Families (audio_lm_family_*.cpp).
std::unique_ptr<alm_family> alm_family_soprano(audio_lm_context * alm);
std::unique_ptr<alm_family> alm_family_neutts(audio_lm_context * alm);
std::unique_ptr<alm_family> alm_family_outetts(audio_lm_context * alm);
std::unique_ptr<alm_family> alm_family_kitten(audio_lm_context * alm);
bool alm_is_kitten(audio_lm_context * alm);
std::unique_ptr<alm_family> alm_family_codebook(audio_lm_context * alm);
std::unique_ptr<alm_family> alm_family_realtime(audio_lm_context * alm);
std::unique_ptr<alm_family> alm_family_lfm2(audio_lm_context * alm);
std::unique_ptr<alm_family> alm_family_chatterbox(audio_lm_context * alm, struct codec_lm * lm);
std::unique_ptr<alm_family> alm_family_cosyvoice(audio_lm_context * alm, struct codec_lm * lm);
std::unique_ptr<alm_family> alm_family_continuous(audio_lm_context * alm);
std::unique_ptr<alm_family> alm_family_flow(audio_lm_context * alm, struct codec_lm * lm);

}  // namespace codec_common

#endif  // CODEC_AUDIO_LM_FAMILY_H
