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

struct alm_completion {
    codec_audio_lm_flow flow = CODEC_AUDIO_LM_FLOW_TOKENS;
    int32_t n_embd = 0;
    std::vector<std::vector<alm_chunk>> seqs;   // one chunk list per sequence
    codec_audio_lm_sampler sampler{};
    std::vector<int32_t> penalty_exempt;
    std::string          samplers;   // order; "" = llama.cpp default
    std::string grammar;
    std::vector<codec_audio_lm_token_range> allowed;
    bool    needs_embeddings    = false;
    bool    needs_prompt_hidden = false;
    bool    sample_first        = true;
    bool    streamable          = false;
    int32_t max_frames          = 0;
    int32_t min_frames          = 0;
    int32_t stop_token          = -1;

    alm_completion();   // sampler "off" values
};

struct alm_tts_request {
    std::string text;
    std::string instruct;
    std::string preset;
    uint32_t seed = 0;               // 0 = default
    float   temperature = -1.0f;     // < 0 = model default (likewise below)
    float   top_p = -1.0f;
    int32_t top_k = -1;
    float   min_p = -1.0f;
    float   repetition_penalty = -1.0f;
    int32_t min_frames = -1;
    int32_t max_frames = 0;          // <= 0 = model default
    float   lm_guidance = -1.0f;
    float   flow_guidance = -1.0f;
    int32_t flow_steps = -1;

    uint32_t seed_or_default() const { return seed ? seed : 0xC0DEC1ABu; }
    // The request's overrides on top of a model's sampler defaults.
    void apply(codec_audio_lm_sampler & sp) const;
};

// A generation's decoder input (codec_audio_lm_generation): codec quantizer
// codes or frame-major latents, [n_frames, width].
struct alm_generation {
    bool                 latent   = false;
    int32_t              n_frames = 0;
    int32_t              width    = 0;
    std::vector<int32_t> codes;
    std::vector<float>   latents;
};

struct alm_tts_voice {
    std::string        voice;
    std::string        speaker_file;
    std::vector<float> ref_pcm;
    int32_t            ref_sample_rate = 0;
    std::string        ref_text;
    std::string        save_speaker;
};

// One model family.  Lifetime: owned by the codec_audio_lm context; `alm`
// (codec + codec_lm) outlives it.
class alm_family {
public:
    explicit alm_family(audio_lm_context * alm) : alm_(alm) {}
    std::string backbone_path;   // set by the factory (models that read its embedding table)
    virtual ~alm_family() = default;

    virtual const char * name() const = 0;
    virtual void caps(codec_audio_lm_caps * out) const;
    virtual bool backbone_requirements(const alm_vocab & vocab, int32_t max_frames,
                                       codec_audio_lm_backbone_req * out);
    // Default: the decoder speaker (speaker file > ref audio > voice > default).
    virtual bool set_voice(const alm_tts_voice & v);
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
    virtual bool tts_prompt(const alm_tts_request & req, const alm_vocab & vocab, alm_completion * out) = 0;
    virtual bool prefill(const float * hiddens, int32_t n_pos);
    virtual bool observe(const int32_t * tokens, const float * hiddens, codec_audio_lm_verdict * v) = 0;
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
    alm_tts_voice      voice_;
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
