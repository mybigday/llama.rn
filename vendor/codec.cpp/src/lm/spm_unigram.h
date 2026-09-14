#ifndef CODEC_LM_SPM_UNIGRAM_H
#define CODEC_LM_SPM_UNIGRAM_H

#include <cstdint>
#include <string>
#include <vector>

// Minimal SentencePiece UNIGRAM tokenizer.
//
// Pocket-TTS ships a SentencePiece unigram model (vocab 4000, byte_fallback,
// identity normalizer, add_dummy_prefix, escape_whitespaces).  We can't link
// libsentencepiece into libcodec, so this parses the model protobuf (the raw
// `.model` bytes, base64-decoded from the `codec.lm.tokenizer.spm_b64` GGUF KV)
// and implements Viterbi unigram encoding + byte-fallback.  NO runtime python.
//
// Only the pieces (string + score + type) and a couple of trainer/normalizer
// flags are read from the protobuf; the precompiled charsmap is empty for this
// model (identity normalizer), so no Unicode transform is needed.
class SpmUnigram {
public:
    // Parse from the raw SentencePiece .model protobuf bytes.  Returns false on
    // a malformed buffer.
    bool load(const uint8_t * data, size_t n);
    bool loaded() const { return !pieces_.empty(); }

    int32_t vocab_size() const { return (int32_t) pieces_.size(); }

    // Encode `text` to token ids, mirroring sp.encode(text, out_type=int):
    // escape whitespace (' ' -> U+2581), add_dummy_prefix, then Viterbi over
    // the unigram vocabulary with per-byte fallback for unknown spans.
    std::vector<int32_t> encode(const std::string & text) const;

private:
    enum PieceType { NORMAL = 1, UNKNOWN = 2, CONTROL = 3, USER_DEFINED = 4,
                     BYTE = 6, UNUSED = 5 };
    struct Piece { std::string piece; float score; int32_t type; };

    std::vector<Piece>  pieces_;
    // piece string -> id (all pieces, for prefix matching we use a length-bounded
    // scan against the map).
    // Byte-fallback base: id of "<0x00>", or -1 if absent.
    int32_t byte0_id_ = -1;
    int32_t unk_id_    = 0;
    int32_t max_piece_len_ = 1;
    float   min_score_ = 0.0f;   // over NORMAL/USER pieces; byte fallback uses min_score-10

    // Fast lookup: map normalized-piece-string -> id.
    // (std::unordered_map kept in the .cpp to avoid pulling <unordered_map> here.)
    void * lookup_ = nullptr;  // opaque std::unordered_map<std::string,int32_t>*
    int32_t find_piece(const std::string & s) const;

public:
    ~SpmUnigram();
    SpmUnigram() = default;
    SpmUnigram(const SpmUnigram &) = delete;
    SpmUnigram & operator=(const SpmUnigram &) = delete;
};

#endif // CODEC_LM_SPM_UNIGRAM_H
