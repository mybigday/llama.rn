#include "spm_unigram.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

// =====================================================================
// SentencePiece UNIGRAM tokenizer (Viterbi + byte fallback).
//
// Protobuf layout (subset we parse):
//   ModelProto {
//     repeated SentencePiece pieces = 1;   // { string piece=1; float score=2; int32 type=3 }
//     TrainerSpec  trainer_spec = 2;        // ignored (we infer from pieces)
//     NormalizerSpec normalizer_spec = 3;   // ignored (identity for this model)
//   }
// Wire types: 0 varint, 1 64-bit, 2 length-delimited, 5 32-bit.
// =====================================================================

namespace {

const char * kUnderscore = "\xE2\x96\x81";   // U+2581 LOWER ONE EIGHTH BLOCK (▁)

// Read a base-128 varint at *p (advancing p); returns false past `end`.
bool read_varint(const uint8_t *& p, const uint8_t * end, uint64_t & out) {
    uint64_t v = 0; int shift = 0;
    while (p < end) {
        uint8_t b = *p++;
        v |= (uint64_t) (b & 0x7f) << shift;
        if (!(b & 0x80)) { out = v; return true; }
        shift += 7;
        if (shift >= 64) return false;
    }
    return false;
}

// Parse one SentencePiece submessage (fields: 1=piece(str), 2=score(float32), 3=type(varint)).
bool parse_piece(const uint8_t * p, const uint8_t * end, std::string & piece, float & score, int32_t & type) {
    piece.clear(); score = 0.0f; type = 1;
    while (p < end) {
        uint64_t tag;
        if (!read_varint(p, end, tag)) return false;
        const uint32_t field = (uint32_t) (tag >> 3);
        const uint32_t wire  = (uint32_t) (tag & 7);
        if (field == 1 && wire == 2) {
            uint64_t len; if (!read_varint(p, end, len)) return false;
            if (p + len > end) return false;
            piece.assign((const char *) p, (size_t) len);
            p += len;
        } else if (field == 2 && wire == 5) {
            if (p + 4 > end) return false;
            float f; std::memcpy(&f, p, 4); score = f; p += 4;
        } else if (field == 3 && wire == 0) {
            uint64_t t; if (!read_varint(p, end, t)) return false;
            type = (int32_t) t;
        } else {
            // Skip unknown field by wire type.
            if (wire == 0) { uint64_t s; if (!read_varint(p, end, s)) return false; }
            else if (wire == 2) { uint64_t l; if (!read_varint(p, end, l)) return false; if (p + l > end) return false; p += l; }
            else if (wire == 5) { if (p + 4 > end) return false; p += 4; }
            else if (wire == 1) { if (p + 8 > end) return false; p += 8; }
            else return false;
        }
    }
    return true;
}

}  // namespace

SpmUnigram::~SpmUnigram() {
    delete static_cast<std::unordered_map<std::string, int32_t> *>(lookup_);
}

int32_t SpmUnigram::find_piece(const std::string & s) const {
    auto * m = static_cast<std::unordered_map<std::string, int32_t> *>(lookup_);
    auto it = m->find(s);
    return it == m->end() ? -1 : it->second;
}

bool SpmUnigram::load(const uint8_t * data, size_t n) {
    pieces_.clear();
    delete static_cast<std::unordered_map<std::string, int32_t> *>(lookup_);
    auto * m = new std::unordered_map<std::string, int32_t>();
    lookup_ = m;

    const uint8_t * p = data;
    const uint8_t * end = data + n;
    while (p < end) {
        uint64_t tag;
        if (!read_varint(p, end, tag)) break;
        const uint32_t field = (uint32_t) (tag >> 3);
        const uint32_t wire  = (uint32_t) (tag & 7);
        if (field == 1 && wire == 2) {   // pieces
            uint64_t len; if (!read_varint(p, end, len)) return false;
            if (p + len > end) return false;
            std::string piece; float score; int32_t type;
            if (!parse_piece(p, p + len, piece, score, type)) return false;
            const int32_t id = (int32_t) pieces_.size();
            pieces_.push_back({ piece, score, type });
            p += len;
        } else {
            if (wire == 0) { uint64_t s; if (!read_varint(p, end, s)) return false; }
            else if (wire == 2) { uint64_t l; if (!read_varint(p, end, l)) return false; if (p + l > end) return false; p += l; }
            else if (wire == 5) { if (p + 4 > end) return false; p += 4; }
            else if (wire == 1) { if (p + 8 > end) return false; p += 8; }
            else return false;
        }
    }
    if (pieces_.empty()) return false;

    // Build lookup (skip CONTROL / UNKNOWN pieces from normal matching, but keep
    // BYTE pieces for byte fallback).  Record byte0 base and unk id.
    for (int32_t i = 0; i < (int32_t) pieces_.size(); ++i) {
        const Piece & pc = pieces_[(size_t) i];
        if (pc.type == UNKNOWN) unk_id_ = i;
        if (pc.type == BYTE) {
            // "<0x00>" is the first byte piece.
            if (pc.piece == "<0x00>") byte0_id_ = i;
            continue;   // byte pieces are matched via byte fallback, not text
        }
        if (pc.type == CONTROL || pc.type == UNKNOWN) continue;
        // NORMAL / USER_DEFINED: index by exact string.
        if (!pc.piece.empty()) {
            (*m)[pc.piece] = i;
            if ((int32_t) pc.piece.size() > max_piece_len_) max_piece_len_ = (int32_t) pc.piece.size();
            if (pc.score < min_score_) min_score_ = pc.score;
        }
    }
    return true;
}

std::vector<int32_t> SpmUnigram::encode(const std::string & text) const {
    std::vector<int32_t> out;
    if (pieces_.empty()) return out;

    // Normalize: escape whitespace (' ' -> ▁) with add_dummy_prefix.  This model
    // uses identity normalizer + remove_extra_whitespaces=false, so we only map
    // spaces and prepend a leading ▁ (matching sp.encode).
    std::string norm;
    norm.reserve(text.size() + 4);
    norm += kUnderscore;                 // add_dummy_prefix
    for (char c : text) {
        if (c == ' ') norm += kUnderscore;
        else norm.push_back(c);
    }

    const int32_t N = (int32_t) norm.size();
    // Viterbi over byte positions.  best[i] = best score to reach byte i;
    // back[i] = (start_pos, piece_id) of the last piece ending at i.
    const float NEG_INF = -std::numeric_limits<float>::infinity();
    std::vector<float>   best((size_t) N + 1, NEG_INF);
    std::vector<int32_t> back_pos((size_t) N + 1, -1);
    std::vector<int32_t> back_id((size_t) N + 1, -1);
    best[0] = 0.0f;

    // Unknown-span penalty for byte fallback: SentencePiece assigns byte pieces a
    // trained score; we look them up.  Fallback covers exactly one input byte.
    for (int32_t i = 0; i < N; ++i) {
        if (best[(size_t) i] == NEG_INF) continue;
        // Try all vocabulary pieces starting at i (bounded by max_piece_len_).
        const int32_t max_len = (i + max_piece_len_ <= N) ? max_piece_len_ : (N - i);
        bool matched_any = false;
        for (int32_t len = max_len; len >= 1; --len) {
            const int32_t id = find_piece(norm.substr((size_t) i, (size_t) len));
            if (id < 0) continue;
            matched_any = true;
            const float sc = best[(size_t) i] + pieces_[(size_t) id].score;
            const int32_t j = i + len;
            if (sc > best[(size_t) j]) {
                best[(size_t) j] = sc;
                back_pos[(size_t) j] = i;
                back_id[(size_t) j] = id;
            }
        }
        // Byte fallback for the single byte at i (always available so the lattice
        // never dead-ends).  SentencePiece emits <0xXX> for each raw byte.
        (void) matched_any;
        if (byte0_id_ >= 0) {
            const uint8_t byte = (uint8_t) norm[(size_t) i];
            const int32_t id = byte0_id_ + (int32_t) byte;
            // Byte fallback carries a heavy penalty (min piece score - 10) so a
            // real matching piece always wins; matches SentencePiece's unk_score.
            const float sc = best[(size_t) i] + (min_score_ - 10.0f);
            const int32_t j = i + 1;
            if (sc > best[(size_t) j]) {
                best[(size_t) j] = sc;
                back_pos[(size_t) j] = i;
                back_id[(size_t) j] = id;
            }
        } else {
            // No byte fallback: emit UNK for one byte.
            const int32_t j = i + 1;
            const float sc = best[(size_t) i] + pieces_[(size_t) unk_id_].score - 10.0f;
            if (sc > best[(size_t) j]) {
                best[(size_t) j] = sc;
                back_pos[(size_t) j] = i;
                back_id[(size_t) j] = unk_id_;
            }
        }
    }

    // Backtrace.
    std::vector<int32_t> rev;
    int32_t pos = N;
    while (pos > 0 && back_pos[(size_t) pos] >= 0) {
        rev.push_back(back_id[(size_t) pos]);
        pos = back_pos[(size_t) pos];
    }
    out.assign(rev.rbegin(), rev.rend());
    return out;
}
