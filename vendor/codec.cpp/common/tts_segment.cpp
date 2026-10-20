// tts_segment.cpp — split synthesis input into segments a model handles well.
//
// TTS models degrade on long inputs (drift, skipped words, running into the
// frame cap), so the runner synthesizes long text a segment at a time.  The
// budget is in UTF-8 bytes, which tracks speaking time across scripts (a CJK
// character is 3 bytes and takes about as long to say as 3 Latin letters).

#include "tts_runner.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace codec_common {

namespace {

// Code point at s[i] (0 on a malformed byte) and its length.
uint32_t utf8_at(const std::string & s, size_t i, size_t * len) {
    const unsigned char c = (unsigned char) s[i];
    size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    if (i + n > s.size()) n = 1;
    *len = n;
    if (n == 1) return c < 0x80 ? c : 0;
    uint32_t cp = c & (0xFF >> (n + 1));
    for (size_t k = 1; k < n; ++k) cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
    return cp;
}

bool is_space(uint32_t cp) { return cp == ' ' || cp == '\t' || cp == '\r' || cp == '\n' || cp == 0x3000; }

bool is_sentence_end(uint32_t cp) {
    switch (cp) {
        case '.': case '!': case '?': case ';':
        case 0x3002: case 0xFF01: case 0xFF1F: case 0xFF1B:   // 。！？；
        case 0x2026: case 0xFF0E:                              // … ．
            return true;
        default:
            return false;
    }
}

bool is_clause_end(uint32_t cp) {
    switch (cp) {
        case ',': case ':':
        case 0xFF0C: case 0x3001: case 0xFF1A:                 // ，、：
        case 0x2014:                                           // —
            return true;
        default:
            return false;
    }
}

// Closing quotes / brackets that stay with the punctuation before them.
bool is_closer(uint32_t cp) {
    switch (cp) {
        case '"': case '\'': case ')': case ']':
        case 0x201D: case 0x2019: case 0x300D: case 0x300F: case 0xFF09: case 0x3011:  // ” ’ 」 』 ） 】
            return true;
        default:
            return false;
    }
}

std::string trim(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e) { size_t n; if (!is_space(utf8_at(s, b, &n))) break; b += n; }
    while (e > b) {
        size_t k = e - 1;
        while (k > b && ((unsigned char) s[k] & 0xC0) == 0x80) --k;
        size_t n; if (!is_space(utf8_at(s, k, &n))) break;
        e = k;
    }
    return s.substr(b, e - b);
}

// Pieces ending after each break point `pred` accepts (closers included).
template <typename Pred>
std::vector<std::string> split_after(const std::string & s, Pred pred) {
    std::vector<std::string> out;
    size_t start = 0, i = 0;
    while (i < s.size()) {
        size_t n;
        const uint32_t cp = utf8_at(s, i, &n);
        i += n;
        if (!pred(cp)) continue;
        // Runs like "?!" or "..." and trailing closers stay in this piece.
        while (i < s.size()) {
            size_t m;
            const uint32_t nx = utf8_at(s, i, &m);
            if (!pred(nx) && !is_closer(nx)) break;
            i += m;
        }
        out.push_back(s.substr(start, i - start));
        start = i;
    }
    if (start < s.size()) out.push_back(s.substr(start));
    return out;
}

// Cut one over-long piece at clause punctuation, then at spaces.  A piece
// with neither (an unpunctuated CJK run) stays whole: cutting it would split
// a word, which sounds worse than an over-long segment.
void split_long(const std::string & s, size_t max_bytes, std::vector<std::string> * out) {
    if (s.size() <= max_bytes) { out->push_back(s); return; }
    for (int level = 0; level < 2; ++level) {
        const auto parts = level == 0 ? split_after(s, is_clause_end) : split_after(s, is_space);
        if (parts.size() < 2) continue;
        std::string cur;
        for (const auto & p : parts) {
            if (!cur.empty() && cur.size() + p.size() > max_bytes) {
                split_long(cur, max_bytes, out);
                cur.clear();
            }
            cur += p;
        }
        if (!cur.empty()) split_long(cur, max_bytes, out);
        return;
    }
    std::fprintf(stderr, "segment: %zu bytes with no punctuation or space to split at; kept whole (budget %zu)\n",
                 s.size(), max_bytes);
    out->push_back(s);
}

}  // namespace

std::vector<std::string> tts_split_text(const std::string & text, int32_t max_bytes) {
    std::vector<std::string> segments;
    if (max_bytes <= 0) {
        const std::string t = trim(text);
        if (!t.empty()) segments.push_back(t);
        return segments;
    }
    // Paragraphs (blank-line separated) always start a new segment.
    std::vector<std::string> paragraphs(1);
    for (size_t start = 0; start <= text.size();) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        const std::string line = text.substr(start, end - start);
        if (trim(line).empty()) {
            if (!paragraphs.back().empty()) paragraphs.emplace_back();
        } else {
            std::string & para = paragraphs.back();
            const std::string t = trim(line);
            para += (para.empty() || (para.back() & 0x80) ? "" : " ") + t;
        }
        start = end + 1;
    }
    for (const auto & para : paragraphs) {
        // Sentences packed greedily up to the budget.
        std::string cur;
        auto flush = [&]() {
            const std::string t = trim(cur);
            if (!t.empty()) segments.push_back(t);
            cur.clear();
        };
        for (const auto & sent : split_after(para, is_sentence_end)) {
            const std::string t = trim(sent);
            if (t.empty()) continue;
            if (t.size() > (size_t) max_bytes) {
                flush();
                std::vector<std::string> parts;
                split_long(t, (size_t) max_bytes, &parts);
                for (const auto & p : parts) { cur = p; flush(); }
                continue;
            }
            // Keep a space between packed sentences unless the script uses none.
            const std::string joined = cur.empty() ? t : cur + (cur.back() & 0x80 ? "" : " ") + t;
            if (!cur.empty() && joined.size() > (size_t) max_bytes) flush();
            cur = cur.empty() ? t : joined;
        }
        flush();
    }
    return segments;
}

}  // namespace codec_common
