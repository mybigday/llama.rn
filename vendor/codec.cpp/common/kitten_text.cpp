#include "kitten_text.h"
#include "tts_segment.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <regex>

namespace codec_common {

namespace {

const char * const kExpressionTags[] = {
    "mundane", "nervous", "tender", "angry", "excited",
    "stern", "sad", "contemplative", "surprised", "joyful",
};
const char * const kVocalEventTags[] = {
    "pause", "sigh", "gasp", "laugh", "giggle",
    "sob", "scoff", "growl", "um", "gulp",
};

std::string lower(std::string s) {
    for (char & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

std::string rstrip(const std::string & s) {
    size_t e = s.size();
    while (e > 0 && std::isspace((unsigned char) s[e - 1])) --e;
    return s.substr(0, e);
}

std::string strip(const std::string & s) {
    size_t b = 0;
    while (b < s.size() && std::isspace((unsigned char) s[b])) ++b;
    return rstrip(s.substr(b));
}

// Closing marks that already end a sentence: . ! ? " ' ) ] and the curly ’ ”.
bool already_terminated(const std::string & s) {
    if (s.empty()) return false;
    const char c = s.back();
    if (c == '.' || c == '!' || c == '?' || c == '"' || c == '\'' || c == ')' || c == ']') return true;
    return s.size() >= 3 && (s.compare(s.size() - 3, 3, "\xE2\x80\x99") == 0 ||
                             s.compare(s.size() - 3, 3, "\xE2\x80\x9D") == 0);
}

std::string capitalize_and_terminate(std::string text) {
    if (text.empty()) return text;
    text = std::regex_replace(text, std::regex(R"(\s+([?!]))"), "$1");
    for (char & c : text) {
        if (std::isalpha((unsigned char) c)) { c = (char) std::toupper((unsigned char) c); break; }
    }
    std::string s = rstrip(text);
    if (s.empty()) return text;
    if (already_terminated(s)) return s;
    s = rstrip(std::regex_replace(s, std::regex(R"([,;:\-]+\s*$)"), ""));
    return s.empty() ? s : s + ".";
}

std::string dashes_to_commas(std::string text) {
    if (text.empty()) return text;
    text = replace_all(text, "\xE2\x80\x94", "--");   // em dash
    text = replace_all(text, "\xE2\x80\x93", "--");   // en dash
    text = std::regex_replace(text, std::regex(R"(\s*--\s*)"), ", ");
    text = std::regex_replace(text, std::regex(R"(,\s*,+)"), ", ");
    text = std::regex_replace(text, std::regex(R"(\s+([,.!?;:]))"), "$1");
    text = std::regex_replace(text, std::regex(R"(,\s*([.!?]))"), "$1");
    text = std::regex_replace(text, std::regex(R"(\s{2,})"), " ");
    text = std::regex_replace(text, std::regex(R"(^\s*,\s*)"), "");
    return strip(text);
}

std::string ensure_terminal_punct(const std::string & chunk) {
    std::string c = rstrip(chunk);
    if (c.empty()) return c;
    const char last = c.back();
    if (std::string(".!?,;:\"')]").find(last) != std::string::npos) return c;
    if (c.size() >= 3 && (c.compare(c.size() - 3, 3, "\xE2\x80\x99") == 0 ||
                          c.compare(c.size() - 3, 3, "\xE2\x80\x9D") == 0)) {
        return c;
    }
    return c + ",";
}

std::vector<std::string> merge_short(std::vector<std::string> out, int max_chars, int min_chars) {
    if (out.empty() || min_chars <= 0) return out;
    const size_t limit = (size_t) (max_chars * 1.4);
    bool changed = true;
    while (changed && out.size() > 1) {
        changed = false;
        for (size_t i = 0; i < out.size(); ++i) {
            const std::string & chunk = out[i];
            if ((int) chunk.size() >= min_chars) continue;
            const bool prev_ok = i > 0 && out[i - 1].size() + 1 + chunk.size() <= limit;
            const bool next_ok = i + 1 < out.size() && chunk.size() + 1 + out[i + 1].size() <= limit;
            if (prev_ok && (!next_ok || out[i - 1].size() <= out[i + 1].size())) {
                out[i - 1] = rstrip(out[i - 1]) + " " + strip(chunk);
                out.erase(out.begin() + (long) i);
            } else if (next_ok) {
                out[i] = rstrip(chunk) + " " + strip(out[i + 1]);
                out.erase(out.begin() + (long) i + 1);
            } else {
                continue;
            }
            changed = true;
            break;
        }
    }
    return out;
}

std::vector<float> trim_silence(const std::vector<float> & wav, int sr) {
    if ((int) wav.size() < (int) (sr * 0.05)) return wav;
    const int hop = std::max(1, (int) (sr * 0.01));
    const int n_frames = (int) wav.size() / hop;
    const float thresh = std::pow(10.0f, -45.0f / 20.0f);
    int first = -1, last = -1;
    for (int f = 0; f < n_frames; ++f) {
        double e = 0.0;
        for (int i = 0; i < hop; ++i) e += (double) wav[(size_t) (f * hop + i)] * wav[(size_t) (f * hop + i)];
        if (std::sqrt(e / hop + 1e-12) > thresh) {
            if (first < 0) first = f;
            last = f;
        }
    }
    if (first < 0) return wav;
    const int keep = (int) (sr * 0.02);
    const size_t start = (size_t) std::max(0, first * hop - keep);
    const size_t end = std::min(wav.size(), (size_t) ((last + 1) * hop + keep));
    return std::vector<float>(wav.begin() + (long) start, wav.begin() + (long) end);
}

}  // namespace

bool kitten_has_expression_tag(const std::string & text) {
    const std::string s = lower(text);
    for (const char * t : kExpressionTags) {
        if (s.find(std::string("[") + t + "]") != std::string::npos) return true;
    }
    for (const char * t : kVocalEventTags) {
        if (s.find(std::string("<") + t + ">") != std::string::npos) return true;
    }
    return std::regex_search(text, std::regex(R"(\(\(\([^()\n]{1,80}\)\)\))"));
}

std::string kitten_prepare_text(const std::string & text) {
    std::string s = std::regex_replace(text, std::regex(R"(\s*\n+\s*)"), " ");
    s = std::regex_replace(s, std::regex(R"(([a-z])([.!?;:])([A-Z]))"), "$1$2 $3");
    // Ellipses are a pause, not spoken dots.
    s = replace_all(s, "\xE2\x80\xA6", ", ");
    s = std::regex_replace(s, std::regex(R"((?:\.\s+){2,}\.|\.\.\.)"), ", ");
    s = dashes_to_commas(s);
    return capitalize_and_terminate(s);
}

std::vector<std::string> kitten_split_for_synthesis(const std::string & text, int max_chars, int min_chars) {
    const std::string stripped = strip(text);
    if (stripped.empty()) return {};
    if (max_chars <= 0 || (int) stripped.size() <= max_chars) return {stripped};

    // Sentence boundaries: after . ! ? (plus closing quotes/brackets), at whitespace.
    std::vector<std::string> sentences;
    {
        const std::regex sent_end(R"(([.!?]["'\)\]]*|[.!?](?:\xE2\x80\x99|\xE2\x80\x9D)+)\s+)");
        size_t pos = 0;
        for (std::sregex_iterator it(stripped.begin(), stripped.end(), sent_end), end; it != end; ++it) {
            const size_t cut = (size_t) it->position(0) + (size_t) it->length(1);
            sentences.push_back(stripped.substr(pos, cut - pos));
            pos = (size_t) it->position(0) + (size_t) it->length(0);
        }
        sentences.push_back(stripped.substr(pos));
    }

    std::vector<std::string> out;
    std::string current;
    for (std::string sentence : sentences) {
        sentence = strip(sentence);
        if (sentence.empty()) continue;
        while ((int) sentence.size() > max_chars) {
            const size_t lim = (size_t) max_chars;
            long cut = -1;
            for (const char * sep : {", ", "; ", " \xE2\x80\x94 "}) {
                const size_t p = sentence.rfind(sep, lim);
                if (p != std::string::npos && p < lim) cut = std::max(cut, (long) p);
            }
            const size_t at = cut > max_chars / 3 ? (size_t) cut + 1 : lim;
            const std::string head = strip(sentence.substr(0, at));
            sentence = strip(sentence.substr(at));
            if (!current.empty()) { out.push_back(current); current.clear(); }
            out.push_back(head);
        }
        if (current.empty()) {
            current = sentence;
        } else if ((int) (current.size() + 1 + sentence.size()) <= max_chars) {
            current += " " + sentence;
        } else {
            out.push_back(current);
            current = sentence;
        }
    }
    if (!current.empty()) out.push_back(current);
    out = merge_short(out, max_chars, min_chars);
    for (auto & c : out) c = ensure_terminal_punct(c);
    return out;
}

std::vector<float> kitten_shape_chunk(const std::vector<float> & wav, int sample_rate) {
    return tts_shape_segment(wav, sample_rate, /*trim=*/true, /*fade_ms=*/8);
}

std::vector<float> kitten_join_chunks(const std::vector<std::vector<float>> & wavs, int sample_rate, float gap_s) {
    std::vector<const std::vector<float> *> parts;
    for (const auto & w : wavs) {
        if (!w.empty()) parts.push_back(&w);
    }
    if (parts.empty()) return {};
    if (parts.size() == 1) return *parts[0];
    const size_t gap = (size_t) (sample_rate * gap_s);
    const int fade = std::max(1, (int) (sample_rate * 0.008));
    std::vector<float> out;
    (void) fade;
    for (size_t i = 0; i < parts.size(); ++i) {
        const std::vector<float> w = kitten_shape_chunk(*parts[i], sample_rate);
        out.insert(out.end(), w.begin(), w.end());
        if (i + 1 != parts.size()) out.insert(out.end(), gap, 0.0f);
    }
    return out;
}

}  // namespace codec_common
