#ifndef CODEC_COMMON_KITTEN_TEXT_H
#define CODEC_COMMON_KITTEN_TEXT_H

// KittenTTS 2 text handling, ported from kittenml/kittentts2/text.py.
//
// Covers the `normalize=False` path: expression-tag detection, line/dash
// cleanup, capitalise-and-terminate, sentence chunking and chunk joining. The
// written→spoken normalizer (`kitten_text_processing`, an English WFST
// grammar) is NOT ported: pass already-spoken text (numbers spelled out).

#include <string>
#include <vector>

namespace codec_common {

// True when the text uses a known `[emotion]`, `<event>` or `(((emphasis)))`
// control — the model then receives the `{emo: 1}` conditioning prefix.
bool kitten_has_expression_tag(const std::string & text);

// Light cleanup the reference applies even with normalize=False.
std::string kitten_prepare_text(const std::string & text);

// Sentence-aware greedy packing (split_for_synthesis). A text that fits in
// `max_chars` comes back as a single chunk.
std::vector<std::string> kitten_split_for_synthesis(const std::string & text, int max_chars, int min_chars);

// One chunk's part of join_chunks: edge silence trimmed, 8 ms fades.
std::vector<float> kitten_shape_chunk(const std::vector<float> & wav, int sample_rate);

// Trim each chunk's edge silence, fade 8 ms at the seams and insert `gap_s`
// of silence between chunks (join_chunks).
std::vector<float> kitten_join_chunks(const std::vector<std::vector<float>> & wavs, int sample_rate, float gap_s);

}  // namespace codec_common

#endif
