#ifndef RN_COMMON_HPP
#define RN_COMMON_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <cinttypes>
#include "llama.h"

namespace rnllama {

// Helper function to check if string ends with suffix
static bool ends_with(const std::string& str, const std::string& suffix) {
  if (suffix.size() > str.size()) return false;
  return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

// Helper function to find partial stop string at end of text
static size_t find_partial_stop_string(const std::string& stop, const std::string& text) {
  if (!text.empty() && !stop.empty()) {
      const char text_last_char = text.back();
      for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
          if (stop[char_index] == text_last_char) {
              const std::string current_partial = stop.substr(0, char_index + 1);
              if (ends_with(text, current_partial)) {
                  return text.size() - char_index - 1;
              }
          }
      }
  }
  return std::string::npos;
}

// Helper function to format rerank task: [BOS]query[EOS][SEP]doc[EOS]
static std::vector<llama_token> format_rerank_tokens(
  const llama_vocab* vocab,
  const std::vector<llama_token>& query_tokens,
  const std::vector<llama_token>& doc_tokens
) {
  std::vector<llama_token> result;

  llama_token eos_token = llama_vocab_eos(vocab);
  if (eos_token == LLAMA_TOKEN_NULL) {
      eos_token = llama_vocab_sep(vocab);
  }

  result.reserve(doc_tokens.size() + query_tokens.size() + 4);

  if (llama_vocab_get_add_bos(vocab)) {
      result.push_back(llama_vocab_bos(vocab));
  }

  result.insert(result.end(), query_tokens.begin(), query_tokens.end());

  if (llama_vocab_get_add_eos(vocab)) {
      result.push_back(eos_token);
  }

  if (llama_vocab_get_add_sep(vocab)) {
      result.push_back(llama_vocab_sep(vocab));
  }

  result.insert(result.end(), doc_tokens.begin(), doc_tokens.end());

  if (llama_vocab_get_add_eos(vocab)) {
      result.push_back(eos_token);
  }

  return result;
}

}

#endif
