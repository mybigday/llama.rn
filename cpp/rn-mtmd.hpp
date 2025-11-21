#pragma once

#include "rn-llama.h"
#include "rn-common.hpp"
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/mtmd-helper.h"
#include "tools/mtmd/clip.h"
#include <string>
#include <vector>
#include <cstdint>

namespace rnllama {

// MTMD context structure
struct llama_rn_context_mtmd {
    mtmd_context *mtmd_ctx = nullptr;

    // State fields
    std::vector<std::string> bitmap_past_hashes;

    // Constructor - Initialize multimodal
    llama_rn_context_mtmd(
        const std::string &mmproj_path,
        bool use_gpu,
        llama_model *model,
        llama_context *ctx,
        const common_params &params,
        bool &has_multimodal,
        common_params &mutable_params
    );

    // Destructor - Release multimodal resources
    ~llama_rn_context_mtmd();

    // Process media
    void processMedia(
        llama_context *ctx,
        const std::string &prompt,
        const std::vector<std::string> &media_paths,
        int n_ctx,
        int n_batch,
        llama_pos &n_past,
        std::vector<llama_token> &embd,
        bool &context_full,
        common_sampler *ctx_sampling,
        std::vector<std::string> &bitmap_past_hashes,  // Per-slot bitmap hashes
        int32_t seq_id  // Sequence ID for parallel slots
    );

    // Check if multimodal is enabled
    bool isEnabled(bool has_multimodal) const;

    // Check if multimodal supports vision
    bool supportVision() const;

    // Check if multimodal supports audio
    bool supportAudio() const;
};

// FNV-1a hash function for bitmap hashing
inline std::string fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return std::to_string(hash);
}

// Base64 encoding/decoding utilities
static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

inline bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

using raw_buffer = std::vector<uint8_t>;

inline raw_buffer base64_decode(const std::string & encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    raw_buffer ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }

            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]      ) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

// MTMD tokenization result structure
struct mtmd_tokenize_result {
    std::vector<std::string> bitmap_hashes;
    std::vector<llama_token> tokens;
    std::vector<size_t> chunk_pos; // both text and media
    std::vector<size_t> chunk_pos_media; // media only
    mtmd_input_chunks* chunks = nullptr;
};

// Forward declaration for llama_rn_context
struct llama_rn_context;

// Tokenize text with media function
inline mtmd_tokenize_result tokenizeWithMedia(llama_rn_context_mtmd *mtmd_wrapper, const std::string &prompt, const std::vector<std::string> &media_paths) {
    mtmd_tokenize_result result;
    mtmd::bitmaps bitmaps;

    // Load all media paths
    for (const auto& media_path : media_paths) {
        LOG_INFO("[DEBUG] Loading media: %s",
                 media_path.substr(0, 50).c_str()); // Only log part of path for base64

        // Check if it's a base64 media
        if (media_path.compare(0, 11, "data:image/") == 0 || media_path.compare(0, 11, "data:audio/") == 0) {
            LOG_INFO("[DEBUG] Detected base64 encoded media");

            // Parse base64 data
            std::vector<std::string> parts;
            size_t comma_pos = media_path.find(',');
            if (comma_pos == std::string::npos) {
                throw std::runtime_error("Invalid base64 media format, missing comma separator");
            }

            std::string header = media_path.substr(0, comma_pos);
            std::string base64_data = media_path.substr(comma_pos + 1);

            if (header.find("base64") == std::string::npos) {
                bitmaps.entries.clear();
                throw std::runtime_error("Image must be base64 encoded");
            }

            // Decode base64
            raw_buffer media_data = base64_decode(base64_data);
            LOG_INFO("[DEBUG] Base64 decoded, size: %zu bytes", media_data.size());

            // Load bitmap from memory buffer using direct initialization
            mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(mtmd_wrapper->mtmd_ctx, media_data.data(), media_data.size()));
            if (!bmp.ptr) {
                bitmaps.entries.clear();
                throw std::runtime_error("Failed to load base64 media");
            }

            // Calculate bitmap hash (for KV caching)
            std::string hash = fnv_hash(bmp.data(), bmp.n_bytes());
            bmp.set_id(hash.c_str());
            LOG_INFO("[DEBUG] Bitmap hash: %s", hash.c_str());
            bitmaps.entries.push_back(std::move(bmp));
            result.bitmap_hashes.push_back(hash.c_str());
        } else if (media_path.compare(0, 7, "http://") == 0 || media_path.compare(0, 8, "https://") == 0) {
            // HTTP URLs are not supported yet
            LOG_ERROR("[DEBUG] HTTP/HTTPS URLs are not supported yet: %s", media_path.c_str());
            throw std::runtime_error("HTTP/HTTPS URLs are not supported yet");
        } else {
            // Regular file path
            LOG_INFO("[DEBUG] Loading media from file");

            // Check if file exists
            FILE* file = fopen(media_path.c_str(), "rb");
            if (file == nullptr) {
                bitmaps.entries.clear();
                throw std::runtime_error("File does not exist or cannot be opened");
            }

            // Get file size
            fseek(file, 0, SEEK_END);
            long file_size = ftell(file);
            fseek(file, 0, SEEK_SET);
            LOG_INFO("[DEBUG] File exists and size is %ld bytes", file_size);
            fclose(file);

            // Create bitmap directly
            mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(mtmd_wrapper->mtmd_ctx, media_path.c_str()));
            if (!bmp.ptr) {
                bitmaps.entries.clear();
                throw std::runtime_error("Failed to load media");
            }

            // Calculate bitmap hash (for KV caching)
            std::string hash = fnv_hash(bmp.data(), bmp.nx()*bmp.ny()*3);
            bmp.set_id(hash.c_str());
            LOG_INFO("[DEBUG] Bitmap hash: %s", hash.c_str());
            bitmaps.entries.push_back(std::move(bmp));
            result.bitmap_hashes.push_back(hash.c_str());
        }
    }

    // Create input chunks
    LOG_INFO("[DEBUG] Initializing input chunks");
    result.chunks = mtmd_input_chunks_init();
    if (result.chunks == nullptr) {
        bitmaps.entries.clear();
        throw std::runtime_error("Failed to initialize input chunks");
    }

    mtmd_input_text input_text;
    input_text.text = prompt.c_str(); // Use the full prompt with image marker
    input_text.add_special = true;  // Add BOS token if this is the first message
    input_text.parse_special = true;       // Parse special tokens like <__media__>

    /**
     * Tokenize the text and media together.
     *
     * Example of tokenization for "foo bar <__media__> baz <__media__>":
     *
     * 1. Input text with media markers:
     *
     *    "foo bar <__media__> baz <__media__>"
     *
     * 2. Model-specific markers are added.
     *
     * 3. Text is split and tokenized into chunks:
     *
     *    ┌─────────────┐  ┌─────────────────────────┐  ┌─────────┐  ┌─────────────────────────┐
     *    │ TEXT CHUNK  │  │ IMAGE CHUNK             │  │ TEXT    │  │ IMAGE CHUNK             │
     *    │ "foo bar "  │  │                         │  │ " baz " │  │                         │
     *    └─────────────┘  └─────────────────────────┘  └─────────┘  └─────────────────────────┘
     *          │                     │                      │                    │
     *          ▼                     ▼                      ▼                    ▼
     *    ┌─────────────┐  ┌─────────────────────────┐  ┌─────────┐  ┌─────────────────────────┐
     *    │ [1234,5678] │  │ Image Data Structure    │  │ [9012]  │  │ Image Data Structure    │
     *    └─────────────┘  └─────────────────────────┘  └─────────┘  └─────────────────────────┘
     *
     * 4. Image token structure differences:
     *
     *    For Qwen2VL (uses M-RoPE with 2D positions):
     *    ┌─────────────────────────────────────────┐
     *    │ MEDIA_CHUNK                             │
     *    │ ┌───────────────────────────────────┐   │
     *    │ │ mtmd_image_tokens:                │   │
     *    │ │  nx = 16, ny = 16                 │   │ ← 2D grid (16×16 = 256 tokens)
     *    │ │  use_mrope_pos = true             │   │ ← Uses M-RoPE positioning
     *    │ │  batch_f32 = [image_embeddings]   │   │
     *    │ └───────────────────────────────────┘   │
     *    └─────────────────────────────────────────┘
     *
     *    For other models (uses 1D positions):
     *    ┌─────────────────────────────────────────┐
     *    │ MEDIA_CHUNK                             │
     *    │ ┌───────────────────────────────────┐   │
     *    │ │ mtmd_image_tokens:                │   │
     *    │ │  nx = 256, ny = 1                 │   │ ← 1D sequence (256 tokens)
     *    │ │  use_mrope_pos = false            │   │ ← Uses standard positioning
     *    │ │  batch_f32 = [image_embeddings]   │   │
     *    │ └───────────────────────────────────┘   │
     *    └─────────────────────────────────────────┘
     *
     * 5. Final chunks array:
     *    chunks[0] = TEXT_CHUNK([1234, 5678])
     *    chunks[1] = MEDIA_CHUNK(first_image)
     *    chunks[2] = TEXT_CHUNK([9012])
     *    chunks[3] = MEDIA_CHUNK(second_image)
     */
    LOG_INFO("[DEBUG] Tokenizing text and %zu media", bitmaps.entries.size());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(mtmd_wrapper->mtmd_ctx, result.chunks, &input_text, bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    if (res != 0) {
        mtmd_input_chunks_free(result.chunks);
        bitmaps.entries.clear();
        throw std::runtime_error("Failed to tokenize text and media");
    }

    // Log chunk information
    size_t num_chunks = mtmd_input_chunks_size(result.chunks);
    LOG_INFO("[DEBUG] Tokenization successful: num_chunks=%zu", num_chunks);

    // Track the total number of tokens (both text and image)
    size_t total_token_count = 0;

    /**
     * Evaluate the chunks.
     *
     * For our example "foo bar <__media__> baz <__media__>":
     *
     * Token organization in memory:
     *
     *    all_tokens: [t0][t1][NULL][NULL]...[NULL][t2][NULL][NULL]...[NULL]
     *    positions:   0   1    2    3   ...  257   258  259  260 ...  514
     *    chunk_pos:   0        2                   258  259
     *
     *    Where:
     *    - [t0][t1] are text tokens for "foo bar " (positions 0-1)
     *    - [NULL]x256 are placeholder tokens for the first image (positions 2-257)
     *    - [t2] is the text token for " baz " (position 258)
     *    - [NULL]x256 are placeholder tokens for the second image (positions 259-514)
     */
    for (size_t i = 0; i < num_chunks; i++) {
        result.chunk_pos.push_back(total_token_count);

        const mtmd_input_chunk* chunk = mtmd_input_chunks_get(result.chunks, i);
        mtmd_input_chunk_type chunk_type = mtmd_input_chunk_get_type(chunk);

        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens;
            const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            LOG_INFO("[DEBUG] Chunk %zu: type=TEXT, n_tokens=%zu", i, n_tokens);

            // Add text tokens
            result.tokens.insert(result.tokens.end(), tokens, tokens + n_tokens);
            total_token_count += n_tokens;
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            result.chunk_pos_media.push_back(total_token_count);

            size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
            size_t n_pos = mtmd_input_chunk_get_n_pos(chunk);
            LOG_INFO("[DEBUG] Chunk %zu: type=%s, n_tokens=%zu, n_pos=%zu",
                     i, chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "IMAGE" : "AUDIO", n_tokens, n_pos);

            for (size_t j = 0; j < n_pos; j++) {
                result.tokens.push_back(LLAMA_TOKEN_NULL); // Placeholder token
            }
            total_token_count += n_pos;
        }
    }

    bitmaps.entries.clear();

    return result;
}

inline void llama_rn_context_mtmd::processMedia(
    llama_context *ctx,
    const std::string &prompt,
    const std::vector<std::string> &media_paths,
    int n_ctx,
    int n_batch,
    llama_pos &n_past,
    std::vector<llama_token> &embd,
    bool &context_full,
    common_sampler *ctx_sampling,
    std::vector<std::string> &bitmap_past_hashes_ref,  // Per-slot bitmap hashes
    int32_t seq_id  // Sequence ID for parallel slots
) {
    // Multimodal path
    std::string full_prompt = prompt;
    auto default_media_marker = mtmd_default_marker();
    // Add media marker if it doesn't already exist
    if (full_prompt.find(default_media_marker) == std::string::npos) {
        full_prompt += " ";
        full_prompt += default_media_marker;
    }

    LOG_INFO("[DEBUG] Processing message with role=user, content=%s", full_prompt.c_str());
    LOG_INFO("[DEBUG] Processing %zu media with prompt: %s", media_paths.size(), prompt.c_str());
    LOG_INFO("[DEBUG] Current context state: n_past=%d, n_ctx=%d", n_past, n_ctx);

    auto result = tokenizeWithMedia(this, full_prompt, media_paths);

    auto all_tokens = result.tokens;
    auto chunks = result.chunks;
    auto chunk_pos = result.chunk_pos;
    auto chunk_pos_media = result.chunk_pos_media;
    auto bitmap_hashes = result.bitmap_hashes;

    // Check if we have enough context space for all tokens
    if (all_tokens.size() >= (size_t)n_ctx) {
        mtmd_input_chunks_free(chunks);
        context_full = true;
        throw std::runtime_error("Not enough context space");
    }

    n_past = find_common_prefix_length(embd, all_tokens);

    llama_pos new_n_past = n_past;

    // Adjust n_past to position of the text chunk
    // TODO: Edit the text chunk to remove the tokens before n_past to speed up
    // need to update the mtmd api
    auto adjusted_n_past = -1;
    for (size_t i = 0; i < chunk_pos.size(); i++) {
        if (n_past < chunk_pos[i]) {
            break;
        }
        bool is_end = i + 1 == chunk_pos.size();
        if (
            chunk_pos[i] < n_past &&
            (!is_end && chunk_pos[i + 1] > n_past)
            // is_end & n_past < total_token_count:
            // don't need to adjust and it will skip eval_chunk_single, let nextToken() to finish the job
        ) {
            adjusted_n_past = chunk_pos[i];
        }
    }
    if (adjusted_n_past != -1) {
        n_past = adjusted_n_past;
        new_n_past = n_past;
        LOG_INFO("[DEBUG] Adjusted n_past to %d", n_past);
    }

    // Compare bitmap hashes, if they are not the same, backtrack n_past to the position of the first mismatch
    if (bitmap_past_hashes_ref.size() > 0) {
        for (size_t i = 0; i < bitmap_hashes.size(); i++) {
            auto pos = chunk_pos_media[i];
            if (n_past < pos) {
                break;
            }
            if (i >= bitmap_past_hashes_ref.size()) {
                break;
            }
            if (bitmap_hashes[i] != bitmap_past_hashes_ref[i]) {
                LOG_INFO(
                    "[DEBUG] Bitmap hash mismatch at position %zu, %s != %s",
                    i, bitmap_hashes[i].c_str(), bitmap_past_hashes_ref[i].c_str()
                );
                n_past = chunk_pos_media[i];
                new_n_past = n_past;
                break;
            }
        }
    }

    // Clear all KV cache entries after position n_past for this slot's sequence
    auto * kv = llama_get_memory(ctx);

    bool clear_result = llama_memory_seq_rm(kv, seq_id, n_past, -1);
    if (!clear_result) {
        LOG_ERROR("[DEBUG] llama_memory_seq_rm failed (likely using a non-Transformer model)! Trying full clear...");
        llama_memory_clear(kv, false);
        n_past = 0;
        new_n_past = n_past;
    }


    LOG_INFO("[DEBUG] Evaluating chunks: n_past=%d, n_batch=%d", n_past, n_batch);

    size_t num_chunks = mtmd_input_chunks_size(chunks);

    for (size_t i = 0; i < chunk_pos.size(); i++) {

        LOG_INFO("[DEBUG] Evaluating chunk %zu: n_past=%d, chunk_pos=%zu", i, n_past, chunk_pos[i]);

        // Process chunk only if it's after the current n_past
        if (chunk_pos[i] >= n_past) {
            bool chunk_logits_last = (i == num_chunks - 1);
            auto chunk = mtmd_input_chunks_get(chunks, i);

            int32_t res = mtmd_helper_eval_chunk_single(
                this->mtmd_ctx,
                ctx,
                chunk,
                n_past,
                seq_id,
                n_batch,
                chunk_logits_last,
                &new_n_past
            );
            if (res != 0) {
                mtmd_input_chunks_free(chunks);
                throw std::runtime_error("Failed to evaluate chunks");
            }
            n_past = new_n_past;
        }
    }

    if (n_past == all_tokens.size() && n_past > 0 && all_tokens[n_past - 1] != LLAMA_TOKEN_NULL) {
        // we have to evaluate at least 1 token to generate logits.
        n_past--;
    }

    // Update embd with all tokens (both text and media)
    embd = all_tokens;

    bitmap_past_hashes_ref = bitmap_hashes;

    // Update sampling context with text tokens only
    for (auto & token : all_tokens) {
        if (token == LLAMA_TOKEN_NULL) {
            continue;
        }
        common_sampler_accept(ctx_sampling, token, false);
    }

    // Clean up media resources
    LOG_INFO("[DEBUG] Cleaning up resources");
    mtmd_input_chunks_free(chunks);
}

inline llama_rn_context_mtmd::llama_rn_context_mtmd(
    const std::string &mmproj_path,
    bool use_gpu,
    llama_model *model,
    llama_context *ctx,
    const common_params &params,
    bool &has_multimodal,
    common_params &mutable_params
) {
    LOG_INFO("[DEBUG] Initializing multimodal with mmproj path: %s", mmproj_path.c_str());

    if (model == nullptr) {
        LOG_ERROR("[DEBUG] Model not loaded, cannot initialize multimodal", "");
        throw std::runtime_error("Model not loaded, cannot initialize multimodal");
    }

    LOG_INFO("[DEBUG] Model info: n_ctx=%d, n_embd=%d",
             llama_n_ctx(ctx),
             llama_model_n_embd(model));

    // Initialize mtmd context
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = use_gpu;
    mtmd_params.print_timings = false;
    mtmd_params.n_threads = params.cpuparams.n_threads;

    LOG_INFO("[DEBUG] Initializing mtmd context with threads=%d", mtmd_params.n_threads);

    auto mtmd_ctx = mtmd_init_from_file(mmproj_path.c_str(), model, mtmd_params);
    if (mtmd_ctx == nullptr) {
        LOG_ERROR("[DEBUG] Failed to initialize multimodal context with mmproj: %s", mmproj_path.c_str());
        throw std::runtime_error("Failed to initialize multimodal context");
    }
    this->mtmd_ctx = mtmd_ctx;

    has_multimodal = true;

    // Check if the model uses M-RoPE or non-causal attention
    bool uses_mrope = mtmd_decode_use_mrope(mtmd_ctx);
    bool uses_non_causal = mtmd_decode_use_non_causal(mtmd_ctx);
    LOG_INFO("[DEBUG] Model multimodal properties: uses_mrope=%d, uses_non_causal=%d",
             uses_mrope ? 1 : 0,
             uses_non_causal ? 1 : 0);

    // Disable context shifting when multimodal is enabled
    // This is because an media chunk may contain multiple tokens
    // and context shifting could break the media representation
    mutable_params.ctx_shift = false;

    // params.n_cache_reuse = 0;

    LOG_INFO("Multimodal context initialized successfully with mmproj: %s", mmproj_path.c_str());
    LOG_INFO("Context shifting disabled for multimodal support");
}

inline llama_rn_context_mtmd::~llama_rn_context_mtmd() {
    if (mtmd_ctx != nullptr) {
        mtmd_free(mtmd_ctx);
        mtmd_ctx = nullptr;
    }
}

inline bool llama_rn_context_mtmd::isEnabled(bool has_multimodal) const {
    return has_multimodal && mtmd_ctx != nullptr;
}

inline bool llama_rn_context_mtmd::supportVision() const {
    return mtmd_ctx != nullptr && mtmd_support_vision(mtmd_ctx);
}

inline bool llama_rn_context_mtmd::supportAudio() const {
    return mtmd_ctx != nullptr && mtmd_support_audio(mtmd_ctx);
}

} // namespace rnllama
