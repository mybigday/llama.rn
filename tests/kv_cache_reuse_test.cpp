// KV-cache-reuse reproduction / regression harness.
//
// Drives multi-turn chat directly through the rn-completion API (the "legacy"
// non-slot completion path that PocketPal and friends use) and measures, for
// every turn, how many prompt tokens were reused from the KV cache vs. how many
// had to be re-decoded.
//
// The point it makes:
//   - On pure-attention models (smollm2) the shared prefix is reused for free:
//     llama_memory_seq_rm() drops the diverged suffix and keeps the prefix.
//   - On recurrent / hybrid models (mamba, lfm2, granite4, qwen35) the in-place
//     removal fails, so the current code wipes the whole cache and reprocesses
//     the entire growing prompt every turn -> reuse == 0. That is the bug.
//
// The assertions below encode the DESIRED (post-fix) behaviour, so:
//   - before the checkpoint fix: recurrent/hybrid rows FAIL  -> issue reproduced
//   - after  the checkpoint fix: every row PASSES            -> issue fixed
//
// A separate invariant that must hold in BOTH states: starting a "new chat
// session" without an explicit cache clear must not leak facts from the
// previous session into the new answer.
//
// Usage:
//   ./kv_cache_reuse_test [model_key ...]     # default: every model found in ../models
//   MODELS_DIR=/path ./kv_cache_reuse_test    # override model directory

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

#include "rn-llama.h"
#include "rn-completion.h"
#include "rn-mtmd.hpp"
#include "common.h"
#include "nlohmann/json.hpp"
#include "tools/mtmd/mtmd.h"

using namespace rnllama;
using json = nlohmann::ordered_json;

namespace {

// ------------------------------------------------------------------ utilities

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

bool contains_ci(const std::string &haystack, const std::string &needle) {
    return to_lower(haystack).find(to_lower(needle)) != std::string::npos;
}

// Match `stem` only at a word start (case-insensitive), so "ali" matches
// "Ali"/"Ali's" but not "finalize", and "cat" matches "cat"/"cats" but not
// "category"/"communicate". Prefix stems ("cycl") match "cycling"/"cycle".
bool has_word(const std::string &text, const std::string &stem) {
    const std::string t = to_lower(text), s = to_lower(stem);
    for (size_t pos = t.find(s); pos != std::string::npos; pos = t.find(s, pos + 1)) {
        const bool left_boundary = (pos == 0) || !std::isalpha((unsigned char) t[pos - 1]);
        if (left_boundary) return true;
    }
    return false;
}

bool has_any_word(const std::string &text, const std::vector<std::string> &stems) {
    for (const auto &s : stems) if (has_word(text, s)) return true;
    return false;
}

// A recorded semantic check: what we asked, what came back, and the verdict.
struct SemCheck { std::string name; bool pass; std::string reply; };

struct TurnMetric {
    std::string label;
    size_t prompt_tokens = 0;   // total tokens in this turn's prompt
    size_t reused        = 0;   // n_past after loadPrompt (tokens NOT re-decoded)
    size_t reusable      = 0;   // tokens shared with the previous turn's prompt (the ceiling)
    std::string reply;          // generated assistant text
};

// A single conversation running against one loaded model. Wraps the exact call
// sequence a host app performs per turn: render chat -> loadPrompt -> generate.
struct ChatSim {
    llama_rn_context ctx;
    std::string system_prompt;
    std::vector<std::pair<std::string, std::string>> history; // (role, content)
    std::vector<llama_token> prev_embd;                        // prev turn's cache contents (prompt+gen)
    bool enable_thinking = true;
    bool use_mtp = false;                                      // enable native MTP speculative decoding
    std::vector<std::string> media_paths;                      // images/audio for the current turn (multimodal)
    int state_cache_budget_mb = 160;                           // <0 keeps the default; 0 disables the cache
    int state_cache_max_checkpoints = -1;                      // <0 keeps the default; 0 = no count cap
    bool ctx_shift = false;                                    // enable context shifting (overflow tests)

    // Tokenize exactly the way loadPrompt does, so reuse counts are comparable.
    std::vector<llama_token> tokenize_like_loadprompt(const std::string &prompt) const {
        const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(ctx.model));
        const bool is_enc_dec = llama_model_has_encoder(ctx.model);
        return ::common_tokenize(ctx.ctx, prompt, add_bos || is_enc_dec, true);
    }

    bool load(const std::string &model_path, int n_ctx) {
        common_params params;
        params.model.path = model_path;
        params.n_ctx = n_ctx;
        params.n_batch = 512;
        params.n_ubatch = 512;
        params.cpuparams.n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
        // Offload to GPU/NPU on-device via RNLLAMA_NGL (default 0 = CPU). Set e.g.
        // RNLLAMA_NGL=99 to push all layers to the OpenCL (Adreno) backend.
        const char *ngl_env = std::getenv("RNLLAMA_NGL");
        params.n_gpu_layers = ngl_env ? std::atoi(ngl_env) : 0;
        params.no_kv_offload = params.n_gpu_layers == 0;
        params.n_predict = -1;
        params.ctx_shift = ctx_shift;
        // Deterministic greedy sampling so leak checks and reuse counts are stable.
        params.sampling.temp = 0.0f;
        params.sampling.top_k = 1;

        if (use_mtp) {
            // Native multi-token-prediction speculative decoding. qwen35-style
            // models carry the MTP head in the target gguf; gemma4-style models
            // need a separate assistant gguf — auto-load "<model>.assistant.gguf"
            // when present (mem-shared draft, ctx_other = target).
            params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            params.speculative.draft.n_max = 3;
            const std::string suffix = ".gguf";
            if (model_path.size() > suffix.size()) {
                std::string assistant = model_path.substr(0, model_path.size() - suffix.size())
                                        + ".assistant.gguf";
                if (std::filesystem::exists(assistant)) {
                    params.speculative.draft.mparams.path = assistant;
                }
            }
        }

        // Host-configurable prompt-state-cache budget (0 disables it).
        if (state_cache_budget_mb >= 0) {
            ctx.state_cache_budget_bytes = (size_t) state_cache_budget_mb * 1024 * 1024;
        }
        if (state_cache_max_checkpoints >= 0) {
            ctx.state_cache_max_checkpoints = state_cache_max_checkpoints;
        }

        if (!ctx.loadModel(params)) return false;
        if (ctx.completion == nullptr) {
            ctx.completion = new llama_rn_context_completion(&ctx);
        }
        return true;
    }

    // Render the current (system + history) plus an optional pending user turn.
    std::string render(bool add_generation_prompt) const {
        json msgs = json::array();
        if (!system_prompt.empty()) {
            msgs.push_back({{"role", "system"}, {"content", system_prompt}});
        }
        for (const auto &m : history) {
            msgs.push_back({{"role", m.first}, {"content", m.second}});
        }
        // Replies are accumulated from raw nextToken() pieces, so on a backend that
        // emits degenerate output (e.g. the Adreno-740 mis-computing qwen35 -> a
        // reply ending mid-codepoint) the history can hold invalid UTF-8. Dump with
        // the replacing error handler so the harness never terminates in its own
        // serialization; a broken GPU should yield a failed answer, not a crash.
        common_chat_params cp = ctx.getFormattedChatWithJinja(
            msgs.dump(-1, ' ', false, json::error_handler_t::replace),
            /*chat_template*/ "", /*json_schema*/ "", /*tools*/ "",
            /*parallel_tool_calls*/ false, /*tool_choice*/ "",
            enable_thinking, /*reasoning_format*/ "none",
            add_generation_prompt, /*now*/ "", /*kwargs*/ {}, /*force_pure_content*/ false);
        return cp.prompt;
    }

    // Run one assistant turn for whatever is currently in `history` (whose last
    // entry must be a user message). Returns reuse metrics + generated reply.
    TurnMetric assistant_turn(const std::string &label, int max_new, bool force_clear = false) {
        TurnMetric tm;
        tm.label = label;

        const std::string prompt = render(/*add_generation_prompt*/ true);
        ctx.params.prompt = prompt;
        ctx.params.n_predict = max_new;

        auto *cmpl = ctx.completion;

        // Simulate a host that wipes the cache every turn (the pre-fix behaviour /
        // the "no reuse" baseline). Used to prove reuse yields identical output.
        if (force_clear) {
            ctx.clearCache(false);
            prev_embd.clear();
        }

        if (!media_paths.empty()) {
            // Multimodal path: prompt tokens include media placeholders, so the
            // text-only reuse ceiling doesn't apply; just drive generation and
            // report how many tokens were reused (n_past after processMedia).
            cmpl->rewind();
            cmpl->initSampling();
            cmpl->loadPrompt(media_paths);
            tm.reused = (size_t) cmpl->n_past;
            tm.prompt_tokens = cmpl->num_prompt_tokens;
            cmpl->beginCompletion();
            std::string reply;
            int n = 0;
            while (cmpl->has_next_token && n < max_new) {
                completion_token_output out = cmpl->nextToken();
                if (out.tok == -1) break;
                if (cmpl->stopped_eos || cmpl->is_interrupted) break;
                reply += out.text;
                n++;
                if (cmpl->stopped_limit) break;
            }
            cmpl->endCompletion();
            prev_embd = cmpl->embd;
            tm.reply = reply;
            history.push_back({"assistant", reply});
            return tm;
        }

        // The reuse ceiling is the shared prefix between this turn's prompt and
        // whatever the cache actually holds (the previous turn's prompt+generation).
        std::vector<llama_token> cur_prompt_tokens = tokenize_like_loadprompt(prompt);
        size_t shared = 0;
        while (shared < cur_prompt_tokens.size() && shared < prev_embd.size() &&
               cur_prompt_tokens[shared] == prev_embd[shared]) {
            shared++;
        }
        tm.reusable = shared;

        cmpl->rewind();
        cmpl->initSampling();
        cmpl->loadPrompt({});

        // Snapshot reuse immediately after loadPrompt, before generation advances n_past.
        tm.reused = (size_t) cmpl->n_past;
        tm.prompt_tokens = cmpl->num_prompt_tokens;

        cmpl->beginCompletion();
        std::string reply;
        int n = 0;
        while (cmpl->has_next_token && n < max_new) {
            completion_token_output out = cmpl->nextToken();
            if (out.tok == -1) break;
            if (cmpl->stopped_eos || cmpl->is_interrupted) break;
            reply += out.text;
            n++;
            if (cmpl->stopped_limit) break;
        }
        cmpl->endCompletion();

        // Remember what the cache now holds so the next turn can measure its ceiling.
        prev_embd = cmpl->embd;

        tm.reply = reply;
        history.push_back({"assistant", reply});
        return tm;
    }

    void user(const std::string &content) { history.push_back({"user", content}); }

    // Start a fresh session: drop the visible conversation but deliberately keep
    // the KV cache warm (this is what a host does to reuse the system prefix).
    void new_session(const std::string &new_system) {
        history.clear();
        system_prompt = new_system;
        // NOTE: prev_embd intentionally preserved so the reuse ceiling still
        // reflects what the cache actually holds from the previous session.
    }
};

// ------------------------------------------------------------------ reporting

struct Check {
    std::string name;
    bool pass;
    std::string detail;
    bool fix_target; // true => expected to fail before the checkpoint fix
};

void print_turn(const TurnMetric &tm) {
    std::cout << "    " << std::left << std::setw(22) << tm.label
              << " prompt=" << std::setw(5) << tm.prompt_tokens
              << " reusable=" << std::setw(5) << tm.reusable
              << " reused=" << std::setw(5) << tm.reused
              << " reprocessed=" << (tm.prompt_tokens - tm.reused)
              << "\n";
}

// ------------------------------------------------------------------ scenarios

// instruct: can hold a coherent chat (used for no-leak checks, robust on any size)
// strong:   large enough for reliable factual recall (positive-answer checks)
std::vector<Check> run_model(const std::string &key, const std::string &path,
                             bool instruct, bool strong) {
    std::vector<Check> checks;
    std::cout << "\n================ " << key << " (" << path << ") ================\n";

    ChatSim sim;
    if (!sim.load(path, /*n_ctx*/ 4096)) {
        checks.push_back({key + ": model load", false, "loadModel failed", false});
        return checks;
    }
    sim.enable_thinking = true;
    sim.system_prompt = "You are a helpful assistant.";

    {
        llama_model *model = sim.ctx.model;
        std::cout << "  [model] recurrent=" << llama_model_is_recurrent(model)
                  << " hybrid=" << llama_model_is_hybrid(model)
                  << " n_swa=" << llama_model_n_swa(model) << "\n";
    }

    // Empirical checkpoint-size probe: decode a chunk, then compare full vs
    // partial-only per-seq state size. Decides full-state vs PARTIAL_ONLY design.
    if (std::getenv("STATE_PROBE")) {
        sim.user("Write a few sentences about the sea, please.");
        (void) sim.assistant_turn("probe", /*max_new*/ 128);
        size_t full_sz = llama_state_seq_get_size_ext(sim.ctx.ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        size_t part_sz = llama_state_seq_get_size_ext(sim.ctx.ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        std::cout << "  [probe] after " << sim.prev_embd.size() << " tokens:"
                  << " full=" << (full_sz / 1024.0) << " KiB"
                  << " partial=" << (part_sz / 1024.0) << " KiB\n";
        sim.history.clear();
        sim.prev_embd.clear();
        sim.ctx.clearCache(false);
    }

    // ---- Scenario 1: append-only multi-turn (pure conversation) -------------
    std::cout << "  [append-only]\n";
    const char *user_msgs[] = {
        "Hi, my name is Jack and I love science fiction books.",
        "What is the capital of France?",
        "Can you name a famous river?",
        "Thanks, that was helpful.",
    };
    std::vector<TurnMetric> append_turns;
    for (int t = 0; t < 4; t++) {
        sim.user(user_msgs[t]);
        TurnMetric tm = sim.assistant_turn("turn" + std::to_string(t + 1), /*max_new*/ 24);
        print_turn(tm);
        append_turns.push_back(tm);
    }
    // From turn 2 on there is a large shared prefix; we expect to reuse most of it.
    for (size_t t = 1; t < append_turns.size(); t++) {
        const auto &tm = append_turns[t];
        bool pass = tm.reusable > 0 && tm.reused >= (size_t) (0.5 * tm.reusable);
        checks.push_back({key + ": append reuse " + tm.label, pass,
                          "reused " + std::to_string(tm.reused) + "/" +
                          std::to_string(tm.reusable) + " reusable",
                          /*fix_target*/ true});
    }

    // ---- Scenario 2: resend the exact same last message (regenerate) --------
    std::cout << "  [resend-last]\n";
    {
        sim.user("Please summarize our chat so far.");
        TurnMetric a = sim.assistant_turn("send", /*max_new*/ 24);
        print_turn(a);
        // Pop the assistant reply and resend the identical user message.
        sim.history.pop_back();
        TurnMetric b = sim.assistant_turn("resend", /*max_new*/ 24);
        print_turn(b);
        // The prompt is identical to the previous one, so regenerate restores the
        // frontier checkpoint (just before the last user message) and reprocesses
        // only that last message + scaffold in a single decode — reusing the whole
        // shared prefix. We assert the reuse fraction, not an exact token count:
        // the near-end margin snapshot (which reprocessed only ~8 scaffold tokens)
        // was removed because re-planting it on every warm turn cost a full extra
        // decode (the append-TTFT regression); reprocessing one message instead is
        // the same single decode. Rollback-0 SSMs restore one frontier back.
        const size_t reprocessed = b.prompt_tokens - b.reused;
        bool pass = b.reusable > 0 && b.reused >= (size_t) (0.5 * b.reusable);
        checks.push_back({key + ": resend-last reuses shared prefix", pass,
                          "reused " + std::to_string(b.reused) + "/" +
                          std::to_string(b.reusable) + " reusable (reprocessed " +
                          std::to_string(reprocessed) + ")",
                          /*fix_target*/ true});
        sim.history.pop_back(); // drop resend reply, keep history tidy
    }

    // ---- Scenario 3: edit a mid-conversation message ------------------------
    // Build a conversation, then edit an earlier user turn (resend the whole
    // conversation with that message changed) and continue. This is where cache
    // reuse pays off AND where a stale-KV bug would leak: after replacing "my name
    // is Ali" with "my hobby is cycling", the model must (a) reuse the shared
    // prefix, (b) recall the NEW fact, and (c) NOT surface the edited-out name.
    std::cout << "  [edit mid-conversation]\n";
    {
        // Direct answers for the semantic checks below (append-only above already
        // covers the thinking / <think>-strip reuse path).
        sim.enable_thinking = false;
        sim.new_session("You are a helpful assistant.");
        sim.ctx.clearCache(false);
        sim.prev_embd.clear();
        sim.user("Hi!");
        sim.assistant_turn("t1", /*max_new*/ 24);
        sim.user("My name is Ali.");
        sim.assistant_turn("t2", /*max_new*/ 24);

        // Edit turn 2: drop its answer and the message, replace it, regenerate.
        sim.history.pop_back(); // assistant reply to "My name is Ali."
        sim.history.pop_back(); // "My name is Ali."
        sim.user("My hobby is cycling.");
        TurnMetric edited = sim.assistant_turn("edited", /*max_new*/ 24);
        print_turn(edited);
        // The shared prefix (system + "Hi!" + its answer) must be reused, i.e. NOT
        // a full wipe. How much is reused depends on where a message-boundary /
        // margin checkpoint lands relative to the edit point, so we require
        // substantial-but-not-full reuse.
        bool reuse_ok = edited.reusable > 0 && edited.reused >= (size_t) (0.3 * edited.reusable);
        checks.push_back({key + ": edit prefix reuse", reuse_ok,
                          "reused " + std::to_string(edited.reused) + "/" +
                          std::to_string(edited.reusable), /*fix_target*/ true});

        if (instruct) {
            sim.user("What is my name? If I never told you, say you don't know.");
            std::string r_name = sim.assistant_turn("q-name", /*max_new*/ 40).reply;
            std::cout << "    Q:what is my name?  A:\"" << r_name << "\"\n";
            // The edited branch never contained a name -> must not surface "Ali".
            checks.push_back({key + ": edit no stale-name leak", !has_word(r_name, "ali"),
                              "reply: '" + r_name + "'", /*fix_target*/ false});
        }
        if (strong) {
            sim.user("What hobby did I tell you I have? Answer with just the hobby.");
            std::string r_hobby = sim.assistant_turn("q-hobby", /*max_new*/ 40).reply;
            std::cout << "    Q:what hobby did I mention? A:\"" << r_hobby << "\"\n";
            // The new fact must be recalled through the reused cache.
            checks.push_back({key + ": edit recalls new fact", has_word(r_hobby, "cycl"),
                              "reply: '" + r_hobby + "'", /*fix_target*/ false});
        }
    }

    // ---- Scenario 5: reuse must be bit-for-bit equivalent to full recompute -
    // Runs the same conversation twice under greedy sampling: once with prompt
    // reuse, once wiping the cache every turn. If the checkpoint restore were
    // subtly wrong, the reused run would diverge. They must match token-for-token.
    std::cout << "  [correctness: reuse == recompute]\n";
    {
        const char *conv[] = {
            "My favorite color is teal. What is yours?",
            "Name three fruits.",
            "Which of those fruits is red?",
        };
        // Reset to a cold, empty conversation.
        sim.history.clear();
        sim.prev_embd.clear();
        sim.system_prompt = "You are a helpful assistant.";
        sim.ctx.clearCache(false);

        // For each turn we compare the reply produced from the WARM cache (reuse)
        // against the reply produced after wiping the cache (full recompute) for
        // the SAME history. The recompute reply is then frozen into the history so
        // both paths see identical inputs on the next turn (no cascade drift).
        //
        // Exact equality is NOT expected: reusing a prefix changes the batch shape
        // of the tail decode, reordering floating-point reductions, which can flip
        // a greedy argmax a few tokens in (this happens for plain-attention prefix
        // reuse too). A CORRUPTED restore, by contrast, garbles output from token
        // zero. So we require a substantial shared leading prefix, not a match.
        std::vector<size_t> prefixes;
        int n_faithful = 0;      // turns that agree for a clearly-non-flip prefix
        bool all_nonempty = true;
        for (int t = 0; t < 3; t++) {
            sim.user(conv[t]);
            std::string r_reuse = sim.assistant_turn("cc-reuse", /*max_new*/ 20, false).reply;
            sim.history.pop_back(); // drop the reuse reply
            std::string r_recompute = sim.assistant_turn("cc-recompute", /*max_new*/ 20, true).reply;
            // history now ends with the recompute reply -> identical for both next turn

            auto shared_prefix = [](const std::string &a, const std::string &b) {
                size_t p = 0;
                while (p < a.size() && p < b.size() && a[p] == b[p]) p++;
                return p;
            };
            size_t p = shared_prefix(r_reuse, r_recompute);
            if (p < 12) {
                // Divergence is only evidence against reuse if recompute is
                // self-consistent. Multi-threaded FP reductions are not bitwise
                // deterministic run-to-run, and near-tied prompts (gemma4-E2B)
                // flip their first token even between two identical recomputes.
                // Re-run the recompute to measure that noise floor.
                sim.history.pop_back();
                std::string r_recompute2 = sim.assistant_turn("cc-recompute2", /*max_new*/ 20, true).reply;
                const size_t p_noise = shared_prefix(r_recompute, r_recompute2);
                std::cout << "    [cc t" << (t + 1) << "] diverged at char " << p
                          << " (recompute self-agreement: " << p_noise << ")"
                          << "\n      reuse:     \"" << r_reuse.substr(0, 60)
                          << "\"\n      recompute: \"" << r_recompute.substr(0, 60) << "\"\n";
                if (p_noise < 12) {
                    // recompute doesn't even reproduce itself here -> the flip is
                    // FP noise, not a reuse defect. Count as faithful.
                    p = p_noise > p ? p_noise : p;
                    n_faithful++;
                }
                // history now ends with recompute2 -> still a pure-recompute reply
            } else {
                n_faithful++;
            }
            prefixes.push_back(p);
            if (r_reuse.empty()) all_nonempty = false;
        }
        // Corruption garbles EVERY turn from token zero. A healthy restore
        // reproduces recompute for many chars on most turns; an occasional turn
        // may still flip its first token when two completions are near-tied
        // (a benign float effect that pure-attention reuse shows too). Require
        // the majority of turns to reproduce faithfully.
        bool ok = all_nonempty && n_faithful >= 2;
        std::string detail = "shared prefixes [";
        for (size_t i = 0; i < prefixes.size(); i++) {
            detail += std::to_string(prefixes[i]) + (i + 1 < prefixes.size() ? "," : "");
        }
        detail += "] chars, " + std::to_string(n_faithful) + "/3 faithful";
        checks.push_back({key + ": reuse~=recompute (no corruption)", ok, detail,
                          /*fix_target*/ false});
    }

    // ---- Scenario 4: new session without a manual cache clear ----------------
    // Same spirit as the edit test but the divergence is at the very start: a
    // fresh chat reuses the shared system-prompt prefix from the previous session
    // but must NOT recall anything the previous session established.
    std::cout << "  [new session]\n";
    {
        // A realistic, long system prompt (a few hundred tokens, like a real
        // assistant persona/policy). This is the stable prefix a new session
        // shares; the point of the fix is to reuse it instead of reprocessing it.
        const std::string kSys =
            "You are Aria, a helpful, knowledgeable, and meticulous AI assistant. "
            "Your goal is to give the user accurate, well-structured, and genuinely "
            "useful answers. Always think carefully before responding. When a "
            "question is ambiguous, briefly state the interpretation you are using. "
            "Prefer concrete examples over vague generalities. When you show code, "
            "make sure it is correct, idiomatic, and complete enough to run. "
            "Never fabricate facts, citations, APIs, or statistics; if you do not "
            "know something, say so plainly instead of guessing. Be honest about "
            "uncertainty and about the limits of your knowledge. Keep a warm, "
            "professional, and respectful tone at all times, and never be "
            "condescending. Do not lecture the user or add unnecessary caveats. "
            "Respect the user's stated preferences about format and length. If the "
            "user asks for a short answer, be concise; if they ask for depth, be "
            "thorough and organized with headings and lists. Avoid filler phrases "
            "and repetition. When a task has multiple reasonable approaches, briefly "
            "compare the trade-offs and then recommend one. If a request is unsafe, "
            "explain why and offer a safer alternative when possible. Protect the "
            "user's privacy and never ask for information you do not need. Follow "
            "the conversation context and remember what the user has told you within "
            "the session. Above all, be truthful, be clear, and be useful.";
        sim.new_session(kSys);
        sim.ctx.clearCache(false);
        sim.prev_embd.clear();
        sim.user("Hi! My hobby is cycling.");
        sim.assistant_turn("s1t1", /*max_new*/ 24);
        sim.user("What hobby did I tell you I have? Answer with just the hobby.");
        TurnMetric recall = sim.assistant_turn("s1-recall", /*max_new*/ 40);
        print_turn(recall);
        if (strong) {
            std::cout << "    [session1] Q:what hobby did I mention? A:\"" << recall.reply << "\"\n";
            // Within the session the fact is present -> must be recalled (and this
            // turn reused the session-1 prefix).
            checks.push_back({key + ": in-session recall", has_word(recall.reply, "cycl"),
                              "reply: '" + recall.reply + "'", /*fix_target*/ false});
        }
        bool recall_reuse = recall.reusable > 0 && recall.reused >= (size_t) (0.5 * recall.reusable);
        checks.push_back({key + ": in-session reuse", recall_reuse,
                          "reused " + std::to_string(recall.reused) + "/" +
                          std::to_string(recall.reusable), /*fix_target*/ true});

        // New session (same system prompt), WITHOUT clearing the cache. The only
        // shared prefix with the warm cache is the system prompt, so this measures
        // cross-session system-prompt reuse specifically.
        sim.new_session(kSys);
        sim.user("What hobby did I tell you I have? If I never told you, say you don't know.");
        TurnMetric s2 = sim.assistant_turn("s2", /*max_new*/ 40);
        print_turn(s2);

        // GAP PROBE: the system prompt should be reused, not reprocessed, on a new
        // session that keeps the warm cache. Attention models get this via seq_rm;
        // recurrent/hybrid/SWA need a checkpoint AT the system-prompt boundary,
        // which the per-turn snapshots don't provide -> they reprocess it.
        size_t s2_reprocessed = s2.prompt_tokens > s2.reused ? s2.prompt_tokens - s2.reused : 0;
        std::cout << "    [session2] system-prefix reuse: reused " << s2.reused << "/"
                  << s2.reusable << "  (reprocessed " << s2_reprocessed << " of "
                  << s2.prompt_tokens << " prompt tokens)\n";
        bool sys_reused = s2.reusable > 0 && s2.reused >= (size_t) (0.5 * s2.reusable);
        checks.push_back({key + ": new-session system-prefix reuse", sys_reused,
                          "reused " + std::to_string(s2.reused) + "/" +
                          std::to_string(s2.reusable), /*fix_target*/ true});

        if (instruct) {
            std::cout << "    [session2] Q:what hobby did I mention? A:\"" << s2.reply << "\"\n";
            // The new session was never told the hobby -> "cycling" must not leak.
            checks.push_back({key + ": new-session no leak", !has_word(s2.reply, "cycl"),
                              "reply: '" + s2.reply + "'", /*fix_target*/ false});
        }
    }

    // ---- Canary: post-restore seq_rm(k) must stay infallible -----------------
    // recoverStateCheckpoint ignores the return of seq_rm(k) after a PARTIAL_ONLY
    // restore (infallible with the current vendored llama.cpp). If a future
    // bootstrap adds a failure path there, this trips before users do.
    {
        auto *cmpl = sim.ctx.completion;
        if (!cmpl->state_checkpoints.empty()) {
            const size_t k = cmpl->state_checkpoints.front().n_tokens();
            const bool restored = cmpl->restoreStateCheckpoint(0);
            const bool rm_ok = restored &&
                llama_memory_seq_rm(llama_get_memory(sim.ctx.ctx), 0, (llama_pos) k, -1);
            checks.push_back({key + ": canary post-restore seq_rm(k) (llama.cpp semantics changed?)",
                              restored && rm_ok,
                              "restored=" + std::to_string(restored) +
                              " seq_rm=" + std::to_string(rm_ok) + " k=" + std::to_string(k),
                              /*fix_target*/ false});
            sim.ctx.clearCache(false); // leave the context in a defined state
        }
    }

    return checks;
}

// MTP (multi-token prediction) has its own prompt-eval path that used to clear the
// cache and reprocess the whole prompt every turn. This drives append-only chat
// with MTP enabled and checks that the per-turn *reprocessed* token count stays
// bounded instead of growing with the conversation, plus a no-leak invariant.
std::vector<Check> run_model_mtp(const std::string &key, const std::string &path) {
    std::vector<Check> checks;
    std::cout << "\n========= " << key << " [MTP] (" << path << ") =========\n";

    ChatSim sim;
    sim.use_mtp = true;
    if (!sim.load(path, /*n_ctx*/ 4096)) {
        checks.push_back({key + " [MTP]: model load", false, "loadModel failed", false});
        return checks;
    }
    sim.enable_thinking = true;
    sim.system_prompt = "You are a helpful assistant.";

    const char *user_msgs[] = {
        "Hi, my name is Jack and I love science fiction books.",
        "What is the capital of France?",
        "Can you name a famous river?",
        "Thanks, that was helpful.",
    };
    std::vector<size_t> reprocessed;
    std::vector<double> accept_rate;
    try {
        for (int t = 0; t < 4; t++) {
            sim.user(user_msgs[t]);
            TurnMetric tm = sim.assistant_turn("turn" + std::to_string(t + 1), /*max_new*/ 24);
            size_t rp = sim.ctx.completion->mtp_prompt_reprocessed;
            reprocessed.push_back(rp);
            // Draft acceptance per turn: a collapse on reused-prefix turns would
            // mean the draft was starved by the reuse (invisible in output).
            const size_t drafted  = sim.ctx.completion->num_draft_tokens;
            const size_t accepted = sim.ctx.completion->num_draft_tokens_accepted;
            accept_rate.push_back(drafted > 0 ? (double) accepted / drafted : -1.0);
            std::cout << "    " << std::left << std::setw(8) << tm.label
                      << " prompt=" << std::setw(5) << tm.prompt_tokens
                      << " mtp_reprocessed=" << rp
                      << " draft_accept=" << accepted << "/" << drafted << "\n";
        }
    } catch (const std::exception &e) {
        checks.push_back({key + " [MTP]: runnable", false,
                          std::string("MTP not supported by this model: ") + e.what(), false});
        return checks;
    }
    // Mem-shared MTP drafts (gemma4/EAGLE3) deliberately DON'T reuse: the shared
    // KV window can't be safely restored (TAG_KV_CACHE_SHARE_CELLS), so we force
    // a full reprocess each turn to avoid a corrupt state. Assert that safe
    // fallback here; only non-mem-shared drafts (qwen35) get bounded reprocess.
    const bool mem_shared = sim.ctx.completion->mtp_draft_mem_shared;
    for (size_t t = 1; t < reprocessed.size(); t++) {
        const size_t np = sim.ctx.completion->num_prompt_tokens;
        if (mem_shared) {
            // Full reprocess is correct here (the guard). Just require no crash /
            // non-degenerate: it reprocessed roughly the whole prompt.
            bool pass = reprocessed[t] * 2 >= np;
            checks.push_back({key + " [MTP]: mem-shared full-reprocess turn" + std::to_string(t + 1), pass,
                              "reprocessed " + std::to_string(reprocessed[t]) + "/" + std::to_string(np) +
                              " (reuse safely disabled)", /*fix_target*/ false});
        } else {
            bool pass = reprocessed[t] * 2 < np;
            checks.push_back({key + " [MTP]: bounded reprocess turn" + std::to_string(t + 1), pass,
                              "reprocessed " + std::to_string(reprocessed[t]) + "/" + std::to_string(np) +
                              " prompt tokens", /*fix_target*/ true});
        }
    }
    // Canary: on reused-prefix turns acceptance must not collapse vs the cold
    // turn (the draft is never seeded with the reused prefix; measured benign).
    if (!accept_rate.empty() && accept_rate[0] > 0) {
        for (size_t t = 1; t < accept_rate.size(); t++) {
            if (accept_rate[t] < 0) continue; // nothing drafted this turn
            bool pass = accept_rate[t] >= 0.5 * accept_rate[0];
            checks.push_back({key + " [MTP]: draft acceptance turn" + std::to_string(t + 1), pass,
                              "rate " + std::to_string(accept_rate[t]).substr(0, 4) +
                              " vs cold " + std::to_string(accept_rate[0]).substr(0, 4),
                              /*fix_target*/ false});
        }
    }

    // Correctness: MTP reuse must reproduce MTP full-recompute (see the non-MTP
    // scenario for why we compare a shared prefix rather than exact equality).
    {
        const char *conv[] = {"My favorite color is teal. What is yours?",
                              "Name three fruits."};
        sim.history.clear();
        sim.prev_embd.clear();
        sim.system_prompt = "You are a helpful assistant.";
        sim.ctx.clearCache(false);
        int n_faithful = 0;
        for (int t = 0; t < 2; t++) {
            sim.user(conv[t]);
            std::string r_reuse = sim.assistant_turn("cc-reuse", /*max_new*/ 16, false).reply;
            sim.history.pop_back();
            std::string r_recompute = sim.assistant_turn("cc-recompute", /*max_new*/ 16, true).reply;
            size_t p = 0;
            while (p < r_reuse.size() && p < r_recompute.size() && r_reuse[p] == r_recompute[p]) p++;
            if (p >= 12) n_faithful++;
        }
        checks.push_back({key + " [MTP]: reuse~=recompute", n_faithful >= 1,
                          std::to_string(n_faithful) + "/2 turns reproduce recompute",
                          /*fix_target*/ false});
    }

    // New-session correctness + no-leak under MTP (see the non-MTP version).
    sim.enable_thinking = false; // direct answers for the semantic checks
    sim.new_session("You are a helpful assistant.");
    sim.ctx.clearCache(false);
    sim.prev_embd.clear();
    sim.user("Hi! My hobby is cycling.");
    (void) sim.assistant_turn("s1t1", /*max_new*/ 16);
    sim.user("What is my hobby?");
    std::string mtp_recall = sim.assistant_turn("s1-recall", /*max_new*/ 32).reply;
    std::cout << "    [MTP session1] Q:what is my hobby? A:\"" << mtp_recall << "\"\n";
    checks.push_back({key + " [MTP]: in-session recall", has_word(mtp_recall, "cycl"),
                      "reply: '" + mtp_recall + "'", /*fix_target*/ false});
    sim.new_session("You are a helpful assistant.");
    sim.user("What is my hobby?");
    std::string mtp_s2 = sim.assistant_turn("s2", /*max_new*/ 32).reply;
    std::cout << "    [MTP session2] Q:what is my hobby? A:\"" << mtp_s2 << "\"\n";
    checks.push_back({key + " [MTP]: new-session no leak", !has_word(mtp_s2, "cycl"),
                      "reply: '" + mtp_s2 + "'", /*fix_target*/ false});
    return checks;
}

// Mean NLL of a probe continuation given the context loadPrompt just built
// (defined below, near the text fidelity test).
double score_probe(ChatSim &sim, const std::string &probe_text);

// Distribution-level fidelity on the MULTIMODAL restore path (chunk-aligned
// recoverStateCheckpoint), which is more complex than text and untested at the
// NLL level. Score a text probe after (a) a media checkpoint-restore reuse of an
// image turn and (b) a cold re-encode of the same image+prompt; a corrupted
// image state shifts the distribution. Two cold re-encodes give the noise floor.
std::vector<Check> run_vision_fidelity(const std::string &key, const std::string &model_path,
                                       const std::string &mmproj_path, const std::string &img) {
    std::vector<Check> checks;
    std::cout << "\n===== " << key << " [vision-fidelity]: image restore must not shift the distribution =====\n";
    ChatSim sim;
    if (!sim.load(model_path, /*n_ctx*/ 4096)) {
        checks.push_back({key + " [vision-fidelity]: model load", false, "loadModel failed", false});
        return checks;
    }
    if (!sim.ctx.initMultimodal(mmproj_path, /*use_gpu*/ false)) {
        checks.push_back({key + " [vision-fidelity]: mmproj init", false, "initMultimodal failed", false});
        return checks;
    }
    sim.system_prompt = "You are a helpful assistant.";
    sim.enable_thinking = false;
    const std::string marker = mtmd_default_marker();
    const std::string probe = "The animal in the picture has fur and four legs, and it is looking toward the camera with a calm expression.";
    auto *cmpl = sim.ctx.completion;

    auto ingest = [&]() {
        sim.ctx.params.prompt = sim.render(/*add_generation_prompt*/ true);
        sim.ctx.params.n_predict = 0;
        cmpl->rewind(); cmpl->initSampling();
        cmpl->loadPrompt(sim.media_paths);
        return (size_t) sim.ctx.mtmd_wrapper->last_reused_n_past;
    };

    try {
        // Turn 1: image + question, generate a real reply into history so the
        // image + reply are committed to the cache with a checkpoint.
        sim.media_paths = { img };
        sim.user(marker + "\nDescribe the animal in this image.");
        // No generated reply: a greedy reply forks across runs on near-ties
        // (threadpool FP jitter), which changes the scored context and makes the
        // probe flaky. max_new=0 ingests the image+prompt deterministically.
        (void) sim.assistant_turn("vf-warm", /*max_new*/ 0);

        // Turn 2 (follow-up on the SAME image): this is the case that restores the
        // image prefix from a checkpoint instead of re-encoding it.
        sim.media_paths = { img };
        sim.user("What color is its fur? Answer in one short sentence.");

        // (a) Reuse: media checkpoint restore of the image prefix.
        const size_t reused = ingest();
        const double nll_reuse = score_probe(sim, probe);

        // (b) Cold re-encode of the identical history x2 (2nd = noise floor).
        sim.ctx.clearCache(false); sim.prev_embd.clear();
        (void) ingest();
        const double nll_cold1 = score_probe(sim, probe);
        sim.ctx.clearCache(false);
        (void) ingest();
        const double nll_cold2 = score_probe(sim, probe);

        const double floor = std::fabs(nll_cold1 - nll_cold2);
        // The inherent cross-turn-chunking reuse residual is roughly absolute
        // (~0.12 on hybrid-recurrent lfm2vl, ~0.05 on SWA gemma4), so a 0.15 floor
        // clears it while still failing a real KV corruption (~0.20, e.g. the
        // duplicated-last-token bug). 5*floor guards the FP noise; 2% of the NLL
        // keeps headroom on high-NLL models.
        const double tol = std::max({0.15, 5.0 * floor, 0.02 * std::fabs(nll_cold1)});
        const double d = std::fabs(nll_reuse - nll_cold1);
        char buf[220];
        snprintf(buf, sizeof(buf), "reuse %.4f vs cold %.4f (d=%.4f), floor %.4f, tol %.4f, reused_prefix %zu",
                 nll_reuse, nll_cold1, d, floor, tol, reused);
        std::cout << "    [vision-fidelity] " << buf << "\n";
        const bool valid = nll_reuse==nll_reuse && nll_cold1==nll_cold1 && nll_cold2==nll_cold2;
        checks.push_back({key + " [vision-fidelity]: probe scored", valid, buf, false});
        // Guard against a vacuous pass: the reuse path must actually have restored
        // the image prefix, not fallen back to a full re-encode.
        checks.push_back({key + " [vision-fidelity]: image prefix actually restored",
                          reused > 0, "reused_prefix " + std::to_string(reused), /*fix_target*/ true});
        if (valid && reused > 0) {
            // Deterministic (fixed turn-1 context): reuse-vs-cold d is the inherent
            // residual of keeping a prefix computed under turn-1's chunk layout vs
            // re-encoding it under turn-2's — roughly ABSOLUTE (~0.05 gemma4 SWA,
            // ~0.12 lfm2vl hybrid-recurrent), not proportional to the NLL. A real
            // KV corruption (the duplicated-last-token bug this probe now guards
            // against) is ~0.20, well clear of it. So assert against an absolute
            // floor between the two rather than the (too-tight-for-low-NLL) 2% band.
            checks.push_back({key + " [vision-fidelity]: image restore keeps the distribution",
                              d <= tol, buf, /*fix_target*/ false});
        }
    } catch (const std::exception &e) {
        checks.push_back({key + " [vision-fidelity]: runnable", false,
                          std::string("threw: ") + e.what(), false});
    }
    return checks;
}

// Vision chat carries an image (a few hundred tokens) in the history. Re-encoding
// it every turn (the full-clear fallback) is costly, so we check the image is
// reused across turns. And crucially we check the ANSWERS are correct: a dog image
// is called a dog, swapping to a cat is called a cat (the dog must not leak), and
// removing the image entirely must not surface either animal.
std::vector<Check> run_model_multimodal(const std::string &key,
                                        const std::string &model_path,
                                        const std::string &mmproj_path,
                                        const std::string &img_dog,
                                        const std::string &img_cat) {
    std::vector<Check> checks;
    std::cout << "\n===== " << key << " [vision] (" << model_path << ") =====\n";

    ChatSim sim;
    if (!sim.load(model_path, /*n_ctx*/ 4096)) {
        checks.push_back({key + " [vision]: model load", false, "loadModel failed", false});
        return checks;
    }
    if (!sim.ctx.initMultimodal(mmproj_path, /*use_gpu*/ false)) {
        checks.push_back({key + " [vision]: mmproj init", false, "initMultimodal failed", false});
        return checks;
    }
    sim.system_prompt = "You are a helpful assistant.";
    sim.enable_thinking = false; // want a direct one-word answer, not reasoning
    const std::string marker = mtmd_default_marker();
    const std::string q = marker + "\nIs the animal in this image a dog or a cat? Answer with one word.";
    // processMedia decodes inside loadPrompt, so n_past afterward is the full
    // position; read the reused prefix (where the chunk eval resumed) instead.
    auto reused_prefix = [&]() { return (size_t) sim.ctx.mtmd_wrapper->last_reused_n_past; };
    const std::vector<std::string> DOG = {"dog", "puppy", "canine"};
    const std::vector<std::string> CAT = {"cat", "kitten", "kitty", "feline"};

    try {
        // Text-only turn first: lays text checkpoints at positions that land
        // mid-chunk once the next prompt has an image chunk — recovery must
        // reject those (chunk-alignment retry loop) instead of restoring them.
        sim.user("Hello! Please remember the word starfish.");
        (void) sim.assistant_turn("text-pre", /*max_new*/ 12);

        // --- Session 1: dog image -------------------------------------------
        sim.media_paths = { img_dog };
        sim.user(q);
        std::string a_dog = sim.assistant_turn("dog", /*max_new*/ 12).reply;
        size_t dog_prompt = sim.ctx.completion->num_prompt_tokens;
        std::cout << "    [dog img]  A:\"" << a_dog << "\"\n";
        checks.push_back({key + " [vision]: dog recognized", has_any_word(a_dog, DOG) && !has_any_word(a_dog, CAT),
                          "reply: '" + a_dog + "'", /*fix_target*/ false});

        // Follow-up on the SAME image -> the image must be reused, not re-encoded.
        sim.user("What color is it? Answer briefly.");
        (void) sim.assistant_turn("dog-follow", /*max_new*/ 16);
        size_t reuse_follow = reused_prefix();
        std::cout << "    [follow-up] reused_prefix=" << reuse_follow << "/" << dog_prompt << "\n";
        checks.push_back({key + " [vision]: image reused (not re-encoded)",
                          reuse_follow + 12 >= dog_prompt,
                          "reused_prefix " + std::to_string(reuse_follow) + "/" +
                          std::to_string(dog_prompt), /*fix_target*/ true});

        // --- Edit the image turn's text (same image) --------------------------
        // Divergence before the image's trailing text: recovery must pick a
        // chunk-aligned restore point or fall back cleanly — a skipped/shifted
        // image chunk would show up as a wrong animal.
        sim.history.pop_back(); // follow-up reply
        sim.history.pop_back(); // follow-up question
        sim.history.pop_back(); // dog reply
        sim.history.pop_back(); // original dog question
        sim.user(marker + "\nLook carefully: is the animal in this picture a dog or a cat? One word.");
        std::string a_edit = sim.assistant_turn("dog-edit", /*max_new*/ 12).reply;
        std::cout << "    [edit img-turn] A:\"" << a_edit << "\"  reused_prefix="
                  << reused_prefix() << "\n";
        checks.push_back({key + " [vision]: edited image turn still sees the dog",
                          has_any_word(a_edit, DOG) && !has_any_word(a_edit, CAT),
                          "reply: '" + a_edit + "'", /*fix_target*/ false});

        // --- Trailing image (regression: media placeholders at embd's tail) --
        // No marker in the text: processMedia auto-appends it, so the prompt
        // ENDS with image tokens — the shape that crashed computePreBoundary
        // (std::out_of_range on LLAMA_TOKEN_NULL) before the media branch
        // stopped arming prompt-region snapshots.
        sim.history.pop_back();
        sim.history.pop_back();
        sim.user("Answer with one word: is the animal shown here a dog or a cat?");
        std::string a_tail = sim.assistant_turn("dog-tail", /*max_new*/ 12).reply;
        std::cout << "    [trailing img] A:\"" << a_tail << "\"\n";
        // The regression this guards is the std::out_of_range crash / busy context
        // (finding #1): reaching here without throwing IS the pass. Some models
        // (gemma4/gemma3n) decline an image-at-end prompt and emit nothing — that
        // is fine; only a crash or the WRONG animal (stale-image leak) must fail.
        checks.push_back({key + " [vision]: trailing-image prompt (no crash, no wrong-animal)",
                          !has_any_word(a_tail, CAT),
                          "reply: '" + a_tail + "'", /*fix_target*/ false});

        // --- Session 2: swap to the cat image, no cache clear ----------------
        sim.media_paths = { img_cat };
        sim.new_session("You are a helpful assistant.");
        sim.user(q);
        std::string a_cat = sim.assistant_turn("cat", /*max_new*/ 12).reply;
        std::cout << "    [cat img]  A:\"" << a_cat << "\"  (swapped from dog)\n";
        // Correct answer AND the previous dog image must not leak into it.
        checks.push_back({key + " [vision]: swapped image -> cat, no dog leak",
                          has_any_word(a_cat, CAT) && !has_any_word(a_cat, DOG),
                          "reply: '" + a_cat + "'", /*fix_target*/ false});

        // --- Session 3: no image at all, no cache clear ----------------------
        sim.media_paths.clear();
        sim.new_session("You are a helpful assistant.");
        sim.user("Describe what is currently shown to you in one sentence.");
        std::string a_none = sim.assistant_turn("no-img", /*max_new*/ 40).reply;
        std::cout << "    [no img]   A:\"" << a_none << "\"\n";
        // Nothing is shown -> neither the dog nor the cat may be mentioned.
        checks.push_back({key + " [vision]: removed image, no animal leak",
                          !has_any_word(a_none, DOG) && !has_any_word(a_none, CAT),
                          "reply: '" + a_none + "'", /*fix_target*/ false});
    } catch (const std::exception &e) {
        checks.push_back({key + " [vision]: runnable", false,
                          std::string("multimodal decode failed: ") + e.what(), false});
    }
    return checks;
}

// Verify the host-configurable budget actually takes effect: with the cache
// disabled (budget 0) a recurrent/hybrid model must fall back to the full-wipe
// behaviour, i.e. an append turn reuses nothing.
std::vector<Check> run_config_test(const std::string &key, const std::string &path) {
    std::vector<Check> checks;
    std::cout << "\n===== " << key << " [config: budget=0 disables cache] =====\n";
    ChatSim sim;
    sim.state_cache_budget_mb = 0; // disable
    if (!sim.load(path, /*n_ctx*/ 2048)) {
        checks.push_back({key + " [config]: model load", false, "loadModel failed", false});
        return checks;
    }
    sim.system_prompt = "You are a helpful assistant.";
    sim.enable_thinking = false;
    // Use resend (roll back the whole reply) so seq_rm genuinely fails on this
    // hybrid model and reuse depends on a checkpoint — which the disabled cache
    // has none of, forcing the full-wipe fallback.
    sim.user("Hi, tell me about the ocean.");
    TurnMetric send = sim.assistant_turn("send", /*max_new*/ 24);
    sim.history.pop_back();
    TurnMetric resend = sim.assistant_turn("resend", /*max_new*/ 24);
    print_turn(resend);
    checks.push_back({key + " [config]: budget=0 disables reuse", resend.reused == 0,
                      "reused " + std::to_string(resend.reused) + "/" +
                      std::to_string(resend.reusable) + " (expected 0)", /*fix_target*/ false});
    return checks;
}

// embedding()/rerank() drive throwaway prompts through the same completion path;
// their state must NOT pollute the chat's checkpoint cache. Warm the cache with a
// chat turn, then run an append turn (shares the prefix, so no full-wipe) once with
// the guard on and once off: guarded captures nothing new, unguarded does.
std::vector<Check> run_state_cache_guard_test(const std::string &key, const std::string &path) {
    std::vector<Check> checks;
    std::cout << "\n===== " << key << " [guard: embedding/rerank must not pollute checkpoints] =====\n";

    auto probe = [&](bool allow) -> std::pair<size_t, size_t> {
        ChatSim sim;
        if (!sim.load(path, /*n_ctx*/ 2048)) return {0, 999};
        sim.system_prompt = "You are a helpful assistant.";
        sim.enable_thinking = false;
        sim.user("Tell me about the ocean in one sentence.");
        sim.assistant_turn("warm", /*max_new*/ 24);
        auto *cmpl = sim.ctx.completion;
        const size_t base = cmpl->state_checkpoints.size();

        // Append a related turn (shares the warm prefix -> restores a checkpoint,
        // no full-wipe) driven manually with the given capture flag.
        sim.user("Now tell me about mountains in one sentence.");
        sim.ctx.params.prompt = sim.render(/*add_generation_prompt*/ true);
        sim.ctx.params.n_predict = 24;
        cmpl->rewind();
        cmpl->initSampling();
        cmpl->loadPrompt({}, /*allow_state_cache*/ allow);
        cmpl->beginCompletion();
        for (int i = 0; i < 24 && cmpl->has_next_token; i++) {
            auto o = cmpl->nextToken();
            if (o.tok == -1 || cmpl->stopped_eos) break;
        }
        cmpl->endCompletion();
        return {base, cmpl->state_checkpoints.size()};
    };

    auto guarded   = probe(/*allow*/ false);
    auto unguarded = probe(/*allow*/ true);
    std::cout << "    guarded:   base=" << guarded.first  << " after=" << guarded.second  << "\n";
    std::cout << "    unguarded: base=" << unguarded.first << " after=" << unguarded.second << "\n";

    checks.push_back({key + " [guard]: cache was warmed", guarded.first > 0,
                      "base=" + std::to_string(guarded.first), /*fix_target*/ false});
    checks.push_back({key + " [guard]: guarded call captures nothing",
                      guarded.second == guarded.first,
                      "base=" + std::to_string(guarded.first) + " after=" + std::to_string(guarded.second),
                      /*fix_target*/ false});
    checks.push_back({key + " [guard]: unguarded call would capture",
                      unguarded.second > unguarded.first,
                      "base=" + std::to_string(unguarded.first) + " after=" + std::to_string(unguarded.second),
                      /*fix_target*/ false});
    return checks;
}

// Stop a reply mid-stream (the host "stop" button), keep the partial reply in
// history, and keep chatting: the trimmed cache must still be reusable.
std::vector<Check> run_interrupt_test(const std::string &key, const std::string &path) {
    std::vector<Check> checks;
    std::cout << "\n===== " << key << " [interrupt: stop mid-reply, continue] =====\n";
    ChatSim sim;
    if (!sim.load(path, /*n_ctx*/ 2048)) {
        checks.push_back({key + " [interrupt]: model load", false, "loadModel failed", false});
        return checks;
    }
    sim.system_prompt = "You are a helpful assistant.";
    sim.enable_thinking = false;
    sim.user("Tell me a long story about a lighthouse keeper.");

    auto *cmpl = sim.ctx.completion;
    sim.ctx.params.prompt = sim.render(/*add_generation_prompt*/ true);
    sim.ctx.params.n_predict = 64;
    cmpl->rewind();
    cmpl->initSampling();
    cmpl->loadPrompt({});
    cmpl->beginCompletion();
    std::string partial;
    for (int i = 0; i < 8 && cmpl->has_next_token; i++) {
        auto o = cmpl->nextToken();
        if (o.tok == -1) break;
        partial += o.text;
    }
    cmpl->is_interrupted = true;
    (void) cmpl->nextToken(); // takes the interrupt path (embd trimmed to n_past)
    cmpl->endCompletion();
    cmpl->is_interrupted = false;
    sim.prev_embd = cmpl->embd;
    sim.history.push_back({"assistant", partial});
    std::cout << "    [partial] \"" << partial << "\"\n";

    sim.user("Please continue the story briefly.");
    TurnMetric next = sim.assistant_turn("post-intr", /*max_new*/ 24);
    print_turn(next);
    bool reuse_ok = next.reusable > 0 && next.reused >= (size_t) (0.5 * next.reusable);
    checks.push_back({key + " [interrupt]: post-interrupt reuse", reuse_ok,
                      "reused " + std::to_string(next.reused) + "/" +
                      std::to_string(next.reusable), /*fix_target*/ true});
    checks.push_back({key + " [interrupt]: post-interrupt reply non-empty", !next.reply.empty(),
                      "reply: '" + next.reply + "'", /*fix_target*/ false});
    return checks;
}

// Overflow a small context so the shift path runs: checkpoints must be dropped
// (positions remapped), the conversation must survive, and reuse must recover
// once fresh checkpoints are laid down.
std::vector<Check> run_ctx_shift_test(const std::string &key, const std::string &path) {
    std::vector<Check> checks;
    std::cout << "\n===== " << key << " [ctx-shift: overflow a 512-token context] =====\n";
    ChatSim sim;
    sim.ctx_shift = true;
    if (!sim.load(path, /*n_ctx*/ 512)) {
        checks.push_back({key + " [shift]: model load", false, "loadModel failed", false});
        return checks;
    }
    sim.system_prompt = "You are a helpful assistant.";
    sim.enable_thinking = false;

    bool shifted = false, all_nonempty = true, reuse_after_shift = false;
    int shift_turn = -1;
    try {
        // Keep chatting a few turns past the first shift so recovery is observable.
        for (int t = 0; t < 14 && !(shifted && t > shift_turn + 3); t++) {
            sim.user("Tell me a few sentences about interesting topic number " +
                     std::to_string(t + 1) + ".");
            TurnMetric tm = sim.assistant_turn("shift-t" + std::to_string(t + 1), /*max_new*/ 56);
            print_turn(tm);
            if (tm.reply.empty()) all_nonempty = false;
            if (sim.ctx.completion->truncated && !shifted) { shifted = true; shift_turn = t; }
            if (shifted && t > shift_turn && tm.reused > 0) reuse_after_shift = true;
            if (reuse_after_shift) break;
        }
    } catch (const std::exception &e) {
        checks.push_back({key + " [shift]: survives context shift", false,
                          std::string("threw: ") + e.what(), /*fix_target*/ false});
        return checks;
    }
    checks.push_back({key + " [shift]: shift triggered", shifted,
                      shifted ? "truncated at turn " + std::to_string(shift_turn + 1)
                              : "never overflowed (test vacuous)", /*fix_target*/ false});
    checks.push_back({key + " [shift]: replies non-empty across shift", all_nonempty,
                      all_nonempty ? "ok" : "an empty reply appeared", /*fix_target*/ false});
    checks.push_back({key + " [shift]: reuse recovers after shift", reuse_after_shift,
                      reuse_after_shift ? "reused > 0 post-shift" : "no post-shift reuse seen",
                      /*fix_target*/ true});
    return checks;
}

// Boundary-less prompt (no chat template): the 256-token interval fallback must
// still give a restore point inside the shared prefix.
std::vector<Check> run_interval_fallback_test(const std::string &key, const std::string &path) {
    std::vector<Check> checks;
    std::cout << "\n===== " << key << " [interval: boundary-less prompt fallback] =====\n";
    ChatSim sim;
    if (!sim.load(path, /*n_ctx*/ 2048)) {
        checks.push_back({key + " [interval]: model load", false, "loadModel failed", false});
        return checks;
    }
    auto *cmpl = sim.ctx.completion;

    std::string para;
    while (sim.tokenize_like_loadprompt(para).size() < 850) {
        para += "The lighthouse stands on the cliff and the sea below it is grey. ";
    }
    auto drive = [&](const std::string &prompt) -> size_t {
        sim.ctx.params.prompt = prompt;
        sim.ctx.params.n_predict = 4;
        cmpl->rewind();
        cmpl->initSampling();
        cmpl->loadPrompt({});
        const size_t reused = (size_t) cmpl->n_past;
        cmpl->beginCompletion();
        for (int i = 0; i < 4 && cmpl->has_next_token; i++) {
            auto o = cmpl->nextToken();
            if (o.tok == -1 || cmpl->stopped_eos) break;
        }
        cmpl->endCompletion();
        return reused;
    };
    (void) drive(para);
    const bool no_boundaries = cmpl->boundary_ckpts.empty();
    const size_t shared = sim.tokenize_like_loadprompt(para).size();
    const size_t reused = drive(para + "Suddenly a storm came in from the west.");
    std::cout << "    [interval] reused=" << reused << " shared~=" << shared << "\n";
    checks.push_back({key + " [interval]: prompt is boundary-less", no_boundaries,
                      no_boundaries ? "interval fallback active" : "boundaries found (test vacuous)",
                      /*fix_target*/ false});
    // Floor: within one interval + margin of the shared prefix.
    bool ok = reused > 0 && reused + 264 >= shared;
    checks.push_back({key + " [interval]: fallback reuse floor", ok,
                      "reused " + std::to_string(reused) + " of ~" + std::to_string(shared),
                      /*fix_target*/ true});
    return checks;
}

// Byte-budget stress: a budget that fits ~one snapshot must still keep the
// pinned system-boundary snapshot alive for cross-session reuse.
std::vector<Check> run_eviction_stress_test(const std::string &key, const std::string &path) {
    std::vector<Check> checks;
    std::cout << "\n===== " << key << " [eviction: 24 MiB budget vs ~22 MiB snapshots] =====\n";
    ChatSim sim;
    sim.state_cache_budget_mb = 24;
    if (!sim.load(path, /*n_ctx*/ 4096)) {
        checks.push_back({key + " [evict]: model load", false, "loadModel failed", false});
        return checks;
    }
    std::string sys = "You are a helpful assistant. ";
    for (int i = 0; i < 24; i++) sys += "Always answer briefly and stay on topic. ";
    sim.system_prompt = sys;
    sim.enable_thinking = false;
    try {
        for (int t = 0; t < 6; t++) {
            sim.user("Quick fact number " + std::to_string(t + 1) + ", please.");
            (void) sim.assistant_turn("evict-t" + std::to_string(t + 1), /*max_new*/ 16);
        }
    } catch (const std::exception &e) {
        checks.push_back({key + " [evict]: survives budget pressure", false,
                          std::string("threw: ") + e.what(), /*fix_target*/ false});
        return checks;
    }
    const size_t held = sim.ctx.completion->state_checkpoints.size();
    checks.push_back({key + " [evict]: byte budget bounds the cache", held <= 2,
                      "holding " + std::to_string(held) + " snapshots", /*fix_target*/ false});

    sim.new_session(sys);
    sim.user("What is the capital of France?");
    TurnMetric s2 = sim.assistant_turn("evict-s2", /*max_new*/ 12);
    print_turn(s2);
    bool sys_reused = s2.reusable > 0 && s2.reused >= (size_t) (0.5 * s2.reusable);
    checks.push_back({key + " [evict]: pinned snapshot survives for new session", sys_reused,
                      "reused " + std::to_string(s2.reused) + "/" + std::to_string(s2.reusable),
                      /*fix_target*/ true});
    return checks;
}

// Count-cap edges: max_checkpoints=1 (exercises the keep_tail=0 boundary
// pre-filter) and =0 (no count cap; growth bounded by the byte budget only).
std::vector<Check> run_checkpoint_cap_tests(const std::string &key, const std::string &path) {
    std::vector<Check> checks;
    std::cout << "\n===== " << key << " [caps: max_checkpoints = 1 and 0] =====\n";
    {
        ChatSim sim;
        sim.state_cache_max_checkpoints = 1;
        if (!sim.load(path, /*n_ctx*/ 2048)) {
            checks.push_back({key + " [caps]: model load", false, "loadModel failed", false});
            return checks;
        }
        sim.system_prompt = "You are a helpful assistant.";
        sim.enable_thinking = false;
        sim.user("Tell me about rivers in a sentence.");
        (void) sim.assistant_turn("cap1-t1", /*max_new*/ 16);
        sim.user("And lakes?");
        (void) sim.assistant_turn("cap1-t2", /*max_new*/ 16);
        sim.history.pop_back();
        TurnMetric resend = sim.assistant_turn("cap1-resend", /*max_new*/ 16);
        print_turn(resend);
        const size_t held = sim.ctx.completion->state_checkpoints.size();
        checks.push_back({key + " [caps]: max=1 holds one snapshot", held <= 1,
                          "holding " + std::to_string(held), /*fix_target*/ false});
        checks.push_back({key + " [caps]: max=1 still reuses on resend", resend.reused > 0,
                          "reused " + std::to_string(resend.reused) + "/" +
                          std::to_string(resend.reusable), /*fix_target*/ true});
    }
    {
        ChatSim sim;
        sim.state_cache_max_checkpoints = 0; // no count cap
        if (!sim.load(path, /*n_ctx*/ 4096)) {
            checks.push_back({key + " [caps]: model load (0)", false, "loadModel failed", false});
            return checks;
        }
        sim.system_prompt = "You are a helpful assistant.";
        sim.enable_thinking = false;
        for (int t = 0; t < 10; t++) {
            sim.user("Give me two sentences about famous city number " +
                     std::to_string(t + 1) + " and what it is known for.");
            (void) sim.assistant_turn("cap0-t" + std::to_string(t + 1), /*max_new*/ 24);
        }
        const size_t held = sim.ctx.completion->state_checkpoints.size();
        std::cout << "    [caps] max=0 holding " << held << " snapshots\n";
        checks.push_back({key + " [caps]: max=0 exceeds the default cap", held > 8,
                          "holding " + std::to_string(held), /*fix_target*/ false});
    }
    {
        // Default cap (8): the same 10-turn chat must overflow it, evict
        // oldest-first, keep the pinned smallest-position snapshot, and still
        // serve a new session from it afterwards.
        ChatSim sim;
        if (!sim.load(path, /*n_ctx*/ 4096)) {
            checks.push_back({key + " [caps]: model load (dflt)", false, "loadModel failed", false});
            return checks;
        }
        std::string sys = "You are a helpful assistant. ";
        for (int i = 0; i < 20; i++) sys += "Always answer clearly and stay on topic. ";
        sim.system_prompt = sys;
        sim.enable_thinking = false;
        for (int t = 0; t < 10; t++) {
            sim.user("Give me two sentences about famous river number " +
                     std::to_string(t + 1) + " and where it flows.");
            (void) sim.assistant_turn("dflt-t" + std::to_string(t + 1), /*max_new*/ 24);
        }
        auto *cmpl = sim.ctx.completion;
        const size_t held = cmpl->state_checkpoints.size();
        size_t smallest = SIZE_MAX;
        for (const auto &c : cmpl->state_checkpoints) smallest = std::min(smallest, c.n_tokens());
        std::cout << "    [caps] default cap holding " << held
                  << " snapshots, smallest at " << smallest << "\n";
        checks.push_back({key + " [caps]: default cap bounds the cache", held <= 8,
                          "holding " + std::to_string(held), /*fix_target*/ false});
        // The pinned snapshot is the system boundary: well inside the system
        // prompt region, not a recent-turn position.
        const size_t sys_len = sim.tokenize_like_loadprompt(sys).size();
        checks.push_back({key + " [caps]: eviction keeps the pinned system boundary",
                          smallest != SIZE_MAX && smallest <= sys_len + 32,
                          "smallest " + std::to_string(smallest) + " vs system ~" +
                          std::to_string(sys_len), /*fix_target*/ false});
        sim.new_session(sys);
        sim.user("What is the capital of Japan?");
        TurnMetric s2 = sim.assistant_turn("dflt-s2", /*max_new*/ 12);
        checks.push_back({key + " [caps]: post-eviction new-session reuse",
                          s2.reusable > 0 && s2.reused >= (size_t) (0.5 * s2.reusable),
                          "reused " + std::to_string(s2.reused) + "/" +
                          std::to_string(s2.reusable), /*fix_target*/ true});
    }
    return checks;
}

// SWA auto-disable: a budget far below the saturated snapshot size must switch
// the cache off cleanly mid-session, with later turns still correct.
std::vector<Check> run_swa_disable_test(const std::string &key, const std::string &path) {
    std::vector<Check> checks;
    std::cout << "\n===== " << key << " [swa-guard: tiny budget must auto-disable] =====\n";
    ChatSim sim;
    sim.state_cache_budget_mb = 4;
    if (!sim.load(path, /*n_ctx*/ 2048)) {
        checks.push_back({key + " [swa-guard]: model load", false, "loadModel failed", false});
        return checks;
    }
    sim.system_prompt = "You are a helpful assistant.";
    sim.enable_thinking = false;
    sim.user("Say hello in one short sentence.");
    (void) sim.assistant_turn("guard-t1", /*max_new*/ 12);
    auto *cmpl = sim.ctx.completion;
    checks.push_back({key + " [swa-guard]: cache auto-disabled",
                      !cmpl->state_cache_enabled && cmpl->state_checkpoints.empty(),
                      std::string("enabled=") + (cmpl->state_cache_enabled ? "1" : "0") +
                      " snapshots=" + std::to_string(cmpl->state_checkpoints.size()),
                      /*fix_target*/ false});
    sim.user("And say goodbye.");
    TurnMetric t2 = sim.assistant_turn("guard-t2", /*max_new*/ 12);
    checks.push_back({key + " [swa-guard]: fallback still generates", !t2.reply.empty(),
                      "reply: '" + t2.reply + "'", /*fix_target*/ false});
    return checks;
}

// Long SWA conversation: keep chatting well past the window size so it wraps
// several times; reuse and output must hold.
std::vector<Check> run_swa_wrap_test(const std::string &key, const std::string &path) {
    std::vector<Check> checks;
    std::cout << "\n===== " << key << " [swa-wrap: conversation >> window size] =====\n";
    ChatSim sim;
    if (!sim.load(path, /*n_ctx*/ 4096)) {
        checks.push_back({key + " [swa-wrap]: model load", false, "loadModel failed", false});
        return checks;
    }
    sim.system_prompt = "You are a helpful assistant.";
    sim.enable_thinking = false;
    bool all_nonempty = true, reuse_held = true;
    size_t final_tokens = 0;
    try {
        for (int t = 0; t < 10; t++) {
            sim.user("Write a detailed paragraph about ocean animal number " +
                     std::to_string(t + 1) + ": its habitat, diet, and one unusual fact.");
            TurnMetric tm = sim.assistant_turn("wrap-t" + std::to_string(t + 1), /*max_new*/ 112);
            print_turn(tm);
            if (tm.reply.empty()) all_nonempty = false;
            if (t >= 1 && tm.reusable > 0 && tm.reused < (size_t) (0.5 * tm.reusable)) reuse_held = false;
            final_tokens = tm.prompt_tokens;
        }
    } catch (const std::exception &e) {
        checks.push_back({key + " [swa-wrap]: survives window wrap", false,
                          std::string("threw: ") + e.what(), /*fix_target*/ false});
        return checks;
    }
    // "Wrapped" = the window slid well past its size at least once.
    const size_t n_swa = (size_t) llama_model_n_swa(sim.ctx.model);
    checks.push_back({key + " [swa-wrap]: window actually wrapped", final_tokens > n_swa + 128,
                      std::to_string(final_tokens) + " tokens vs window " + std::to_string(n_swa),
                      /*fix_target*/ false});
    checks.push_back({key + " [swa-wrap]: replies non-empty throughout", all_nonempty,
                      all_nonempty ? "ok" : "empty reply appeared", /*fix_target*/ false});
    checks.push_back({key + " [swa-wrap]: reuse holds across wraps", reuse_held,
                      reuse_held ? "every turn >= 0.5x ceiling" : "a turn fell below 0.5x",
                      /*fix_target*/ true});
    return checks;
}

// Session save/load: checkpoints deliberately survive a session swap (restores
// are token-verified, so stale ones are inert); prove no divergent-branch leak.
std::vector<Check> run_session_saveload_test(const std::string &key, const std::string &path) {
    std::vector<Check> checks;
    std::cout << "\n===== " << key << " [session: save / diverge / load / continue] =====\n";
    ChatSim sim;
    if (!sim.load(path, /*n_ctx*/ 2048)) {
        checks.push_back({key + " [session]: model load", false, "loadModel failed", false});
        return checks;
    }
    sim.system_prompt = "You are a helpful assistant.";
    sim.enable_thinking = false;
    auto *cmpl = sim.ctx.completion;

    sim.user("My hobby is cycling. Please remember that.");
    (void) sim.assistant_turn("sess-t1", /*max_new*/ 16);
    sim.user("What is the capital of France?");
    (void) sim.assistant_turn("sess-t2", /*max_new*/ 12);

    const auto save_path =
        (std::filesystem::temp_directory_path() / "kv_reuse_session_test.bin").string();
    const std::vector<llama_token> saved_tokens = cmpl->embd;
    const auto saved_history = sim.history;
    if (!llama_state_save_file(sim.ctx.ctx, save_path.c_str(),
                               saved_tokens.data(), saved_tokens.size())) {
        checks.push_back({key + " [session]: state save", false, "llama_state_save_file failed", false});
        return checks;
    }

    // Divergent branch that must NOT leak after the load.
    sim.user("My secret code is 4711. Never forget it.");
    (void) sim.assistant_turn("sess-diverge", /*max_new*/ 16);

    std::vector<llama_token> loaded(saved_tokens.size() + 16);
    size_t n_loaded = 0;
    if (!llama_state_load_file(sim.ctx.ctx, save_path.c_str(),
                               loaded.data(), loaded.size(), &n_loaded)) {
        checks.push_back({key + " [session]: state load", false, "llama_state_load_file failed", false});
        return checks;
    }
    std::filesystem::remove(save_path);
    loaded.resize(n_loaded);
    cmpl->embd = loaded;
    cmpl->n_past = (llama_pos) n_loaded;
    sim.prev_embd = loaded;
    sim.history = saved_history;

    sim.user("What is my secret code? If I never told you one, say you don't know.");
    TurnMetric q1 = sim.assistant_turn("sess-leak", /*max_new*/ 24);
    std::cout << "    [session] Q:secret code? A:\"" << q1.reply << "\"\n";
    checks.push_back({key + " [session]: no divergent-branch leak", !contains_ci(q1.reply, "4711"),
                      "reply: '" + q1.reply + "'", /*fix_target*/ false});

    // Continue the restored session; output must stay coherent (recall itself is
    // only asserted on `strong` models in the main scenarios — 230M is not one).
    sim.user("What hobby did I mention earlier? One word.");
    TurnMetric q2 = sim.assistant_turn("sess-recall", /*max_new*/ 12);
    std::cout << "    [session] Q:hobby? A:\"" << q2.reply << "\"\n";
    checks.push_back({key + " [session]: restored session keeps generating",
                      !q2.reply.empty(), "reply: '" + q2.reply + "'",
                      /*fix_target*/ false});
    return checks;
}

// Mean NLL (nats/token) of `probe_text` as a continuation of the context that
// loadPrompt just prepared. Decodes the pending prompt tail, then scores each
// probe token against the model's prediction before feeding it.
double score_probe(ChatSim &sim, const std::string &probe_text) {
    auto *cmpl = sim.ctx.completion;
    auto *lctx = sim.ctx.ctx;
    const llama_vocab *vocab = llama_model_get_vocab(sim.ctx.model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_pos pos = cmpl->n_past;
    auto &embd = cmpl->embd;
    while (pos < (llama_pos) embd.size()) {
        const int n_eval = std::min<int>((int) embd.size() - pos, 512);
        if (llama_decode(lctx, llama_batch_get_one(embd.data() + pos, n_eval))) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        pos += n_eval;
    }
    std::vector<llama_token> probe = sim.tokenize_like_loadprompt(probe_text);
    if (!probe.empty() && probe.front() == llama_vocab_bos(vocab)) probe.erase(probe.begin());

    double nll = 0;
    int scored = 0;
    for (llama_token tgt : probe) {
        const float *logits = llama_get_logits_ith(lctx, -1);
        if (logits == nullptr) return std::numeric_limits<double>::quiet_NaN();
        double mx = -1e30;
        for (int i = 0; i < n_vocab; i++) mx = std::max(mx, (double) logits[i]);
        double se = 0;
        for (int i = 0; i < n_vocab; i++) se += std::exp((double) logits[i] - mx);
        nll += -(((double) logits[tgt] - mx) - std::log(se));
        scored++;
        llama_token t = tgt;
        if (llama_decode(lctx, llama_batch_get_one(&t, 1))) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    return scored > 0 ? nll / scored : std::numeric_limits<double>::quiet_NaN();
}

// State fidelity: a corrupted restored state shifts the model's predictive
// distribution. Score the same probe continuation after (a) a checkpoint-restore
// reuse and (b) a cold recompute of the SAME prompt; the two must agree within
// the measured run-to-run FP noise floor. Catches distribution-level damage
// that greedy-prefix comparison cannot.
std::vector<Check> run_state_fidelity_test(const std::string &key, const std::string &path,
                                           bool use_mtp = false) {
    std::vector<Check> checks;
    const std::string tag = use_mtp ? " [fidelity+mtp]" : " [fidelity]";
    std::cout << "\n===== " << key << tag
              << ": reuse must not shift the distribution =====\n";
    ChatSim sim;
    sim.use_mtp = use_mtp;
    if (!sim.load(path, /*n_ctx*/ 2048)) {
        checks.push_back({key + tag + ": model load", false, "loadModel failed", false});
        return checks;
    }
    std::string sys = "You are a helpful assistant. ";
    for (int i = 0; i < 20; i++) sys += "Always answer clearly, briefly, and stay on topic. ";
    sim.system_prompt = sys;
    sim.enable_thinking = false;
    auto *cmpl = sim.ctx.completion;

    const std::string probe =
        "The lighthouse keeper climbed the narrow stairs every evening to light the lamp, "
        "and the ships far out at sea trusted that small point of light more than any map "
        "they carried on board.";

    auto ingest = [&](const std::string &prompt) -> size_t {
        sim.ctx.params.prompt = prompt;
        sim.ctx.params.n_predict = 0;
        cmpl->rewind();
        cmpl->initSampling();
        cmpl->loadPrompt({});
        return (size_t) std::max<llama_pos>(0, cmpl->n_past);
    };
    // FID_NO_WARMGEN=1 skips the warm-turn generation. Diagnostic for the
    // mem-shared-MTP divergence: if skipping generation (so the draft never
    // speculates into the shared window) makes the new-session restore clean,
    // the culprit is the draft's uncleaned speculative cells.
    const bool skip_warmgen = std::getenv("FID_NO_WARMGEN") != nullptr;
    auto generate_a_bit = [&](int n) {
        if (skip_warmgen) return;
        cmpl->beginCompletion();
        sim.ctx.params.n_predict = n;
        cmpl->n_remain = n;
        for (int i = 0; i < n && cmpl->has_next_token; i++) {
            auto o = cmpl->nextToken();
            if (o.tok == -1 || cmpl->stopped_eos) break;
        }
        cmpl->endCompletion();
    };

    sim.user("Tell me briefly about the ocean.");
    const std::string prompt_a = sim.render(/*add_generation_prompt*/ true);
    const std::string prompt_b_sys_only = [&]() {
        auto saved = sim.history;
        sim.history.clear();
        sim.user("What can you help me with today?");
        std::string p = sim.render(true);
        sim.history = saved;
        return p;
    }();

    // Warm the cache: full ingest + a short generation so a resend must restore.
    (void) ingest(prompt_a);
    generate_a_bit(12);

    // (a) resend -> margin/boundary checkpoint restore on non-attention models.
    const size_t reused_resend = ingest(prompt_a);
    const double nll_resend = score_probe(sim, probe);

    // (b) new session sharing only the system prompt -> pinned-boundary restore.
    (void) ingest(prompt_a);
    generate_a_bit(12);
    const size_t reused_newsess = ingest(prompt_b_sys_only);
    const double nll_newsess = score_probe(sim, probe);

    // Cold references + noise floor (two independent cold recomputes).
    sim.ctx.clearCache(false);
    (void) ingest(prompt_a);
    const double nll_cold_a1 = score_probe(sim, probe);
    sim.ctx.clearCache(false);
    (void) ingest(prompt_a);
    const double nll_cold_a2 = score_probe(sim, probe);
    sim.ctx.clearCache(false);
    (void) ingest(prompt_b_sys_only);
    const double nll_cold_b = score_probe(sim, probe);

    const double floor = std::fabs(nll_cold_a1 - nll_cold_a2);
    // Scale-aware tolerance (see run_vision_fidelity): 2% of the NLL, so it
    // does not false-fail high-NLL models while the real corruptions (10x) fail.
    const double tol = std::max({0.05, 5.0 * floor, 0.02 * std::fabs(nll_cold_a1)});
    const double d_resend  = std::fabs(nll_resend - nll_cold_a1);
    const double d_newsess = std::fabs(nll_newsess - nll_cold_b);

    char buf[256];
    snprintf(buf, sizeof(buf),
             "resend %.4f vs cold %.4f (d=%.4f), new-session %.4f vs cold %.4f (d=%.4f), "
             "floor %.4f, tol %.4f, reused %zu/%zu",
             nll_resend, nll_cold_a1, d_resend, nll_newsess, nll_cold_b, d_newsess,
             floor, tol, reused_resend, reused_newsess);
    std::cout << "    [fidelity] " << buf << "\n";

    const bool valid = nll_resend == nll_resend && nll_newsess == nll_newsess &&
                       nll_cold_a1 == nll_cold_a1 && nll_cold_b == nll_cold_b;
    checks.push_back({key + tag + ": probe scored on all paths", valid, buf, false});
    if (valid) {
        checks.push_back({key + tag + ": resend restore keeps the distribution",
                          d_resend <= tol, buf, /*fix_target*/ false});
        checks.push_back({key + tag + ": new-session restore keeps the distribution",
                          d_newsess <= tol, buf, /*fix_target*/ false});
    }
    return checks;
}

// Coherence dump: run a reuse-heavy conversation (appends, a regenerate, an
// edit, a new session) with default cache settings and print every full reply,
// so a human/judge can read the actual text produced through restored state.
std::vector<Check> run_coherence_dump(const std::string &key, const std::string &path,
                                      bool use_mtp = false) {
    std::vector<Check> checks;
    const std::string tag = use_mtp ? " [coherence+mtp]" : " [coherence]";
    std::cout << "\n===== " << key << tag << ": full replies through reuse paths =====\n";
    ChatSim sim;
    sim.use_mtp = use_mtp;
    if (!sim.load(path, /*n_ctx*/ 2048)) {
        checks.push_back({key + tag + ": model load", false, "loadModel failed", false});
        return checks;
    }
    sim.system_prompt = "You are a helpful assistant. Answer in two or three sentences.";
    sim.enable_thinking = false;

    auto say = [&](const std::string &label, const std::string &msg, int n) {
        sim.user(msg);
        TurnMetric tm = sim.assistant_turn(label, n);
        std::cout << "  [" << label << "] reused " << tm.reused << "/" << tm.reusable
                  << "\n  Q: " << msg << "\n  A: " << tm.reply << "\n";
        return tm;
    };
    bool all_nonempty = true;
    try {
        all_nonempty &= !say("t1-append", "Why is the sky blue?", 64).reply.empty();
        all_nonempty &= !say("t2-append", "Does the same effect explain red sunsets?", 64).reply.empty();
        // regenerate: resend -> margin/boundary restore
        sim.history.pop_back();
        sim.history.pop_back();
        all_nonempty &= !say("t2-regen", "Does the same effect explain red sunsets?", 64).reply.empty();
        all_nonempty &= !say("t3-append", "Name one experiment that demonstrates it.", 64).reply.empty();
        // edit an earlier turn -> deep restore
        sim.history.erase(sim.history.begin() + 2, sim.history.end());
        all_nonempty &= !say("t2-edited", "And why is the ocean blue - same reason or different?", 64).reply.empty();
        // new session -> pinned system-boundary restore
        sim.new_session(sim.system_prompt);
        all_nonempty &= !say("s2-t1", "Give me a two-sentence summary of how rainbows form.", 64).reply.empty();
    } catch (const std::exception &e) {
        checks.push_back({key + tag + ": conversation survives", false,
                          std::string("threw: ") + e.what(), false});
        return checks;
    }
    checks.push_back({key + tag + ": all replies non-empty", all_nonempty,
                      all_nonempty ? "ok" : "an empty reply appeared", false});
    return checks;
}

// ------------------------------------------------------------------ model set

struct ModelEntry { std::string key; std::string file; bool mtp; bool instruct; bool strong; };

const std::vector<ModelEntry> kKnownModels = {
    //           key         file             mtp    instruct  strong (reliable recall)
    {"smollm2",  "smollm2.gguf",  false, true,  false}, // pure attention (control), 135M
    {"mamba",    "mamba.gguf",    false, false, false}, // pure recurrent, BASE (no chat)
    {"lfm2",     "lfm2.gguf",     false, true,  false}, // hybrid conv+attn, 230M
    {"granite4", "granite4.gguf", false, true,  false}, // hybrid mamba2+attn, 350M
    {"qwen35",   "qwen35.gguf",   true,  true,  true},  // hybrid + <think>; native MTP; 2B
    {"gemma4",   "gemma4.gguf",   true,  true,  true},  // SWA (+ vision); MTP (mem-shared draft); 2B
    {"lfm2vl",   "lfm2vl.gguf",   false, true,  false}, // hybrid (LFM2) + vision; 450M
};

} // namespace

int main(int argc, char **argv) {
    const char *env_dir = std::getenv("MODELS_DIR");
    std::filesystem::path models_dir =
        env_dir ? std::filesystem::path(env_dir)
                : std::filesystem::path(__FILE__).parent_path() / "models";

    std::vector<std::string> want;
    for (int i = 1; i < argc; i++) want.push_back(argv[i]);

    std::cout << "KV-cache-reuse harness\nmodels dir: " << models_dir << "\n";

    std::vector<Check> all;
    int models_run = 0;
    for (const auto &m : kKnownModels) {
        if (!want.empty() &&
            std::find(want.begin(), want.end(), m.key) == want.end()) {
            continue;
        }
        std::filesystem::path p = models_dir / m.file;
        if (!std::filesystem::exists(p)) {
            std::cout << "[skip] " << m.key << " (" << p << " not found)\n";
            continue;
        }
        models_run++;
        const bool mtp_only = std::getenv("MTP_ONLY") != nullptr;
        const bool vision_only = std::getenv("VISION_ONLY") != nullptr;
        const bool fidelity_only = std::getenv("FIDELITY_ONLY") != nullptr;
        if (fidelity_only) {
            auto f = run_state_fidelity_test(m.key, p.string());
            all.insert(all.end(), f.begin(), f.end());
            if (m.mtp && !std::getenv("SKIP_MTP")) {
                auto fm = run_state_fidelity_test(m.key, p.string(), /*use_mtp*/ true);
                all.insert(all.end(), fm.begin(), fm.end());
            }
            continue;
        }
        if (std::getenv("COHERENCE_ONLY") != nullptr) {
            auto c = run_coherence_dump(m.key, p.string(),
                                        /*use_mtp*/ m.mtp && std::getenv("WITH_MTP") != nullptr);
            all.insert(all.end(), c.begin(), c.end());
            continue;
        }
        if (!mtp_only && !vision_only) {
            auto checks = run_model(m.key, p.string(), m.instruct, m.strong);
            all.insert(all.end(), checks.begin(), checks.end());
            auto f = run_state_fidelity_test(m.key, p.string());
            all.insert(all.end(), f.begin(), f.end());
        }

        // Also exercise the MTP prompt-eval path on models that support it.
        if (m.mtp && !std::getenv("SKIP_MTP") && !vision_only) {
            auto mtp_checks = run_model_mtp(m.key, p.string());
            all.insert(all.end(), mtp_checks.begin(), mtp_checks.end());
        }

        // Exercise the multimodal prompt path when an mmproj and test images exist.
        std::filesystem::path mmproj, img_dog = models_dir / "test_dog.jpg",
                                       img_cat = models_dir / "test_cat.jpg";
        for (const auto &e : std::filesystem::directory_iterator(models_dir)) {
            const std::string fn = e.path().filename().string();
            if (fn.rfind(m.key + ".mmproj", 0) == 0) { mmproj = e.path(); break; }
        }
        if (!mmproj.empty() && std::filesystem::exists(img_dog) &&
            std::filesystem::exists(img_cat) && !std::getenv("SKIP_VISION") && !mtp_only) {
            auto v = run_model_multimodal(m.key, p.string(), mmproj.string(),
                                          img_dog.string(), img_cat.string());
            all.insert(all.end(), v.begin(), v.end());
            auto vf = run_vision_fidelity(m.key, p.string(), mmproj.string(), img_dog.string());
            all.insert(all.end(), vf.begin(), vf.end());
        }

        // Config / lifecycle scenarios: run each on the model it's about, and
        // the generic ones on the fastest hybrid (lfm2).
        if (m.key == "lfm2" && !mtp_only && !vision_only) {
            auto c = run_config_test(m.key, p.string());
            all.insert(all.end(), c.begin(), c.end());
            auto g = run_state_cache_guard_test(m.key, p.string());
            all.insert(all.end(), g.begin(), g.end());
            auto i = run_interrupt_test(m.key, p.string());
            all.insert(all.end(), i.begin(), i.end());
            auto s = run_ctx_shift_test(m.key, p.string());
            all.insert(all.end(), s.begin(), s.end());
            auto k = run_checkpoint_cap_tests(m.key, p.string());
            all.insert(all.end(), k.begin(), k.end());
            auto ss = run_session_saveload_test(m.key, p.string());
            all.insert(all.end(), ss.begin(), ss.end());
        }
        if (m.key == "mamba" && !mtp_only && !vision_only) {
            auto iv = run_interval_fallback_test(m.key, p.string());
            all.insert(all.end(), iv.begin(), iv.end());
        }
        if (m.key == "granite4" && !mtp_only && !vision_only) {
            auto ev = run_eviction_stress_test(m.key, p.string());
            all.insert(all.end(), ev.begin(), ev.end());
        }
        if (m.key == "gemma4" && !mtp_only && !vision_only) {
            auto sd = run_swa_disable_test(m.key, p.string());
            all.insert(all.end(), sd.begin(), sd.end());
            auto sw = run_swa_wrap_test(m.key, p.string());
            all.insert(all.end(), sw.begin(), sw.end());
        }
    }

    if (models_run == 0) {
        std::cout << "\nNo models found. Run tests/models/download.sh first.\n";
        return 2;
    }

    std::cout << "\n==================== SUMMARY ====================\n";
    int failed = 0, fix_target_failed = 0;
    for (const auto &c : all) {
        std::cout << (c.pass ? "  PASS " : "  FAIL ")
                  << std::left << std::setw(40) << c.name
                  << (c.fix_target ? "[fix-target] " : "[invariant]  ")
                  << c.detail << "\n";
        if (!c.pass) {
            failed++;
            if (c.fix_target) fix_target_failed++;
        }
    }
    std::cout << "\n" << (all.size() - failed) << "/" << all.size() << " checks passed";
    if (failed > 0) {
        std::cout << "  (" << fix_target_failed
                  << " fix-target failures = KV reuse not working; "
                  << (failed - fix_target_failed) << " invariant failures)";
    }
    std::cout << "\n";
    return failed == 0 ? 0 : 1;
}
