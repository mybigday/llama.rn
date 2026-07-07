// UTF-8 handling tests (host-only: no model, no GPU).
//
// Covers the utf8 helpers and utf8_stream_gate directly, then the consumer
// seam: parseChatOutput driven with a production-shaped Gemma4 tool-call
// parser, with text entering through the gate exactly as the generation
// loops feed it.
//
// Convention: malformed bytes enter only via utf8_gate.feed()/finish(), never
// by assigning generated_text directly - the production contract, and what
// keeps the parseChatOutput asserts satisfied in debug builds.

#include <iostream>
#include <string>
#include <vector>

#include "rn-llama.h"
#include "rn-completion.h"
#include "rn-slot.h"
#include "chat.h"
#include "chat-peg-parser.h"
#include "nlohmann/json.hpp"

using namespace rnllama;

// Test result tracking (same shape as simple_test.cpp)
struct TestResults {
    int total_tests = 0;
    int passed_tests = 0;

    void run_test(const std::string& name, bool result) {
        total_tests++;
        std::cout << "TEST: " << name << " ... ";
        if (result) {
            std::cout << "PASSED" << std::endl;
            passed_tests++;
        } else {
            std::cout << "FAILED" << std::endl;
        }
    }

    void print_summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << total_tests << std::endl;
        std::cout << "Passed: " << passed_tests << std::endl;
        std::cout << "Failed: " << (total_tests - passed_tests) << std::endl;
    }
};

// Replacement character U+FFFD as UTF-8
static const std::string REPLACEMENT = "\xEF\xBF\xBD";

static bool repro_ends_well(const std::string & s) { return utf8_is_well_formed(s); }

// ---------------------------------------------------------------------------
// Helper unit tests
// ---------------------------------------------------------------------------

// utf8_sanitize: every ill-formed class nlohmann::json rejects (stray bytes,
// overlongs, surrogates, > U+10FFFF, truncations) must come out replaced, and
// valid boundary values must pass through untouched.
static bool test_sanitize_edge_cases() {
    struct Case { std::string in; std::string out; const char * name; };
    const std::string & R = REPLACEMENT;
    const Case cases[] = {
        {"\xC0",                 R,             "lone invalid lead C0"},
        {"\xFF",                 R,             "invalid byte FF"},
        {"a\x9F" "b",            "a" + R + "b", "stray continuation byte"},
        {"\xC0\xAF",             R + R,         "overlong 2-byte (C0 AF)"},
        {"\xE0\x9F\xBF",         R + R + R,     "overlong 3-byte (E0 9F BF)"},
        {"\xED\xA0\x80",         R + R + R,     "UTF-16 surrogate (ED A0 80)"},
        {"\xF4\x90\x80\x80",     R + R + R + R, "above U+10FFFF (F4 90 80 80)"},
        {"\xE2\x82",             R,             "truncated tail -> single replacement"},
        {"\xE0\xA0" "A",         R + "A",       "bad continuation, following ASCII kept"},
        // valid boundary values must be byte-identical
        {"\xC2\x80",             "\xC2\x80",         "U+0080"},
        {"\xED\x9F\xBF",         "\xED\x9F\xBF",     "U+D7FF (last before surrogates)"},
        {"\xEE\x80\x80",         "\xEE\x80\x80",     "U+E000 (first after surrogates)"},
        {"\xF4\x8F\xBF\xBF",     "\xF4\x8F\xBF\xBF", "U+10FFFF"},
        {"Z\xC3\xBCrich \xF0\x9F\x8C\x8D", "Z\xC3\xBCrich \xF0\x9F\x8C\x8D", "valid mixed text"},
    };
    for (const auto & c : cases) {
        auto got = utf8_sanitize(c.in);
        if (got != c.out) {
            std::cout << "[" << c.name << ": unexpected output] ";
            return false;
        }
        try {
            (void) nlohmann::json(got).dump(); // must be strict-dumpable
        } catch (const std::exception & e) {
            std::cout << "[" << c.name << " not dumpable: " << e.what() << "] ";
            return false;
        }
    }
    return true;
}

// Only a well-formed incomplete sequence is completable and held back;
// invalid leads, stray and out-of-range continuations are dead on arrival.
static bool test_incomplete_suffix_edge_cases() {
    if (utf8_incomplete_suffix_length("") != 0) return false;
    // cut-off valid sequences are completable
    if (utf8_incomplete_suffix_length("Hi\xC3") != 1) return false;
    if (utf8_incomplete_suffix_length("Hi\xF0\x9F\x8C") != 3) return false;
    if (utf8_incomplete_suffix_length("Hi\xE0\xA0") != 2) return false;
    // complete sequences leave nothing pending
    if (utf8_incomplete_suffix_length("Hi\xC3\xA9") != 0) return false;
    // dead tails are not completable: stray continuation, invalid leads,
    // out-of-range continuation (E0 9F would be overlong), too many
    // continuations after a shorter lead
    if (utf8_incomplete_suffix_length("Hi\x9F") != 0) return false;
    if (utf8_incomplete_suffix_length("Hi\xC0") != 0) return false;
    if (utf8_incomplete_suffix_length("Hi\xF5") != 0) return false;
    if (utf8_incomplete_suffix_length("Hi\xE0\x9F") != 0) return false;
    if (utf8_incomplete_suffix_length("Hi\xC3\xA9\x9F") != 0) return false;
    return true;
}

// utf8_is_well_formed mirrors utf8_sanitize's strictness without copying.
static bool test_is_well_formed() {
    if (!utf8_is_well_formed("")) return false;
    if (!utf8_is_well_formed("Z\xC3\xBCrich \xF0\x9F\x8C\x8D")) return false;
    if (utf8_is_well_formed("a\x9F" "b")) return false;      // stray continuation
    if (utf8_is_well_formed("Hi\xF0\x9F\x8C")) return false; // truncated tail
    if (utf8_is_well_formed("\xED\xA0\x80")) return false;   // surrogate
    if (utf8_is_well_formed("\xC0\xAF")) return false;        // overlong
    return true;
}

// ---------------------------------------------------------------------------
// Gate unit tests
// ---------------------------------------------------------------------------

// A character split across four 1-byte pieces must be held back and emitted
// whole; the concatenated deltas equal the logical text.
static bool test_gate_reassembles_split_character() {
    utf8_stream_gate gate;
    std::string out;
    out += gate.feed("Hello ");
    if (out != "Hello ") return false;
    out += gate.feed("\xF0");
    out += gate.feed("\x9F");
    out += gate.feed("\x8C");
    if (out != "Hello ") return false;      // tail held, nothing partial emitted
    if (!gate.has_pending()) return false;
    out += gate.feed("\x8D");
    out += gate.feed("!");
    if (gate.has_pending()) return false;
    return out == "Hello \xF0\x9F\x8C\x8D!" && gate.finish().empty();
}

// A stray continuation byte is dead on arrival: replaced immediately, not
// buffered.
static bool test_gate_replaces_stray_byte_immediately() {
    utf8_stream_gate gate;
    std::string out = gate.feed("a\x9F" "b");
    return out == "a" + REPLACEMENT + "b" && !gate.has_pending();
}

// A dangling tail at end of generation becomes exactly one U+FFFD via
// finish(); the gate is reusable afterwards.
static bool test_gate_finish_flushes_tail() {
    utf8_stream_gate gate;
    if (gate.feed("Hi\xF0\x9F") != "Hi") return false;
    if (gate.finish() != REPLACEMENT) return false;
    if (gate.has_pending()) return false;
    return gate.feed("ok") == "ok";
}

// Dead bytes are replaced as soon as they are known dead: an invalid lead
// immediately, a valid lead once the next byte rules out completion.
static bool test_gate_replaces_dead_bytes_immediately() {
    utf8_stream_gate gate;
    // invalid lead: dead on arrival
    if (gate.feed("Hi\xC0") != "Hi" + REPLACEMENT) return false;
    if (gate.has_pending()) return false;
    // valid lead held...
    if (gate.feed("\xF0") != "") return false;
    if (!gate.has_pending()) return false;
    // ...but flushed as dead the moment 'A' makes completion impossible;
    // 'A' itself must not be delayed
    if (gate.feed("A") != REPLACEMENT + "A") return false;
    return !gate.has_pending() && gate.finish().empty();
}

// Property: for any split into pieces, concat(feeds) + finish() equals
// utf8_sanitize(whole stream). Checked over every 3-way split.
static bool test_gate_equivalent_to_whole_sanitize() {
    const std::string streams[] = {
        "Z\xC3\xBCri\x9F ok",                    // stray byte inside valid text
        "Hello \xF0\x9F\x8C\x8D world",          // 4-byte char mid-text
        "\xE0\xA0\x80\xED\xA0\x80\xC2\xA9",      // valid + surrogate + valid
        "abc\xF0\x9F\x8C",                       // dangling tail
        "\xC0\xAF\xC0",                          // overlong + dangling invalid lead
    };
    for (const auto & s : streams) {
        const std::string expected = utf8_sanitize(s);
        for (size_t i = 0; i <= s.size(); ++i) {
            for (size_t j = i; j <= s.size(); ++j) {
                utf8_stream_gate gate;
                std::string out;
                out += gate.feed(s.substr(0, i));
                out += gate.feed(s.substr(i, j - i));
                out += gate.feed(s.substr(j));
                out += gate.finish();
                if (out != expected) {
                    std::cout << "[split " << i << "," << j << " diverged] ";
                    return false;
                }
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Consumer seam tests: parseChatOutput fed through the gate, exactly as the
// generation loops feed it.
// ---------------------------------------------------------------------------

// The tool-call parser common_chat_params_init_gemma4() builds
// (cpp/common/chat.cpp), for a single tool get_weather(city).
static std::string build_gemma4_tool_parser() {
    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start = p.rule("start", p.optional(p.literal("<|turn>model\n")));

        p.rule("thought", p.content(p.literal("<|channel>thought") + p.space() + p.until("<channel|>") + p.literal("<channel|>")));

        auto consume_empty_channels = p.gbnf(p.zero_or_more(p.literal("<|channel>") + p.negate(p.literal("thought"))), "");
        auto thought = (p.peek(p.literal("<|channel>")) + consume_empty_channels + p.ref("thought")) | p.negate(p.literal("<|channel>"));

        p.rule("gemma4-string-content", p.until("<|\"|>"));
        p.rule("gemma4-string", p.literal("<|\"|>") + p.ref("gemma4-string-content") + p.literal("<|\"|>"));
        p.rule("gemma4-bool", p.json_bool());
        p.rule("gemma4-null", p.json_null());
        p.rule("gemma4-number", p.json_number());
        p.rule("gemma4-dict-key", p.rule("gemma4-dict-key-name", p.chars("[^:}]", 1, -1)) + p.literal(":"));
        p.rule("gemma4-dict-kv", p.ref("gemma4-dict-key") + p.space() + p.ref("gemma4-value"));
        p.rule("gemma4-dict", [&]() {
            auto ws = p.space();
            auto member = p.ref("gemma4-dict-kv");
            auto members = p.sequence({member, p.zero_or_more(p.sequence({p.literal(","), ws, member}))});
            return p.sequence({
                p.literal("{"), ws,
                p.choice({p.literal("}"), p.sequence({members, ws, p.literal("}")})})
            });
        });
        p.rule("gemma4-array", [&]() {
            auto ws = p.space();
            auto value = p.ref("gemma4-value");
            auto elements = p.sequence({value, p.zero_or_more(p.sequence({p.literal(","), ws, value}))});
            return p.sequence({
                p.literal("["), ws,
                p.choice({p.literal("]"), p.sequence({elements, ws, p.literal("]")})})
            });
        });
        p.rule("gemma4-value", [&]() {
            return p.choice({
                p.ref("gemma4-string"), p.ref("gemma4-dict"), p.ref("gemma4-array"),
                p.ref("gemma4-number"), p.ref("gemma4-bool"), p.ref("gemma4-null")
            });
        });

        auto tool_choice = p.choice();
        tool_choice |= p.rule("tool-get_weather", p.tool(p.sequence({
            p.tool_open(p.tool_name(p.literal("get_weather")) + p.peek(p.literal("{"))),
            p.tool_args(p.ref("gemma4-dict")),
        })));

        auto tool_call = p.trigger_rule("tool-call", p.repeat(
            "<|tool_call>call:" + tool_choice + "<tool_call|>",
            /* min = */ 0,
            /* max = */ 1
        ));

        auto scan_to_toolcall = p.rule("scan-to-toolcall", p.until("<|tool_call>"));
        auto content = p.rule("content", p.content(p.until_one_of({"<|channel>", "<channel|>", "<|tool_call>"})));
        auto message = p.rule("message", thought + content);
        return start + p.zero_or_more(message) + scan_to_toolcall + tool_call;
    });

    return parser.save();
}

// Assistant output with a stray 0x9F byte inside a tool-call argument.
static std::string gemma4_output_with_invalid_byte() {
    return std::string("Checking the weather.") +
        "<|tool_call>call:get_weather{city:<|\"|>Z\xC3\xBCri\x9F ok<|\"|>}<tool_call|>";
}

// Drive parseChatOutput with `raw` entering as generation does: piece by
// piece through the gate, flushed when generation ends.
static completion_chat_output run_completion_parse(const std::string & raw, int format,
                                                   const std::string & parser, bool is_partial) {
    llama_rn_context_completion completion(nullptr);
    completion.current_chat_format = format;
    completion.current_reasoning_format = COMMON_REASONING_FORMAT_NONE;
    completion.current_chat_parser = parser;
    completion.prefill_text = "";
    for (size_t i = 0; i < raw.size(); i += 3) { // arbitrary small pieces
        completion.generated_text += completion.utf8_gate.feed(raw.substr(i, 3));
    }
    if (!is_partial) {
        completion.generated_text += completion.utf8_gate.finish();
    }
    return completion.parseChatOutput(is_partial);
}

// A tool call with an invalid byte in its arguments must survive the final
// parse: no throw, arguments valid JSON with U+FFFD, content preserved.
static bool test_toolcall_invalid_utf8_final(const std::string & parser) {
    try {
        auto out = run_completion_parse(gemma4_output_with_invalid_byte(),
                                        COMMON_CHAT_FORMAT_PEG_GEMMA4, parser, false);
        if (out.tool_calls.size() != 1) {
            std::cout << "[expected 1 tool call, got " << out.tool_calls.size() << "] ";
            return false;
        }
        auto args = nlohmann::json::parse(out.tool_calls[0].arguments);
        std::string city = args.at("city").get<std::string>();
        if (city != "Z\xC3\xBCri" + REPLACEMENT + " ok") {
            std::cout << "[unexpected city arg: " << args.at("city").dump() << "] ";
            return false;
        }
        if (out.content.find("Checking the weather.") == std::string::npos) {
            std::cout << "[content lost: '" << out.content << "'] ";
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        std::cout << "[threw: " << e.what() << "] ";
        return false;
    }
}

// Same input mid-stream (is_partial=true): must not throw.
static bool test_toolcall_invalid_utf8_partial(const std::string & parser) {
    try {
        run_completion_parse(gemma4_output_with_invalid_byte(),
                             COMMON_CHAT_FORMAT_PEG_GEMMA4, parser, true);
        return true;
    } catch (const std::exception & e) {
        std::cout << "[threw: " << e.what() << "] ";
        return false;
    }
}

// Valid multi-byte text must arrive at the parser byte-identical.
static bool test_toolcall_valid_utf8_passthrough(const std::string & parser) {
    try {
        std::string text = std::string("Done.") +
            "<|tool_call>call:get_weather{city:<|\"|>Z\xC3\xBCrich \xF0\x9F\x8C\x8D<|\"|>}<tool_call|>";
        auto out = run_completion_parse(text, COMMON_CHAT_FORMAT_PEG_GEMMA4, parser, false);
        if (out.tool_calls.size() != 1) {
            std::cout << "[expected 1 tool call, got " << out.tool_calls.size() << "] ";
            return false;
        }
        auto args = nlohmann::json::parse(out.tool_calls[0].arguments);
        return args.at("city").get<std::string>() == "Z\xC3\xBCrich \xF0\x9F\x8C\x8D";
    } catch (const std::exception & e) {
        std::cout << "[threw: " << e.what() << "] ";
        return false;
    }
}

// A split character stays in the gate during streaming: partial parses never
// see U+FFFD for a character still in flight.
static bool test_partial_holds_back_split_multibyte() {
    try {
        auto out = run_completion_parse("Hello \xF0\x9F\x8C", COMMON_CHAT_FORMAT_CONTENT_ONLY, "", true);
        if (out.content != "Hello ") {
            std::cout << "[expected 'Hello ', got '" << out.content << "'] ";
            return false;
        }
        return out.content.find(REPLACEMENT) == std::string::npos;
    } catch (const std::exception & e) {
        std::cout << "[threw: " << e.what() << "] ";
        return false;
    }
}

// At end of generation the flushed tail becomes a single U+FFFD in the final
// content.
static bool test_final_replaces_incomplete_tail() {
    try {
        auto out = run_completion_parse("Hello \xF0\x9F\x8C", COMMON_CHAT_FORMAT_CONTENT_ONLY, "", false);
        if (out.content != "Hello " + REPLACEMENT) {
            std::cout << "[expected 'Hello <U+FFFD>', got '" << out.content << "'] ";
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        std::cout << "[threw: " << e.what() << "] ";
        return false;
    }
}

// A stray byte in plain content is replaced; parseChatOutput results are
// safe for strict JSON serialization.
static bool test_invalid_byte_in_plain_content() {
    try {
        auto out = run_completion_parse("byte:\x9F:done", COMMON_CHAT_FORMAT_CONTENT_ONLY, "", false);
        if (out.content != "byte:" + REPLACEMENT + ":done") {
            std::cout << "[got '" << out.content << "'] ";
            return false;
        }
        (void) nlohmann::json(out.content).dump();
        (void) nlohmann::json(out.accumulated_text).dump();
        return true;
    } catch (const std::exception & e) {
        std::cout << "[threw: " << e.what() << "] ";
        return false;
    }
}

// Real-world fixture: assistant reply captured from qwen3.5-2B on a Galaxy
// S23 (Adreno 740, all layers on GPU, greedy), where generation ended
// mid-emoji. Serializing the raw reply aborted the process with json
// type_error.316 ("incomplete UTF-8 string; last byte: 0x9F").
// "!!" + U+1F482 x7 + a truncated 4-byte tail (F0 9F).
static const std::string QWEN35_S23_FIXTURE =
    "\x21\x21"
    "\xF0\x9F\x92\x82\xF0\x9F\x92\x82\xF0\x9F\x92\x82\xF0\x9F\x92\x82"
    "\xF0\x9F\x92\x82\xF0\x9F\x92\x82\xF0\x9F\x92\x82"
    "\xF0\x9F";

// The raw fixture must still trigger the original failure, and the gate must
// neutralize it.
static bool test_qwen35_s23_device_fixture() {
    // 1. original mechanism: strict serialization of the raw reply throws
    bool threw = false;
    try {
        (void) nlohmann::json(QWEN35_S23_FIXTURE).dump();
    } catch (const nlohmann::json::type_error &) {
        threw = true;
    }
    if (!threw) {
        std::cout << "[raw fixture no longer throws - fixture stale?] ";
        return false;
    }

    const std::string expected = "\x21\x21"
        "\xF0\x9F\x92\x82\xF0\x9F\x92\x82\xF0\x9F\x92\x82\xF0\x9F\x92\x82"
        "\xF0\x9F\x92\x82\xF0\x9F\x92\x82\xF0\x9F\x92\x82" + REPLACEMENT;

    // 2. through the gate, for every split of the byte stream into pieces
    for (size_t i = 0; i <= QWEN35_S23_FIXTURE.size(); ++i) {
        utf8_stream_gate gate;
        std::string out;
        out += gate.feed(QWEN35_S23_FIXTURE.substr(0, i));
        out += gate.feed(QWEN35_S23_FIXTURE.substr(i));
        // mid-generation: the cut-off tail is held back, never emitted broken
        if (!repro_ends_well(out)) { std::cout << "[split " << i << " emitted broken delta] "; return false; }
        out += gate.finish();
        if (out != expected) { std::cout << "[split " << i << " wrong final text] "; return false; }
        (void) nlohmann::json(out).dump(); // must not throw
    }

    // 3. through the production parse seam
    try {
        auto parsed = run_completion_parse(QWEN35_S23_FIXTURE, COMMON_CHAT_FORMAT_CONTENT_ONLY, "", false);
        if (parsed.content != expected) { std::cout << "[parse content mismatch] "; return false; }
        (void) nlohmann::json(parsed.content).dump();
    } catch (const std::exception & e) {
        std::cout << "[threw: " << e.what() << "] ";
        return false;
    }
    return true;
}

// The slot (parallel decoding) path: same gate, its own parseChatOutput.
static bool test_slot_toolcall_invalid_utf8(const std::string & parser) {
    try {
        llama_rn_slot slot;
        slot.current_chat_format = COMMON_CHAT_FORMAT_PEG_GEMMA4;
        slot.current_reasoning_format = COMMON_REASONING_FORMAT_NONE;
        slot.current_chat_parser = parser;
        slot.prefill_text = "";
        const std::string raw = gemma4_output_with_invalid_byte();
        for (size_t i = 0; i < raw.size(); i += 3) {
            slot.generated_text += slot.utf8_gate.feed(raw.substr(i, 3));
        }
        (void) slot.parseChatOutput(true);
        slot.generated_text += slot.utf8_gate.finish();
        auto out = slot.parseChatOutput(false);
        return out.tool_calls.size() == 1;
    } catch (const std::exception & e) {
        std::cout << "[threw: " << e.what() << "] ";
        return false;
    }
}

int main() {
    std::cout << "=== chat parse UTF-8 robustness tests ===" << std::endl;

    const std::string parser = build_gemma4_tool_parser();

    TestResults results;
    // helpers
    results.run_test("utf8_sanitize edge cases (overlong/surrogate/range)", test_sanitize_edge_cases());
    results.run_test("utf8_incomplete_suffix_length edge cases", test_incomplete_suffix_edge_cases());
    results.run_test("utf8_is_well_formed edge cases", test_is_well_formed());
    // gate
    results.run_test("gate reassembles split character", test_gate_reassembles_split_character());
    results.run_test("gate replaces stray byte immediately", test_gate_replaces_stray_byte_immediately());
    results.run_test("gate finish() flushes dangling tail", test_gate_finish_flushes_tail());
    results.run_test("gate replaces dead bytes immediately", test_gate_replaces_dead_bytes_immediately());
    results.run_test("gate == whole-stream sanitize for all splits", test_gate_equivalent_to_whole_sanitize());
    // consumer seam
    results.run_test("toolcall with invalid UTF-8 byte (final parse)", test_toolcall_invalid_utf8_final(parser));
    results.run_test("toolcall with invalid UTF-8 byte (partial/streaming parse)", test_toolcall_invalid_utf8_partial(parser));
    results.run_test("toolcall with valid multi-byte UTF-8 passes through", test_toolcall_valid_utf8_passthrough(parser));
    results.run_test("partial parse holds back split multi-byte char", test_partial_holds_back_split_multibyte());
    results.run_test("final parse replaces incomplete trailing sequence", test_final_replaces_incomplete_tail());
    results.run_test("invalid byte in plain content replaced", test_invalid_byte_in_plain_content());
    results.run_test("real S23/qwen35 device fixture (truncated emoji tail)", test_qwen35_s23_device_fixture());
    results.run_test("slot parseChatOutput with invalid UTF-8 byte", test_slot_toolcall_invalid_utf8(parser));

    results.print_summary();
    return results.passed_tests == results.total_tests ? 0 : 1;
}
