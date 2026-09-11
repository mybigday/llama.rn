#pragma once

#include <jsi/jsi.h>
#include <cstring>
#include <string>
#include <vector>

#include "JSINativeHeaders.h"

using namespace facebook;

namespace rnllama_jsi {

    using json = nlohmann::ordered_json;

    // Convert a JS value into nlohmann json without a JSON.stringify/parse
    // round trip. Must run on the JS thread (it touches the runtime); the
    // returned json is a plain value and can be captured into worker tasks.
    //
    // Semantics follow JSON.stringify: undefined object properties are
    // dropped, functions become null, and numbers stay as double.
    inline json toJson(jsi::Runtime& rt, const jsi::Value& v) {
        if (v.isUndefined() || v.isNull()) return nullptr;
        if (v.isBool())   return v.getBool();
        if (v.isNumber()) return v.getNumber();
        if (v.isString()) return v.getString(rt).utf8(rt);
        if (!v.isObject()) return nullptr;

        auto obj = v.getObject(rt);
        if (obj.isFunction(rt)) return nullptr;

        if (obj.isArray(rt)) {
            auto arr = obj.getArray(rt);
            json out = json::array();
            const size_t n = arr.size(rt);
            for (size_t i = 0; i < n; i++) {
                out.push_back(toJson(rt, arr.getValueAtIndex(rt, i)));
            }
            return out;
        }

        json out = json::object();
        auto names = obj.getPropertyNames(rt);
        const size_t n = names.size(rt);
        for (size_t i = 0; i < n; i++) {
            auto key = names.getValueAtIndex(rt, i).getString(rt).utf8(rt);
            auto val = obj.getProperty(rt, key.c_str());
            if (val.isUndefined()) continue;
            out[key] = toJson(rt, val);
        }
        return out;
    }

    // Convert nlohmann json back into a JS value. Must run on the JS thread.
    inline jsi::Value fromJson(jsi::Runtime& rt, const json& j) {
        switch (j.type()) {
            case json::value_t::null:
                return jsi::Value::null();
            case json::value_t::boolean:
                return jsi::Value(j.get<bool>());
            case json::value_t::number_integer:
            case json::value_t::number_unsigned:
            case json::value_t::number_float:
                return jsi::Value(j.get<double>());
            case json::value_t::string:
                return jsi::String::createFromUtf8(rt, j.get_ref<const std::string&>());
            case json::value_t::array: {
                jsi::Array arr(rt, j.size());
                size_t i = 0;
                for (const auto& el : j) {
                    arr.setValueAtIndex(rt, i++, fromJson(rt, el));
                }
                return arr;
            }
            case json::value_t::object: {
                jsi::Object obj(rt);
                for (const auto& [key, val] : j.items()) {
                    obj.setProperty(rt, key.c_str(), fromJson(rt, val));
                }
                return obj;
            }
            default:
                return jsi::Value::undefined();
        }
    }

    // Read a JS Float32Array or number[] into a float vector. Typed arrays
    // are copied straight out of the backing ArrayBuffer, which avoids one
    // JSI call per element for large PCM buffers.
    inline std::vector<float> toFloatVector(jsi::Runtime& rt, const jsi::Value& v) {
        std::vector<float> out;
        if (!v.isObject()) return out;
        auto obj = v.getObject(rt);

        if (obj.isArray(rt)) {
            auto arr = obj.getArray(rt);
            const size_t n = arr.size(rt);
            out.reserve(n);
            for (size_t i = 0; i < n; i++) {
                out.push_back((float) arr.getValueAtIndex(rt, i).asNumber());
            }
            return out;
        }

        // Typed array: { buffer: ArrayBuffer, byteOffset, length }
        auto bufferVal = obj.getProperty(rt, "buffer");
        if (!bufferVal.isObject() || !bufferVal.getObject(rt).isArrayBuffer(rt)) {
            return out;
        }
        auto buffer = bufferVal.getObject(rt).getArrayBuffer(rt);
        const size_t byteOffset = (size_t) obj.getProperty(rt, "byteOffset").asNumber();
        const size_t length = (size_t) obj.getProperty(rt, "length").asNumber();
        if (byteOffset + length * sizeof(float) > buffer.size(rt)) {
            throw std::runtime_error("typed array exceeds its ArrayBuffer");
        }
        out.resize(length);
        if (length > 0) {
            std::memcpy(out.data(), buffer.data(rt) + byteOffset, length * sizeof(float));
        }
        return out;
    }
}
