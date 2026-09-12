#pragma once

#include <jsi/jsi.h>
#include <cstring>
#include <memory>
#include <stdexcept>
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

    // Copy a typed array's elements into `out` straight from its backing
    // ArrayBuffer. `obj` must be a typed array whose element size matches T;
    // anything else leaves `out` empty.
    template <typename T>
    inline void readTypedArray(jsi::Runtime& rt, const jsi::Object& obj, std::vector<T>& out) {
        auto bufferVal = obj.getProperty(rt, "buffer");
        if (!bufferVal.isObject() || !bufferVal.getObject(rt).isArrayBuffer(rt)) {
            return;
        }
        auto bytesPerElement = obj.getProperty(rt, "BYTES_PER_ELEMENT");
        if (!bytesPerElement.isNumber() || (size_t) bytesPerElement.getNumber() != sizeof(T)) {
            throw std::runtime_error("typed array element size does not match the expected type");
        }
        auto buffer = bufferVal.getObject(rt).getArrayBuffer(rt);
        const size_t byteOffset = (size_t) obj.getProperty(rt, "byteOffset").asNumber();
        const size_t length = (size_t) obj.getProperty(rt, "length").asNumber();
        if (byteOffset + length * sizeof(T) > buffer.size(rt)) {
            throw std::runtime_error("typed array exceeds its ArrayBuffer");
        }
        out.resize(length);
        if (length > 0) {
            std::memcpy(out.data(), buffer.data(rt) + byteOffset, length * sizeof(T));
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

        readTypedArray(rt, obj, out);
        return out;
    }

    // Read a JS Int32Array or number[] into an int32 vector (token ids,
    // audio codes).
    inline std::vector<int32_t> toInt32Vector(jsi::Runtime& rt, const jsi::Value& v) {
        std::vector<int32_t> out;
        if (!v.isObject()) return out;
        auto obj = v.getObject(rt);

        if (obj.isArray(rt)) {
            auto arr = obj.getArray(rt);
            const size_t n = arr.size(rt);
            out.reserve(n);
            for (size_t i = 0; i < n; i++) {
                out.push_back((int32_t) arr.getValueAtIndex(rt, i).asNumber());
            }
            return out;
        }

        readTypedArray(rt, obj, out);
        return out;
    }

    // Plain JS number[] from a numeric vector (public fields typed as
    // number[] keep this shape; large payloads should use makeFloat32Array).
    template <typename T>
    inline jsi::Array toJsNumberArray(jsi::Runtime& rt, const std::vector<T>& values) {
        jsi::Array arr(rt, values.size());
        for (size_t i = 0; i < values.size(); i++) {
            arr.setValueAtIndex(rt, i, (double) values[i]);
        }
        return arr;
    }

    // Hand a float vector to JS as a Float32Array without one JSI call per
    // element: the vector becomes the backing store of an ArrayBuffer and the
    // Float32Array constructor wraps it. The JS side owns the memory after
    // this returns.
    inline jsi::Value makeFloat32Array(jsi::Runtime& rt, std::vector<float> values) {
        struct VectorBuffer : jsi::MutableBuffer {
            std::vector<float> storage;
            explicit VectorBuffer(std::vector<float> v) : storage(std::move(v)) {}
            size_t size() const override { return storage.size() * sizeof(float); }
            uint8_t* data() override { return reinterpret_cast<uint8_t*>(storage.data()); }
        };
        jsi::ArrayBuffer buffer(rt, std::make_shared<VectorBuffer>(std::move(values)));
        auto ctor = rt.global().getPropertyAsFunction(rt, "Float32Array");
        return ctor.callAsConstructor(rt, buffer);
    }
}
