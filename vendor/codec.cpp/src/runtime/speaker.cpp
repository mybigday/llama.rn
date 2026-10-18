// Generic speaker storage: builtin `spk.<name>.<slot>` tensors in model GGUFs,
// standalone speaker files (codec_speaker_save / codec_speaker_load), and the
// public accessors. Slot layout is defined per architecture through
// codec_model_vtable::speaker_validate.

#include "../codec_internal.h"

#include <cstdio>
#include <cstring>
#include <memory>

namespace {

constexpr uint32_t kSpeakerFormatVersion = 1;
constexpr const char * kSpeakerTensorPrefix = "spk.";

std::string spk_key(const std::string & arch, const char * suffix) {
    return arch + ".spk." + suffix;
}

std::string spk_ref_text_key(const std::string & arch, const std::string & name) {
    return arch + ".spk." + name + ".ref_text";
}

// Speaker names become part of tensor names, so keep them to [a-z0-9_].
bool speaker_name_valid(const std::string & name) {
    if (name.empty() || name.size() > 32) return false;
    for (char c : name) {
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return false;
    }
    return true;
}

// Splits "spk.<name>.<slot>"; false when `tensor_name` is not a speaker tensor.
bool split_speaker_tensor(const char * tensor_name, std::string * name, std::string * slot) {
    const size_t plen = std::strlen(kSpeakerTensorPrefix);
    if (std::strncmp(tensor_name, kSpeakerTensorPrefix, plen) != 0) return false;
    const char * rest = tensor_name + plen;
    const char * dot = std::strchr(rest, '.');
    if (dot == nullptr || dot == rest || dot[1] == '\0') return false;
    name->assign(rest, (size_t) (dot - rest));
    slot->assign(dot + 1);
    return true;
}

codec_speaker * find_speaker(std::vector<codec_speaker> & list, const std::string & name) {
    for (auto & s : list) {
        if (s.name == name) return &s;
    }
    return nullptr;
}

void log_error(const char * fn, const std::string & msg) {
    std::fprintf(stderr, "%s: %s\n", fn, msg.c_str());
}

// Collects speakers from `gf` + tensors in `tensors`. `read` copies a tensor's
// bytes to host (weights may live in a device buffer).
template <typename ReadFn>
bool collect_speakers(
        gguf_context * gf, ggml_context * tensors, const std::string & arch,
        const std::string & encoder_hash, ReadFn read,
        std::vector<codec_speaker> * out, std::string * err) {
    const int64_t n = gguf_get_n_tensors(gf);
    for (int64_t i = 0; i < n; ++i) {
        const char * tname = gguf_get_tensor_name(gf, i);
        std::string name, slot;
        if (tname == nullptr || !split_speaker_tensor(tname, &name, &slot)) continue;
        if (!speaker_name_valid(name)) {
            *err = std::string("invalid speaker name in tensor '") + tname + "'";
            return false;
        }
        ggml_tensor * t = ggml_get_tensor(tensors, tname);
        if (t == nullptr) {
            *err = std::string("missing speaker tensor '") + tname + "'";
            return false;
        }
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) {
            *err = std::string("speaker tensor '") + tname + "' must be F32 or I32";
            return false;
        }
        codec_speaker * spk = find_speaker(*out, name);
        if (spk == nullptr) {
            out->push_back(codec_speaker{});
            spk = &out->back();
            spk->arch = arch;
            spk->encoder_hash = encoder_hash;
            spk->name = name;
            spk->ref_text = codec_read_str_kv(gf, spk_ref_text_key(arch, name).c_str(), "");
        }
        codec_speaker_slot s;
        s.name = slot;
        s.type = t->type;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) s.ne[d] = t->ne[d];
        s.data.resize(ggml_nbytes(t));
        read(t, s.data.data(), s.data.size());
        spk->slots.push_back(std::move(s));
    }
    return true;
}

bool validate_speaker(const codec_model * model, const codec_speaker & spk, std::string * err) {
    if (model->vtable == nullptr || model->vtable->speaker_validate == nullptr) {
        *err = "model architecture takes no speaker";
        return false;
    }
    std::string why;
    if (!model->vtable->speaker_validate(model, &spk, &why)) {
        *err = "speaker '" + spk.name + "': " + why;
        return false;
    }
    return true;
}


// Index of the speaker named by `<arch>.spk.default`; with no key, a lone
// speaker is the default. False when the key names a missing speaker.
bool resolve_default_speaker(
        gguf_context * gf, const std::string & arch, const std::vector<codec_speaker> & speakers,
        int32_t * out, std::string * err) {
    *out = -1;
    const std::string def = codec_read_str_kv(gf, spk_key(arch, "default").c_str(), "");
    if (def.empty()) {
        if (speakers.size() == 1) *out = 0;
        return true;
    }
    for (size_t i = 0; i < speakers.size(); ++i) {
        if (speakers[i].name == def) *out = (int32_t) i;
    }
    if (*out < 0) {
        *err = "default speaker '" + def + "' not found";
        return false;
    }
    return true;
}

}  // namespace

struct codec_speaker_file {
    std::vector<codec_speaker> speakers;
    int32_t default_speaker = -1;
};

namespace {

// Reads a speaker file and checks it (type, architecture, format version,
// encoder hash, every speaker's slots) against `model`.
codec_status open_speaker_file(
        const codec_model * model, const char * path, const char * fn, codec_speaker_file * file) {
    ggml_context * tctx = nullptr;
    gguf_init_params gp = {};
    gp.no_alloc = false;
    gp.ctx = &tctx;
    gguf_context * gf = gguf_init_from_file(path, gp);
    if (gf == nullptr) {
        log_error(fn, std::string("cannot read GGUF ") + path);
        return CODEC_STATUS_IO_ERROR;
    }
    std::unique_ptr<gguf_context, void (*)(gguf_context *)> gf_guard(gf, gguf_free);
    std::unique_ptr<ggml_context, void (*)(ggml_context *)> tctx_guard(tctx, ggml_free);

    auto reject = [&](const std::string & why) {
        log_error(fn, std::string(path) + ": " + why);
        return CODEC_STATUS_INVALID_ARG;
    };

    const std::string type = codec_read_str_kv(gf, "general.type", "");
    if (type != "speaker") {
        return reject("general.type is '" + type + "', expected 'speaker'");
    }
    const std::string arch = codec_read_str_kv(gf, "general.architecture", "");
    if (arch != model->arch_name) {
        return reject("architecture '" + arch + "' does not match model '" + model->arch_name + "'");
    }
    const int32_t version = codec_read_i32_kv(gf, spk_key(arch, "format_version").c_str(), 0);
    if (version != (int32_t) kSpeakerFormatVersion) {
        return reject("unsupported speaker format version " + std::to_string(version));
    }
    const std::string hash = codec_read_str_kv(gf, spk_key(arch, "encoder_hash").c_str(), "");
    if (hash != model->speaker_encoder_hash) {
        return reject("encoder hash '" + hash + "' does not match model '" + model->speaker_encoder_hash +
                      "' (speaker was made by a different encoder)");
    }

    std::string err;
    auto read = [](const ggml_tensor * t, void * dst, size_t n) { std::memcpy(dst, t->data, n); };
    if (!collect_speakers(gf, tctx, arch, hash, read, &file->speakers, &err)) {
        return reject(err);
    }
    if (file->speakers.empty()) {
        return reject("no speakers in file");
    }
    for (const auto & s : file->speakers) {
        if (!validate_speaker(model, s, &err)) return reject(err);
    }
    if (!resolve_default_speaker(gf, arch, file->speakers, &file->default_speaker, &err)) {
        return reject(err);
    }
    return CODEC_STATUS_SUCCESS;
}

}  // namespace

void codec_speaker_add_slot(
        codec_speaker * spk, const char * slot, enum ggml_type type,
        const int64_t * ne, int n_dims, const void * data) {
    codec_speaker_slot s;
    s.name = slot;
    s.type = type;
    for (int d = 0; d < n_dims && d < GGML_MAX_DIMS; ++d) s.ne[d] = ne[d];
    const size_t nbytes = (size_t) s.nelements() * ggml_type_size(type);
    s.data.assign(static_cast<const uint8_t *>(data), static_cast<const uint8_t *>(data) + nbytes);
    spk->slots.push_back(std::move(s));
}

bool codec_model_load_speakers(codec_model * model, std::string * err) {
    const std::string & arch = model->arch_name;
    model->speaker_encoder_hash = codec_read_str_kv(model->gguf, spk_key(arch, "encoder_hash").c_str(),
                                                    model->speaker_encoder_hash.c_str());
    // Speakers registered by the arch init (legacy layouts) predate the hash key.
    for (auto & s : model->speakers) {
        s.arch = arch;
        s.encoder_hash = model->speaker_encoder_hash;
    }

    std::vector<codec_speaker> loaded;
    auto read = [](const ggml_tensor * t, void * dst, size_t n) { ggml_backend_tensor_get(t, dst, 0, n); };
    if (!collect_speakers(model->gguf, model->weights, arch, model->speaker_encoder_hash, read, &loaded, err)) {
        return false;
    }
    for (auto & s : loaded) {
        if (find_speaker(model->speakers, s.name) != nullptr) {
            *err = "duplicate speaker '" + s.name + "'";
            return false;
        }
        model->speakers.push_back(std::move(s));
    }
    for (const auto & s : model->speakers) {
        if (!validate_speaker(model, s, err)) return false;
    }

    return resolve_default_speaker(model->gguf, arch, model->speakers, &model->default_speaker, err);
}

extern "C" {

int32_t codec_model_n_speakers(const struct codec_model * model) {
    return model == nullptr ? 0 : (int32_t) model->speakers.size();
}

const char * codec_model_speaker_name(const struct codec_model * model, int32_t i) {
    if (model == nullptr || i < 0 || i >= (int32_t) model->speakers.size()) return nullptr;
    return model->speakers[(size_t) i].name.c_str();
}

const struct codec_speaker * codec_get_speaker(const struct codec_model * model, const char * name) {
    if (model == nullptr) return nullptr;
    if (name == nullptr) {
        return model->default_speaker >= 0 ? &model->speakers[(size_t) model->default_speaker] : nullptr;
    }
    for (const auto & s : model->speakers) {
        if (s.name == name) return &s;
    }
    return nullptr;
}

bool codec_model_can_speaker_from_audio(const struct codec_model * model) {
    return model != nullptr && model->can_speaker_from_audio &&
           model->vtable != nullptr && model->vtable->speaker_from_audio != nullptr;
}

enum codec_status codec_speaker_from_audio(
        struct codec_context * ctx,
        const struct codec_audio * ref,
        struct codec_speaker ** out) {
    if (ctx == nullptr || ctx->model == nullptr || ref == nullptr || out == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    *out = nullptr;
    codec_context_set_error(ctx, "");
    const codec_model * model = ctx->model;
    if (!codec_model_can_speaker_from_audio(model)) {
        codec_context_set_error(ctx, "model cannot derive a speaker from audio (no speaker encoder bundled)");
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    if (ref->sample_rate <= 0) {
        codec_context_set_error(ctx, "reference audio has no sample rate");
        return CODEC_STATUS_INVALID_ARG;
    }
    std::vector<float> mono;
    std::string err;
    if (!codec_prepare_mono_f32(ref, &mono, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INVALID_ARG;
    }
    auto spk = std::make_unique<codec_speaker>();
    spk->arch = model->arch_name;
    spk->encoder_hash = model->speaker_encoder_hash;
    spk->name = "ref";
    const enum codec_status st = model->vtable->speaker_from_audio(ctx, mono, ref->sample_rate, spk.get());
    if (st != CODEC_STATUS_SUCCESS) return st;
    if (!validate_speaker(model, *spk, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    *out = spk.release();
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_speaker_save(
        const struct codec_model * model,
        const struct codec_speaker * speaker,
        const char * name,
        const char * path) {
    static const char * fn = "codec_speaker_save";
    if (model == nullptr || speaker == nullptr || name == nullptr || path == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (!speaker_name_valid(name)) {
        log_error(fn, std::string("speaker name '") + name + "' must be 1-32 chars of [a-z0-9_]");
        return CODEC_STATUS_INVALID_ARG;
    }
    if (speaker->arch != model->arch_name || speaker->encoder_hash != model->speaker_encoder_hash) {
        log_error(fn, "speaker was produced by a different model architecture or encoder");
        return CODEC_STATUS_INVALID_ARG;
    }

    size_t data_size = 0;
    for (const auto & s : speaker->slots) data_size += s.data.size() + GGML_MEM_ALIGN;
    ggml_init_params ip = {};
    ip.mem_size = data_size + (speaker->slots.size() + 1) * ggml_tensor_overhead();
    ip.no_alloc = false;
    ggml_context * tctx = ggml_init(ip);
    gguf_context * gf = gguf_init_empty();
    if (tctx == nullptr || gf == nullptr) {
        if (tctx != nullptr) ggml_free(tctx);
        if (gf != nullptr) gguf_free(gf);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const std::string & arch = model->arch_name;
    gguf_set_val_str(gf, "general.type", "speaker");
    gguf_set_val_str(gf, "general.architecture", arch.c_str());
    gguf_set_val_u32(gf, spk_key(arch, "format_version").c_str(), kSpeakerFormatVersion);
    gguf_set_val_str(gf, spk_key(arch, "encoder_hash").c_str(), model->speaker_encoder_hash.c_str());
    gguf_set_val_str(gf, spk_key(arch, "default").c_str(), name);
    if (!speaker->ref_text.empty()) {
        gguf_set_val_str(gf, spk_ref_text_key(arch, name).c_str(), speaker->ref_text.c_str());
    }
    for (const auto & s : speaker->slots) {
        ggml_tensor * t = ggml_new_tensor(tctx, s.type, GGML_MAX_DIMS, s.ne);
        const std::string tname = std::string(kSpeakerTensorPrefix) + name + "." + s.name;
        ggml_set_name(t, tname.c_str());
        std::memcpy(t->data, s.data.data(), s.data.size());
        gguf_add_tensor(gf, t);
    }
    const bool ok = gguf_write_to_file(gf, path, /*only_meta=*/false);
    gguf_free(gf);
    ggml_free(tctx);
    if (!ok) {
        log_error(fn, std::string("failed to write ") + path);
        return CODEC_STATUS_IO_ERROR;
    }
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_speaker_file_open(
        const struct codec_model * model,
        const char * path,
        struct codec_speaker_file ** out) {
    if (model == nullptr || path == nullptr || out == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    *out = nullptr;
    std::unique_ptr<codec_speaker_file> file(new (std::nothrow) codec_speaker_file());
    if (file == nullptr) return CODEC_STATUS_INTERNAL_ERROR;
    const codec_status st = open_speaker_file(model, path, "codec_speaker_file_open", file.get());
    if (st == CODEC_STATUS_SUCCESS) *out = file.release();
    return st;
}

int32_t codec_speaker_file_n_speakers(const struct codec_speaker_file * file) {
    return file == nullptr ? 0 : (int32_t) file->speakers.size();
}

const char * codec_speaker_file_speaker_name(const struct codec_speaker_file * file, int32_t i) {
    if (file == nullptr || i < 0 || i >= (int32_t) file->speakers.size()) return nullptr;
    return file->speakers[(size_t) i].name.c_str();
}

const struct codec_speaker * codec_speaker_file_get_speaker(
        const struct codec_speaker_file * file, const char * name) {
    if (file == nullptr) return nullptr;
    if (name == nullptr) {
        return file->default_speaker >= 0 ? &file->speakers[(size_t) file->default_speaker] : nullptr;
    }
    for (const auto & s : file->speakers) {
        if (s.name == name) return &s;
    }
    return nullptr;
}

void codec_speaker_file_free(struct codec_speaker_file * file) {
    delete file;
}

enum codec_status codec_speaker_load(
        const struct codec_model * model,
        const char * path,
        const char * name,
        struct codec_speaker ** out) {
    static const char * fn = "codec_speaker_load";
    if (model == nullptr || path == nullptr || out == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    *out = nullptr;
    codec_speaker_file file;
    const codec_status st = open_speaker_file(model, path, fn, &file);
    if (st != CODEC_STATUS_SUCCESS) return st;
    const codec_speaker * found = codec_speaker_file_get_speaker(&file, name);
    if (found == nullptr) {
        std::string avail;
        for (const auto & s : file.speakers) avail += (avail.empty() ? "" : ", ") + s.name;
        log_error(fn, std::string(path) + ": speaker '" + (name != nullptr ? name : "<default>") +
                      "' not found (available: " + avail + ")");
        return CODEC_STATUS_INVALID_ARG;
    }
    *out = new (std::nothrow) codec_speaker(*found);
    return *out != nullptr ? CODEC_STATUS_SUCCESS : CODEC_STATUS_INTERNAL_ERROR;
}

enum codec_status codec_speaker_set_ref_text(struct codec_speaker * speaker, const char * text) {
    if (speaker == nullptr) return CODEC_STATUS_INVALID_ARG;
    speaker->ref_text = text != nullptr ? text : "";
    return CODEC_STATUS_SUCCESS;
}

bool codec_speaker_get_slot(
        const struct codec_speaker * speaker,
        const char * slot,
        enum codec_speaker_slot_type * type,
        int64_t ne[4],
        const void ** data) {
    if (speaker == nullptr || slot == nullptr) return false;
    const codec_speaker_slot * s = speaker->slot(slot);
    if (s == nullptr) return false;
    if (type != nullptr) *type = s->type == GGML_TYPE_I32 ? CODEC_SPEAKER_SLOT_I32 : CODEC_SPEAKER_SLOT_F32;
    if (ne != nullptr) {
        for (int d = 0; d < 4; ++d) ne[d] = s->ne[d];
    }
    if (data != nullptr) *data = s->data.data();
    return true;
}

void codec_speaker_free(struct codec_speaker * speaker) {
    delete speaker;
}

const char * codec_speaker_name(const struct codec_speaker * speaker) {
    return speaker == nullptr ? nullptr : speaker->name.c_str();
}

const char * codec_speaker_ref_text(const struct codec_speaker * speaker) {
    if (speaker == nullptr || speaker->ref_text.empty()) return nullptr;
    return speaker->ref_text.c_str();
}

}  // extern "C"
