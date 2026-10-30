#ifndef CODEC_SPEAKER_H
#define CODEC_SPEAKER_H

#include "codec.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =====================================================================
// codec_speaker — voice conditioning that lives next to a codec in the
// same GGUF but is not part of encode / decode, like codec_lm.  It
// borrows a `codec_model` loaded by codec.h; a speaker is passed to
// decode through codec_decode_params.speaker and to an audio LM through
// codec_audio_lm.h.  See docs/codec_speaker.md.
// =====================================================================

// A speaker is the reference conditioning a model derives from a voice
// sample: a set of named tensors ("slots") whose layout each architecture
// defines. Builtin speakers live in the model GGUF as `spk.<name>.<slot>`
// tensors, with `<arch>.spk.default` naming the default one.
//
// Speaker files written by codec_speaker_save carry general.type="speaker",
// general.architecture, `<arch>.spk.format_version` and
// `<arch>.spk.encoder_hash`; codec_speaker_load rejects a file whose
// architecture, format version or encoder hash differs from the model's.

// Builtin speakers. Owned by the model (valid for its lifetime; do not free).
int32_t codec_model_n_speakers(const struct codec_model * model);
const char * codec_model_speaker_name(const struct codec_model * model, int32_t i);
// name = NULL returns the default speaker. Returns NULL when not found.
const struct codec_speaker * codec_get_speaker(const struct codec_model * model, const char * name);

// True when codec_speaker_from_audio is available for this model.
bool codec_model_can_speaker_from_audio(const struct codec_model * model);
// Derive a speaker from reference audio (any sample rate; mono or
// interleaved multi-channel is downmixed). Caller frees with codec_speaker_free.
enum codec_status codec_speaker_from_audio(
    struct codec_context * ctx,
    const struct codec_audio * ref,
    struct codec_speaker ** out);

enum codec_status codec_speaker_save(
    const struct codec_model * model,
    const struct codec_speaker * speaker,
    const char * name,
    const char * path);
// name = NULL loads the file's default speaker. Caller frees with codec_speaker_free.
enum codec_status codec_speaker_load(
    const struct codec_model * model,
    const char * path,
    const char * name,
    struct codec_speaker ** out);
void codec_speaker_free(struct codec_speaker * speaker);

// Speaker files may hold several speakers. Opening checks the file and every
// speaker in it against the model, like codec_speaker_load. Speakers returned
// by codec_speaker_file_get_speaker are owned by the file handle.
struct codec_speaker_file;
enum codec_status codec_speaker_file_open(
    const struct codec_model * model,
    const char * path,
    struct codec_speaker_file ** out);
int32_t codec_speaker_file_n_speakers(const struct codec_speaker_file * file);
const char * codec_speaker_file_speaker_name(const struct codec_speaker_file * file, int32_t i);
// name = NULL returns the file's default speaker. Returns NULL when not found.
const struct codec_speaker * codec_speaker_file_get_speaker(
    const struct codec_speaker_file * file, const char * name);
void codec_speaker_file_free(struct codec_speaker_file * file);

const char * codec_speaker_name(const struct codec_speaker * speaker);
// Transcript of the reference audio, or NULL when unknown.
const char * codec_speaker_ref_text(const struct codec_speaker * speaker);
// Attach the transcript of the reference audio (saved with the speaker).
enum codec_status codec_speaker_set_ref_text(struct codec_speaker * speaker, const char * text);

enum codec_speaker_slot_type {
    CODEC_SPEAKER_SLOT_F32 = 0,
    CODEC_SPEAKER_SLOT_I32 = 1,
};

// Read one slot (model-specific name, e.g. "lm_cond"). `ne` receives 4 dims
// (unused = 1, ggml order: ne[0] fastest). `*data` stays valid for the
// speaker's lifetime. Returns false when the slot is absent.
bool codec_speaker_get_slot(
    const struct codec_speaker * speaker,
    const char * slot,
    enum codec_speaker_slot_type * type,
    int64_t ne[4],
    const void ** data);

#ifdef __cplusplus
}
#endif

#endif  // CODEC_SPEAKER_H
