# vendor

Vendored copies of the native dependencies, in their upstream directory layout.
There are no git submodules: `git clone` gives you everything the builds need,
and every build (CocoaPods, the CMake projects, the tests) compiles these
files in place.

| Directory | Upstream | What is vendored |
| --- | --- | --- |
| `llama.cpp/` | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `include/`, `src/`, `ggml/{include,src}` (CPU, Metal, BLAS, OpenCL, Hexagon backends), the parts of `common/` and `tools/mtmd/` we use, `vendor/{nlohmann,hash,miniaudio,stb}` |
| `codec.cpp/` | [mybigday/codec.cpp](https://github.com/mybigday/codec.cpp) | `include/`, `src/`, `common/`, `examples/utils/` |
| `OpenCL-Headers/` | [KhronosGroup/OpenCL-Headers](https://github.com/KhronosGroup/OpenCL-Headers) | `CL/` |
| `OpenCL-ICD-Loader/` | [KhronosGroup/OpenCL-ICD-Loader](https://github.com/KhronosGroup/OpenCL-ICD-Loader) | the CMake project that builds the Android `libOpenCL.so` link stub |

`VERSIONS` pins the upstream ref and resolved commit of each tree. The exact
file list lives in `scripts/sync-vendor.sh`.

## Updating a dependency

1. Edit the `*_REF` line in `VERSIONS` (a tag such as `b10829`, a branch, or a commit).
2. Run `npm run sync:vendor`. It fetches the upstream repo into
   `~/.cache/llama.rn/` (override with `LLAMA_RN_CACHE_DIR`), exports the
   subset, applies `scripts/patches/<dep>/*.patch`, regenerates
   `src/version.ts` and the version headers upstream normally produces at
   build time (`src/llama-version.h`, `ggml/src/ggml-version.h`,
   `common/build-info.cpp`), and writes the resolved `*_COMMIT` back.
3. Fix any patch that no longer applies (see below), then commit the result.

The daily `sync-llama-cpp` workflow does exactly this for the newest llama.cpp
`bNNNNN` release via `scripts/update-llama-cpp.sh`, on the `auto/sync-llama.cpp`
branch. The branch and its PR are pushed even when the sync or a build fails
(the PR description says which step broke), and while that PR is open later
runs commit on top of the branch instead of resetting it, so fixes pushed there
by hand survive.

## Patching upstream code

Patches are plain `-p1` unified diffs against the upstream tree, one file per
patch, under `scripts/patches/<dep>/`. Text above the first `---` line is kept
when a patch is regenerated, so use it to explain why the patch exists.

To change or add one:

1. Edit the file in place, e.g. `vendor/llama.cpp/src/llama-arch.cpp`.
2. Regenerate the patch from the pristine upstream file:

   ```bash
   scripts/update-patch.sh llama.cpp src/llama-arch.cpp barbet-llama-arch.cpp
   ```

   A file upstream does not have becomes a file-creating patch
   (`src/models/barbet.cpp` is one).
3. `npm run sync:vendor` must leave the tree unchanged (`git status` clean);
   that is the check that the patch set is complete.

Do not edit vendored files without regenerating the patch: the next sync
re-exports upstream and reapplies only what is in `scripts/patches/`.

## Generated files inside the trees

These are produced by scripts, not synced from upstream. `sync-vendor.sh`
never deletes them:

- `llama.cpp/src/llama-version.h`, `llama.cpp/ggml/src/ggml-version.h`,
  `llama.cpp/common/build-info.cpp` (committed, written by the sync)
- `llama.cpp/ggml/src/ggml-metal/ggml-metal-embed-*.s` (gitignored, written by
  `npm run bootstrap`; the Metal kernels embedded into the Apple frameworks)
- `llama.cpp/ggml/src/ggml-hexagon/htp/v73/` (gitignored, written by
  `scripts/build-hexagon-htp.sh`; FastRPC stub for the Android HTP build)
