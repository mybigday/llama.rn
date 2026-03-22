#!/bin/bash
# WorktreeCreate hook: creates git worktree, copies gitignored configs, installs & builds.
# stdout is reserved for the worktree path — all logging goes to stderr.
set -euo pipefail

INPUT=$(cat)
NAME=$(echo "$INPUT" | jq -r '.name')
REPO_ROOT=$(git rev-parse --show-toplevel)
WORKTREE_PATH="$REPO_ROOT/.claude/worktrees/$NAME"

log() { echo "$@" >&2; }

copy_if_exists() {
  local src="$1" dst="$2"
  if [ -f "$src" ]; then
    cp "$src" "$dst" && log "  Copied $(basename "$src")"
  fi
}

# Create git worktree
git worktree add -b "worktree/$NAME" "$WORKTREE_PATH" HEAD >&2

log "Setting up worktree at $WORKTREE_PATH..."

# Claude Code local settings (user allowed permissions)
mkdir -p "$WORKTREE_PATH/.claude"
copy_if_exists "$REPO_ROOT/.claude/settings.local.json" "$WORKTREE_PATH/.claude/"

# Pre-built iOS frameworks (gitignored .framework bundles inside xcframework)
if [ -d "$REPO_ROOT/ios/rnllama.xcframework" ]; then
  log "Copying pre-built iOS frameworks..."
  for arch_dir in "$REPO_ROOT"/ios/rnllama.xcframework/*/; do
    arch=$(basename "$arch_dir")
    fw_src="$arch_dir/rnllama.framework"
    if [ -d "$fw_src" ]; then
      mkdir -p "$WORKTREE_PATH/ios/rnllama.xcframework/$arch"
      cp -R "$fw_src" "$WORKTREE_PATH/ios/rnllama.xcframework/$arch/"
      log "  Copied $arch/rnllama.framework"
    fi
  done
fi

# Pre-built Android native libs
if [ -d "$REPO_ROOT/jniLibs" ]; then
  log "Copying pre-built Android libs..."
  cp -R "$REPO_ROOT/jniLibs" "$WORKTREE_PATH/"
fi

# Install dependencies
if [ ! -d "$WORKTREE_PATH/node_modules" ]; then
  log "Installing dependencies..."
  cd "$WORKTREE_PATH"
  npm install >&2
fi

# Print worktree path to stdout (Claude Code uses this)
echo "$WORKTREE_PATH"
