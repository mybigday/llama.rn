#!/bin/bash
#
# CI entry point of .github/workflows/sync-llama-cpp.yml: bump LLAMA_CPP_REF in
# vendor/VERSIONS to the latest llama.cpp bNNNNN release, re-vendor with
# scripts/sync-vendor.sh, run bootstrap, and commit to the staging branch.
set -euo pipefail

STAGING_BRANCH="auto/sync-llama.cpp-staging"
VERSIONS_FILE="vendor/VERSIONS"

echo "🌱 Preparing staging branch: $STAGING_BRANCH"
git fetch origin main

# Clean up any existing staging branch to ensure fresh start
git push origin --delete "$STAGING_BRANCH" 2>/dev/null || echo "No existing staging branch to delete"
git branch -D "$STAGING_BRANCH" 2>/dev/null || echo "No local staging branch to delete"

git checkout -B "$STAGING_BRANCH" origin/main

echo "🔍 Checking latest llama.cpp build release..."
RELEASES_URL="https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=100"
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  RELEASES=$(curl -fsSL -H "Authorization: token $GITHUB_TOKEN" "$RELEASES_URL")
else
  RELEASES=$(curl -fsSL "$RELEASES_URL")
fi

# GitHub's /releases/latest endpoint excludes prereleases and may return a
# stable vX.Y.Z tag older than the rolling bNNNNN builds we vendor.
LATEST_TAG=$(jq -r '
  map(select(.tag_name | test("^b[0-9]+$")))
  | max_by(.tag_name | ltrimstr("b") | tonumber)
  | .tag_name // empty
' <<< "$RELEASES")

if [[ -z "$LATEST_TAG" ]]; then
  echo "❌ Failed to fetch latest build tag"
  exit 1
fi

CURRENT_TAG=$(sed -n 's/^LLAMA_CPP_REF=//p' "$VERSIONS_FILE")

echo "📌 Latest tag: $LATEST_TAG"
echo "📦 Current tag in $VERSIONS_FILE: $CURRENT_TAG"

commit_if_changed() {
  local message="$1"
  if git diff --quiet && git diff --cached --quiet && [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
    echo "✅ No changes to commit"
    return
  fi
  echo "💾 Committing: $message"
  git add -A
  git commit -m "$message"
}

if [[ "$LATEST_TAG" == "$CURRENT_TAG" ]]; then
  echo "✅ Already at $LATEST_TAG"
  echo "🛠 Re-running the vendor sync to make sure vendor/ matches the pins..."
  ./scripts/sync-vendor.sh
  npm run bootstrap
  commit_if_changed "chore(sync): re-vendor llama.cpp $CURRENT_TAG (no version change)"
else
  echo "📥 Updating llama.cpp to $LATEST_TAG..."
  tmp=$(mktemp)
  sed "s/^LLAMA_CPP_REF=.*/LLAMA_CPP_REF=$LATEST_TAG/" "$VERSIONS_FILE" > "$tmp"
  mv "$tmp" "$VERSIONS_FILE"

  echo "🛠 Vendoring sources and applying patches..."
  ./scripts/sync-vendor.sh
  npm run bootstrap
  commit_if_changed "chore: update llama.cpp to $LATEST_TAG"
fi

if [[ -z "${IGNORE_PUSH:-}" ]]; then
  git push origin "$STAGING_BRANCH"
  echo "🚀 Sync committed and pushed to staging branch"
else
  echo "Ignoring push due to IGNORE_PUSH flag"
fi
