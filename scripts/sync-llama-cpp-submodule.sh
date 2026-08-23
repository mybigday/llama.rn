#!/bin/bash
set -euo pipefail

STAGING_BRANCH="auto/sync-llama.cpp-staging"
LLAMA_DIR="third_party/llama.cpp"

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

cd "$LLAMA_DIR"
CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "none")
cd -

echo "📌 Latest tag: $LATEST_TAG"
echo "📦 Current tag in llama.cpp: $CURRENT_TAG"

if [[ "$LATEST_TAG" == "$CURRENT_TAG" ]]; then
  echo "✅ Already synced to $LATEST_TAG"
  echo "🛠 Running bootstrap to ensure cpp/ directory is up to date..."
  npm run bootstrap

  # Check if bootstrap created any changes
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "💾 Committing bootstrap changes..."
    git add -A
    git commit -m "chore(sync): update cpp/ directory with bootstrap (no llama.cpp version change)"
  fi

  # Still need to push the staging branch for the workflow to continue
  if [[ -z "${IGNORE_PUSH:-}" ]]; then
    git push origin "$STAGING_BRANCH"
  fi
  exit 0
fi

echo "📥 Updating llama.cpp to $LATEST_TAG..."
cd "$LLAMA_DIR"
git fetch --tags
git checkout "refs/tags/$LATEST_TAG"
cd -

git add "$LLAMA_DIR"
git commit -m "chore: update llama.cpp to $LATEST_TAG (submodule ref)"

echo "🛠 Running bootstrap to copy files and apply patches..."
npm run bootstrap

# Check if bootstrap created any changes in cpp/ directory
if git diff --quiet && git diff --cached --quiet; then
  echo "✅ No changes after bootstrap — cpp/ directory already up to date."
else
  echo "💾 Committing bootstrap changes..."
  git add -A
  git commit -m "chore(sync): update cpp/ directory after llama.cpp $LATEST_TAG bootstrap"
fi

if [[ -z "${IGNORE_PUSH:-}" ]]; then
  git push origin "$STAGING_BRANCH"
  echo "🚀 Submodule updated, bootstrap completed, and committed to staging branch"
else
  echo "Ignoring push due to IGNORE_PUSH flag"
fi
