#!/bin/bash
set -euo pipefail

STAGING_BRANCH="auto/sync-llama.cpp-staging"
LLAMA_DIR="llama.cpp"

echo "🌱 Preparing staging branch: $STAGING_BRANCH"
git fetch origin main
git checkout -B "$STAGING_BRANCH" origin/main

echo "🔍 Checking latest llama.cpp release..."
LATEST_TAG=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | jq -r .tag_name)

if [[ -z "$LATEST_TAG" || "$LATEST_TAG" == "null" ]]; then
  echo "❌ Failed to fetch latest tag"
  exit 1
fi

cd "$LLAMA_DIR"
CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "none")
cd ..

echo "📌 Latest tag: $LATEST_TAG"
echo "📦 Current tag in llama.cpp: $CURRENT_TAG"

if [[ "$LATEST_TAG" == "$CURRENT_TAG" ]]; then
  echo "✅ Already synced to $LATEST_TAG"
  # Still need to push the staging branch for the workflow to continue
  git push origin "$STAGING_BRANCH"
  exit 0
fi

echo "📥 Updating llama.cpp to $LATEST_TAG..."
cd "$LLAMA_DIR"
git fetch --tags
git checkout "refs/tags/$LATEST_TAG"
cd ..

git add "$LLAMA_DIR"
git commit -m "chore: update llama.cpp to $LATEST_TAG (submodule ref)"

echo "🚀 Submodule updated and committed to staging branch"
