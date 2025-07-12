#!/bin/bash
set -euo pipefail

WORK_BRANCH="auto/sync-llama.cpp"
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
  exit 0
fi

echo "📥 Updating llama.cpp to $LATEST_TAG..."
cd "$LLAMA_DIR"
git fetch --tags
git checkout "refs/tags/$LATEST_TAG"
cd ..

git add "$LLAMA_DIR"
git commit -m "chore: update llama.cpp to $LATEST_TAG (submodule ref)"

echo "🛠 Running bootstrap and build..."
yarn bootstrap
yarn prepack

if git diff --quiet && git diff --cached --quiet; then
  echo "✅ No changes after sync — nothing to commit."
  exit 0
fi

echo "💾 Committing changes to staging branch..."
git add -A
git commit -m "chore(sync): update llama.cpp to $LATEST_TAG"

echo "✅ Build successful! Moving to persistent sync branch..."

# Check if the persistent sync branch exists
if git show-ref --verify --quiet refs/remotes/origin/"$WORK_BRANCH"; then
  echo "📦 Persistent branch exists, checking out..."
  git fetch origin "$WORK_BRANCH"
  git checkout -B "$WORK_BRANCH" origin/"$WORK_BRANCH"
else
  echo "🆕 Creating new persistent branch..."
  git checkout -b "$WORK_BRANCH"
fi

# Fast-forward merge the staging branch changes
echo "🔄 Fast-forwarding persistent branch..."
git merge --ff-only "$STAGING_BRANCH"

# Push the persistent branch
git push origin "$WORK_BRANCH"

# Clean up staging branch
git branch -D "$STAGING_BRANCH"

echo "🚀 Successfully synced and pushed to $WORK_BRANCH"
