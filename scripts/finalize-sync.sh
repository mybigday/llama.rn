#!/bin/bash
set -euo pipefail

WORK_BRANCH="auto/sync-llama.cpp"
STAGING_BRANCH="auto/sync-llama.cpp-staging"

echo "🔄 Finalizing sync after successful builds..."

# Checkout the staging branch
git fetch origin "$STAGING_BRANCH"
git checkout "$STAGING_BRANCH"

# Run TypeScript build to complete the sync
echo "🛠 Running TypeScript build..."
yarn build

# Check if there are any additional changes after TypeScript build
if git diff --quiet && git diff --cached --quiet; then
  echo "✅ No additional changes after TypeScript build."
else
  echo "💾 Committing TypeScript build changes..."
  git add -A
  git commit -m "chore(sync): regenerate TypeScript build artifacts"
fi

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
git push origin --delete "$STAGING_BRANCH" || echo "Staging branch already deleted"

echo "🚀 Successfully synced and pushed to $WORK_BRANCH"
