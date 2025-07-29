#!/bin/bash
set -euo pipefail

WORK_BRANCH="auto/sync-llama.cpp"
STAGING_BRANCH="auto/sync-llama.cpp-staging"

echo "🔄 Finalizing sync after successful builds..."

# We're already on the staging branch, just need to move to persistent branch
echo "✅ All builds successful! Moving to persistent sync branch..."

# Check if the persistent sync branch exists
if git show-ref --verify --quiet refs/remotes/origin/"$WORK_BRANCH"; then
  echo "📦 Persistent branch exists, checking out..."
  git fetch origin "$WORK_BRANCH"
  git checkout -B "$WORK_BRANCH" origin/"$WORK_BRANCH"
else
  echo "🆕 Creating new persistent branch..."
  git checkout -b "$WORK_BRANCH"
fi

# Reset the persistent branch to match staging branch exactly
# This handles cases where branches have diverged
echo "🔄 Updating persistent branch to match staging branch..."
git reset --hard "$STAGING_BRANCH"

# Push the persistent branch (force push since we may have reset)
git push origin "$WORK_BRANCH" --force-with-lease

# Clean up staging branch
git push origin --delete "$STAGING_BRANCH" || echo "Staging branch already deleted"

echo "🚀 Successfully synced and pushed to $WORK_BRANCH"
