#!/bin/bash
# Writes the attempt-2 training job IDs into the Actions cache. Exits 0 even
# when the save fails so this dispatch cannot fail the data-daemon workflow.
set -u
cd "${GITHUB_WORKSPACE:-.}"
dir="$(mktemp -d)"
if ! npm install --prefix "$dir" @actions/cache@4.1.0; then
  echo "::warning::Could not install @actions/cache"
  exit 0
fi
if ! NODE_PATH="$dir/node_modules" node "${GITHUB_WORKSPACE}/.github/reseed-training-cache.js"; then
  echo "::warning::Training cache reseed failed"
fi
exit 0
