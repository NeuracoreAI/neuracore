#!/bin/bash
# Writes the attempt-2 training job IDs into the Actions cache. Exits 0 even
# when the save fails so this dispatch cannot fail the data-daemon workflow.
set -u
summary="${GITHUB_STEP_SUMMARY:-/tmp/reseed-summary.md}"
{
  echo "### Training cache reseed"
  echo "pwd=$(pwd)"
  echo "workspace=${GITHUB_WORKSPACE:-unset}"
  echo "node=$(command -v node || echo missing)"
  echo "npm=$(command -v npm || echo missing)"
  echo "results_url=${ACTIONS_RESULTS_URL:+set}"
  echo "cache_url=${ACTIONS_CACHE_URL:+set}"
  echo "token=${ACTIONS_RUNTIME_TOKEN:+set}"
} >> "$summary"

cd "${GITHUB_WORKSPACE:-.}"
dir="$(mktemp -d)"
if ! npm install --prefix "$dir" @actions/cache@4.1.0 >> "$summary" 2>&1; then
  echo "npm install failed" >> "$summary"
  exit 0
fi
if ! NODE_PATH="$dir/node_modules" node "${GITHUB_WORKSPACE}/.github/reseed-training-cache.js" >> "$summary" 2>&1; then
  echo "node reseed failed" >> "$summary"
fi
exit 0
