#!/usr/bin/env bash

set -e

echo "🧹 Cleaning project artifacts..."

# delete recordings directory (project-local only)
rm -rf ./recordings

# delete sqlite files (handle both spellings just in case)
rm -f data_deamon_state.db*
rm -f data_daemon_state.db*

# delete wal/shm variants anywhere in repo root
rm -f *.db-wal
rm -f *.db-shm

mkdir -p ./recordings

echo "✅ Cleanup complete and /db/recordings/ directory recreated."
