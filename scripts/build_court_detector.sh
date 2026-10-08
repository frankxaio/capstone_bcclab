#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$repo_root/court_detection" -B "$repo_root/court_detection/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$repo_root/court_detection/build" --parallel 2
