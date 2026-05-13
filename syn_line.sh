#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

CONFIG_PATH="${1:-configs/syn_line.json}"

python scripts/PolyCreases/syn_polypalm_mp.py --config "${CONFIG_PATH}"
