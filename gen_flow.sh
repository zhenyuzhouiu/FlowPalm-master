#!/bin/bash

set -euo pipefail

CONFIG_PATH="${1:-configs/gen_flow.json}"

python tools/flow/generate_flow_files.py --config "${CONFIG_PATH}"
