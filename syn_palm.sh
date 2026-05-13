#!/bin/bash

set -euo pipefail

CONFIG_PATH="${1:-configs/sample_palm.json}"

accelerate launch scripts/inference/generate_palm_images.py --config "${CONFIG_PATH}"
