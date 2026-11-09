# Base project directory: the repository root by default (override with `export VLM_ROOT=/path/to/PathKGen`)
VLM_ROOT="${VLM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Key subdirectories
DATA_DIR="$VLM_ROOT/data"
DATASET_DIR="$DATA_DIR/dataset"
CHECKPOINTS_DIR="$VLM_ROOT/checkpoints"
RESULTS_DIR="$VLM_ROOT/results"
