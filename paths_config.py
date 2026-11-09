import os

# Base project directory: the repository root by default (override with `export VLM_ROOT=/path/to/PathKGen`)
VLM_ROOT = os.environ.get("VLM_ROOT", os.path.dirname(os.path.abspath(__file__)))

# Key subdirectories
DATA_DIR = os.path.join(VLM_ROOT, "data")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
CHECKPOINTS_DIR = os.path.join(VLM_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(VLM_ROOT, "results")
