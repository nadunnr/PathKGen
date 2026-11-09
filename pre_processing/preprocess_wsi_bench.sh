#!/bin/bash
# Convert raw WSI-Bench to standardised JSON, then keep the Report task and slides with available features.
#
# Usage:  bash pre_processing/preprocess_wsi_bench.sh [--use_gdc]   (--use_gdc fetches metadata from the GDC API)
# Needs:  $DATA_DIR/wsi_bench/dataset_raw/{WSI-Bench-train.json,WSI-Bench-test.jsonl}
#         $DATA_DIR/slide_features/{train,test}/*.h5
# Output: $DATASET_DIR/wsi_bench_Report_with_features/wsi_bench_Report_{train,test}.json
set -euo pipefail

cd "$(dirname "$0")"

python convert_wsi_bench.py "$@"
python filter_wsi_bench.py
