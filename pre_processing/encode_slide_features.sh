#!/bin/bash
# Encode WSIs into slide features with TRIDENT (CONCH v1.5 patch encoder + TITAN slide encoder).
#
# Usage:  bash pre_processing/encode_slide_features.sh /path/to/TRIDENT
# Needs:  .svs slides in $DATA_DIR/temp_wsis/
# Output: $DATA_DIR/trident_processed/ (log: pipeline_status.log). Place the resulting .h5 files in
#         $DATA_DIR/slide_features/{train,test}/ before running preprocess_wsi_bench.sh.
# Note:   the TITAN code in TRIDENT must be modified to also save the 2D visual tokens.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TRIDENT_DIR="${1:?Usage: bash $0 /path/to/TRIDENT}"

python "$HERE/generate_mpp.py"      # writes temp_wsis/slide_list.csv (slides without MPP metadata are removed)

cd "$TRIDENT_DIR"
bash "$HERE/encode_wsis.sh"         # run from the TRIDENT directory (needs run_batch_of_slides.py)
