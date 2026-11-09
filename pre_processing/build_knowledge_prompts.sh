#!/bin/bash
# Train the auxiliary histology classifier, predict histology for train/test, and write the knowledge prompts
# into the dataset JSONs.
#
# Usage:  bash pre_processing/build_knowledge_prompts.sh
# Needs:  output of preprocess_wsi_bench.sh
#         $DATA_DIR/wsi_bench/kg_data/descriptions.json
#         $DATA_DIR/wsi_bench/custom_kg/relations.csv
# Output: $DATASET_DIR/WSI-Bench-{train,test}_Report_knowledge.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/paths_config.sh"

# The classifier reads slide features from $DATA_DIR/wsi_bench/slide_features
mkdir -p "$DATA_DIR/wsi_bench"
[ -e "$DATA_DIR/wsi_bench/slide_features" ] || ln -s "$DATA_DIR/slide_features" "$DATA_DIR/wsi_bench/slide_features"

# 1. Auxiliary classifier: train, then predict histology for each split
cd "$ROOT/auxiliary_classifier"
python classifier.py

PRED_DIR="$DATASET_DIR/wsi_bench_Report_knowledge"   # predictions in the JSON format enrich_with_knowledge.py reads
mkdir -p "$PRED_DIR"
for SPLIT in train test; do
    sed "s/^SPLIT = .*/SPLIT = \"$SPLIT\"/" inference.py > "inference_$SPLIT.py"   # inference.py hard-codes SPLIT
    python "inference_$SPLIT.py"
    rm "inference_$SPLIT.py"

    python - "$SPLIT" "$PRED_DIR" <<'EOF'
import csv, json, sys
split, pred_dir = sys.argv[1:3]
with open(f"output/classifier_{split}_pred.csv") as f:
    preds = [{"tcga_id": r["tcga_id"], "pred_histology": r["predicted_label"]} for r in csv.DictReader(f)]
with open(f"{pred_dir}/WSI-Bench-{split}_Report_knowledge.json", "w") as f:
    json.dump(preds, f, indent=2)
EOF
done

# 2. Per-histology knowledge prompts, then enrich the dataset JSONs with them
python "$ROOT/knowledge_path/graph_construct/create_knowledge_prompt.py"
python "$ROOT/knowledge_path/graph_construct/enrich_with_knowledge.py"
