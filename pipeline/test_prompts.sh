#!/bin/bash
# Test different prompts to find the best format for concise diagnoses

# Create tmp directory if it doesn't exist
TMP_DIR="$(dirname "$0")/tmp"
mkdir -p "${TMP_DIR}"

CHECKPOINT="checkpoints/stage2/best_model.pt"
TEST_SLIDE="/home/nadun/wd/datasets/trident_processed/breast_wsi_dataset/20x_512px_0px_overlap/slide_features_titan/TCGA-A1-A0SE-01Z-00-DX1.04B09232-C6C4-46EF-AA2C-41D078D0A80A.h5"

echo "Testing different prompts for concise output..."
echo "=============================================="

# Test 1: Empty prompt
echo -e "\n[Test 1] Empty prompt"
python inference.py \
    --checkpoint ${CHECKPOINT} \
    --feature_path ${TEST_SLIDE} \
    --skip_slidechat_init \
    --prompt "" \
    --max_new_tokens 200 \
    --temperature 0.7 \
    --output_path "${TMP_DIR}/test1_empty.txt"

echo -e "\nGenerated:"
cat "${TMP_DIR}/test1_empty.txt"
echo -e "\n---"

# Test 2: Just "Diagnosis:"
echo -e "\n[Test 2] Prompt: 'Diagnosis:'"
python inference.py \
    --checkpoint ${CHECKPOINT} \
    --feature_path ${TEST_SLIDE} \
    --skip_slidechat_init \
    --prompt "Diagnosis:" \
    --max_new_tokens 200 \
    --temperature 0.7 \
    --output_path "${TMP_DIR}/test2_diagnosis.txt"

echo -e "\nGenerated:"
cat "${TMP_DIR}/test2_diagnosis.txt"
echo -e "\n---"

# Test 3: Concise instruction
echo -e "\n[Test 3] Prompt: 'Provide a concise single-paragraph pathology diagnosis:'"
python inference.py \
    --checkpoint ${CHECKPOINT} \
    --feature_path ${TEST_SLIDE} \
    --skip_slidechat_init \
    --prompt "Provide a concise single-paragraph pathology diagnosis:" \
    --max_new_tokens 200 \
    --temperature 0.7 \
    --output_path "${TMP_DIR}/test3_concise.txt"

echo -e "\nGenerated:"
cat "${TMP_DIR}/test3_concise.txt"
echo -e "\n---"

# Test 4: Format instruction
echo -e "\n[Test 4] Prompt with format example"
python inference.py \
    --checkpoint ${CHECKPOINT} \
    --feature_path ${TEST_SLIDE} \
    --skip_slidechat_init \
    --prompt "Generate a diagnosis in this format: [tumor type], [grade], [size], [classification].\nDiagnosis:" \
    --max_new_tokens 200 \
    --temperature 0.7 \
    --output_path "${TMP_DIR}/test4_format.txt"

echo -e "\nGenerated:"
cat "${TMP_DIR}/test4_format.txt"
echo -e "\n---"

echo -e "\nGround Truth for comparison:"
echo "Moderately differentiated invasive lobular carcinoma (tumor diameter: 8 cm) found in ablated breast sample. Tumor-free dorsal resection margin. Tumor classification for right breast: lobular carcinoma, G II, pT3N0L0V0R1."
