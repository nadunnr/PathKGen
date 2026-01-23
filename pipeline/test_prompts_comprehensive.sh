#!/bin/bash
# Optimized prompt testing: 5 prompts × test split batch inference
# Uses batch inference to load model only 5 times instead of 25

# Setup
TMP_DIR="$(dirname "$0")/tmp"
mkdir -p "${TMP_DIR}"

CHECKPOINT="checkpoints/stage2/best_model.pt"
TEST_DATA="/home/nadun/wd/datasets/tcga_report_histgen/primary_site_splits/breast_dataset_split_0.8_0.19_0.01.json"

# Prompts to test
declare -a PROMPTS=(
    "Provide a concise single-paragraph pathology diagnosis:"
    "Pathology diagnosis in one paragraph:"
    "Describe the tissue findings in a single diagnostic paragraph:"
    "Generate a factual pathology diagnosis based only on tissue morphology (single paragraph, no patient demographics):"
    "Clinical pathology diagnosis (concise summary, tissue findings only):"
)

# Prompt names for output directories
PROMPT_NAMES=(
    "prompt1_concise_diag"
    "prompt2_short_direct"
    "prompt3_tissue_findings"
    "prompt4_factual_no_demo"
    "prompt5_clinical_style"
)

echo "========================================================"
echo "Testing 5 Prompts on Test Split (Batch Inference)"
echo "Model will be loaded 5 times (once per prompt)"
echo "========================================================"
echo ""

# Loop through each prompt
for prompt_idx in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$prompt_idx]}"
    prompt_name="${PROMPT_NAMES[$prompt_idx]}"
    output_dir="${TMP_DIR}/${prompt_name}"
    
    echo ""
    echo "========================================================"
    echo "PROMPT $((prompt_idx + 1))/5: ${prompt_name}"
    echo "Prompt: \"${prompt}\""
    echo "========================================================"
    echo ""
    
    # Create output directory for this prompt
    mkdir -p "${output_dir}"
    
    # Run batch inference for this prompt on test split
    echo "Running batch inference on test split..."
    python inference.py \
        --checkpoint "${CHECKPOINT}" \
        --data_json "${TEST_DATA}" \
        --split "test" \
        --skip_slidechat_init \
        --prompt "${prompt}" \
        --max_new_tokens 200 \
        --temperature 0.7 \
        --output_dir "${output_dir}"
    
    # Count generated files
    num_files=$(ls -1 "${output_dir}"/*.txt 2>/dev/null | wc -l)
    echo "✓ Generated ${num_files} reports in ${output_dir}/"
    
    # Show first 3 examples
    echo ""
    echo "Sample outputs (first 3):"
    count=0
    for file in "${output_dir}"/*.txt; do
        if [ -f "$file" ]; then
            echo "  - $(basename "$file"):"
            head -c 120 "$file"
            echo "..."
            echo ""
            count=$((count + 1))
            if [ $count -ge 3 ]; then
                break
            fi
        fi
    done
done

echo ""
echo "========================================================"
echo "Testing Complete!"
echo "========================================================"
echo ""
echo "Results saved to: ${TMP_DIR}/"
echo ""
for prompt_name in "${PROMPT_NAMES[@]}"; do
    count=$(ls -1 "${TMP_DIR}/${prompt_name}"/*.txt 2>/dev/null | wc -l)
    echo "  ${prompt_name}: ${count} reports"
done

echo ""
echo "========================================================"
echo "Next Steps:"
echo "1. Review outputs in ${TMP_DIR}/"
echo "2. Compare prompts to find best results"
echo "3. Check for hallucinations (age, demographics)"
echo "========================================================"
