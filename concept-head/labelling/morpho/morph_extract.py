import csv
import json
from tqdm import tqdm
import ollama
import re

# CONFIG
INPUT_CSV = "/home/nadun/wd/datasets/wsi_bench/dataset/report_dataset_train.csv"
OUTPUT_JSON = "tumor_features_llm.jsonl"

MODEL_NAME = "gpt-oss"
MAX_FEATURES = 5

# Prompt template
SYSTEM_PROMPT = """You are a histopathology expert.
Extract key tumor morphological features from pathology reports.
Use concise pathology terms only.
"""

USER_PROMPT_TEMPLATE = """From the pathology report below, extract 3 to 5 primary tumor or morphological features.

Rules:
- Use concise medical terms (1-3 words each)
- Only include features explicitly PRESENT
- Do NOT include negated or absent features
- Do NOT include diagnosis, site, grade, or stage
- Do NOT include normal tissue findings
- Examples of valid features:
  keratinization, nuclear pleomorphism, hyperchromasia,
  well differentiated, poor differentiation, stromal invasion

Return ONLY a JSON array of strings.

Pathology report:
\"\"\"{report}\"\"\"
"""

# LLM call
def extract_features_with_llm(report_text):
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(report=report_text)
            }
        ],
        options={
            "temperature": 0.2,
            "top_p": 0.9
        }
    )

    content = response["message"]["content"].strip()

    # Try to parse JSON array
    try:
        features = json.loads(content)
        if isinstance(features, list):
            # Normalize + truncate
            features = [
                f.lower().strip()
                for f in features
                if isinstance(f, str) and f.strip()
            ]
            return features[:MAX_FEATURES]
    except json.JSONDecodeError:
        pass

    return []


# Main processing loop
def process_csv():
    with open(INPUT_CSV, "r", encoding="utf-8") as in_f, \
         open(OUTPUT_JSON, "a", encoding="utf-8") as out_f:

        reader = csv.DictReader(in_f)

        for row in tqdm(reader, desc="Extracting tumor features"):
            tcga_id = row.get("tcga_id")
            report = row.get("report_content")

            if not tcga_id or not report:
                continue

            features = extract_features_with_llm(report)

            if features:
                record = {
                    "tcga_id": tcga_id,
                    "tumor_features": features
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()


# Run
if __name__ == "__main__":
    process_csv()
