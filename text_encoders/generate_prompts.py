import requests
import json, os
import pandas as pd
import re
import ollama
from pathlib import Path


def get_response(f):
    response = ollama.chat(
        'deepseek-r1:8b',
        messages=[{
            'role': 'user',
            'content': f"""
                You are a clinical pathologist. Describe the microscopic features observed in an H&E-stained histopathology slide of the {f['tissue']} from a {f['species']} in a {f['disease_state']} condition ({f['disease_comment']}).

                Write a single, well-structured paragraph summarizing key histopathological characteristics typical for this condition. 
                Use objective and formal language — as found in a pathology report — without any prefaces, introductions, or conversational phrases.
                Do not start with words like “Okay”, “Here is”, “This is”, or similar.
                Only provide the descriptive report text.
            """
        }],
    )

    msg = response['message']['content']
    clean_text = re.sub(r"<think>.*?</think>", "", msg, flags=re.DOTALL).strip()
    return clean_text

# -------- Load IDs from saved text file --------
ids_file = Path("/home/nadun/hdd/literature-guided-digital-histopathology/text_encoders/file_names.txt")

with open(ids_file, "r", encoding="utf-8") as f:
    ids_to_query = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(ids_to_query)} IDs from {ids_file}")

# -------- Run inference --------
outputs = []
outfile = 'hest_responses.jsonl'

for idx, slide_id in enumerate(ids_to_query):
    print(f"Processing {idx+1}/{len(ids_to_query)}: {slide_id}")
    meta_path = f"/home/nadun/wd/HEST/hest_data_40/metadata/{slide_id}.json"
    if not os.path.exists(meta_path):
        print(f"⚠️ Metadata not found for {slide_id}, skipping...")
        continue

    with open(meta_path, 'r', encoding='utf-8') as f:
        file_content = json.load(f)

    content = get_response(file_content)

    if isinstance(content, dict) and 'error' in content:
        print(f"Error for {slide_id}: {content['error']}")
        continue

    record = {"id": slide_id, "response": content}
    with open(outfile, 'a', encoding='utf-8') as f_out:
        f_out.write(json.dumps(record, ensure_ascii=False))
        f_out.write("\n")