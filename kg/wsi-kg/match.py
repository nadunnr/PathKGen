import json
import scispacy
import spacy
from scispacy.linking import EntityLinker
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Load Scispacy model
print("Loading scispaCy model...")
nlp = spacy.load("en_core_sci_lg")
nlp.add_pipe(
    "scispacy_linker",
    config={
        "resolve_abbreviations": True,
        "linker_name": "umls"
    }
)
linker = nlp.get_pipe("scispacy_linker")
print("Model loaded successfully!")

# Define the 5 axes with their UMLS TUI mappings
AXES = {
    "histology": {
        "tuis": {"T191"}  # Neoplastic Process
    },
    "variant": {
        "tuis": {"T190", "T037", "T033"}  # Anatomical Abnormality, Injury or Poisoning, Finding
    },
    "grade": {
        "tuis": {"T080", "T185"}  # Qualitative Concept, Classification
    },
    "primary_site": {
        "tuis": {"T023", "T029", "T030"}  # Body Part, Organ or Tissue Component, Body Space or Junction
    },
    "tumor_feature": {
        "tuis": {"T046", "T047", "T033", "T191"}  # Pathologic Function, Disease or Syndrome, Finding, Neoplastic Process
    }
}

SCORE_THRESHOLD = 0.8

def link_text_to_umls(text, axis_name):
    """
    Link a text string to UMLS concepts for a specific axis.
    Returns the best matching concept with score >= threshold, or None.
    """
    if not text or text.strip() == "" or text.lower() in ["n/a", "not specified", "-", "null"]:
        return None
    
    # Process text with scispaCy
    doc = nlp(text.lower())
    
    candidates = []
    axis_config = AXES.get(axis_name, {})
    
    for ent in doc.ents:
        for cui, score in ent._.kb_ents:
            # Filter by score threshold
            if score < SCORE_THRESHOLD:
                continue
                
            entity = linker.kb.cui_to_entity[cui]
            entity_tuis = set(entity.types)
            
            # Check TUI compatibility
            if axis_config.get("tuis") and not entity_tuis.intersection(axis_config["tuis"]):
                continue
            
            candidates.append({
                "text": ent.text,
                "canonical_name": entity.canonical_name,
                "cui": cui,
                "score": round(score, 4)
            })
    
    if not candidates:
        return None
    
    # Deduplicate by CUI, keep highest score
    best_per_cui = {}
    for c in candidates:
        if c["cui"] not in best_per_cui or c["score"] > best_per_cui[c["cui"]]["score"]:
            best_per_cui[c["cui"]] = c
    
    # Return highest scoring concept
    return max(best_per_cui.values(), key=lambda x: x["score"])

# Load input JSON
input_file = "/home/nadun/wd/vlm_histo/kg/wsi-kg/extracted_test.json"
output_file = "/home/nadun/wd/vlm_histo/kg/wsi-kg/linked_test.json"

print(f"Loading data from: {input_file}")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total entries to process: {len(data)}")

# Process each entry
linked_results = []

for entry in tqdm(data, desc="Linking to UMLS"):
    tcga_id = entry.get("tcga_id", "")
    
    linked_entry = {
        "tcga_id": tcga_id,
        "histology": link_text_to_umls(entry.get("histology", ""), "histology"),
        "variant": link_text_to_umls(entry.get("variant", ""), "variant"),
        "grade": link_text_to_umls(entry.get("grade", ""), "grade"),
        "primary_site": link_text_to_umls(entry.get("primary_site", ""), "primary_site"),
        "tumor_feature": link_text_to_umls(entry.get("tumor_feature", ""), "tumor_feature")
    }
    
    linked_results.append(linked_entry)
    
    # Save periodically (every 50 entries) to avoid data loss
    if len(linked_results) % 50 == 0:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(linked_results, out_f, indent=2, ensure_ascii=False)

# Final save
print("\nSaving final results...")
with open(output_file, 'w', encoding='utf-8') as out_f:
    json.dump(linked_results, out_f, indent=2, ensure_ascii=False)

print(f"\nProcessing complete!")
print(f"Results saved to: {output_file}")

# Print sample output
print("\nSample output (first entry):")
if linked_results:
    for key, value in linked_results[0].items():
        if key != "tcga_id":
            print(f"{key}: {value}")
