import scispacy
import spacy
from scispacy.linking import EntityLinker
from pprint import pprint
from spacy import displacy
import pandas as pd
import csv
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Open pdf extracted using MinerU as md file
print("Reading extracted markdown file")
extracted_md = Path("/home/nadun/wd/vlm_histo/kg/docs/who-breast/auto/who-breast.md")
with extracted_md.open("r", encoding="utf-8") as f:
    text = f.read()

# Split into chunks based on headings
chunks = text.split("\n# ")
print(f"Total Chunks: {len(chunks)}")

# For each chunk extract entities
print("Loading SciSpacy model and Entity Linker")
nlp = spacy.load("en_core_sci_sm")
linker = nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "mesh"})
print(f"Pipeline components:{nlp.pipe_names}")
csv_path = Path("/home/nadun/wd/vlm_histo/kg/tmp/extracted_entities.csv")
print(f"Saving extracted entities to: {csv_path}")

# Open CSV file for writing and write header
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["chunk_index", "entity_text", "entity_label", "cui", "score"]

    # Set up CSV writer
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # Process chunks and write entities to CSV immediately
    for i, chunk in enumerate(chunks):
        print(f"Processing Chunk {i+1}/{len(chunks)}")
        doc = nlp(chunk)
        for ent in doc.ents:
            for umls_ent in ent._.kb_ents[:1]: # Get top #1 only
                cui = umls_ent[0]
                score = umls_ent[1]
                writer.writerow({
                    "chunk_index": i,
                    "entity_text": ent.text,
                    "entity_label": ent.label_,
                    "cui": cui,
                    "score": score
                })

print(f"Extracted entities saved to {csv_path}")