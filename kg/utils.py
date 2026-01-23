from pathlib import Path
import scispacy
import spacy
from scispacy.linking import EntityLinker
from spacy import displacy
import pandas as pd
import requests
from typing import List, Dict
import warnings
import tqdm
warnings.filterwarnings("ignore")

UMLS_API_KEY = "21b2a93f-3178-4807-b2e1-513bdbf1af8e"  


def read_md(path: Path="/home/nadun/wd/vlm_histo/kg/docs/who-breast/auto/who-breast.md"):
    "read md in chunks (extracted by mineru)"
    md_path = Path(path)
    with md_path.open("r", encoding="utf-8") as f:
        text = f.read()
    
    chunk_size = 10000   #fix later 1M possible
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def entity_linking(text: str):
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    doc = nlp(text)
    # Store linker reference on doc for later use
    linker = nlp.get_pipe("scispacy_linker")
    return doc, linker


def get_relations_from_api(cui: str, api_key=UMLS_API_KEY) -> List[Dict]:
    """Fetch relationships for a CUI from UMLS REST API and return a list of relationship dictionaries
       reference: https://documentation.uts.nlm.nih.gov/rest/home.html
    """
    if not api_key:
        return []
    
    url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/relations"
    try:
        response = requests.get(url, params={"apiKey": api_key}, timeout=10)
        if response.status_code != 200:
            return []
        
        relations = []
        for item in response.json().get("result", []):
            relations.append({
                "relation_type": item.get("relationLabel", ""),
                "additional_relation": item.get("additionalRelationLabel", ""),
                "related_cui": item.get("relatedId", "").split("/")[-1] if item.get("relatedId") else "",
                "related_name": item.get("relatedIdName", ""),
            })
        return relations
    except Exception:
        return []


def linked_entities_with_relations(doc, linker, api_key=UMLS_API_KEY) -> pd.DataFrame:
    """Extract linked entities AND their UMLS relations.
    Args:
        doc: spaCy Doc object
        linker: scispaCy EntityLinker
        api_key: UMLS API key
    Returns:
        DataFrame with entities and their relations"""
    
    data = []
    for ent in tqdm.tqdm(doc.ents):
        if ent._.kb_ents:
            cui, score = ent._.kb_ents[0]  # Top match only
            concept = linker.kb.cui_to_entity[cui]
            definition = getattr(concept, 'definition', None) or ''
            relations = get_relations_from_api(cui, api_key)
            data.append({
                "entity_text": ent.text, "entity_label": ent.label_, "cui": cui,
                "concept_name": concept.canonical_name, "score": round(score, 4),
                "definition": definition, "num_relations": len(relations), "relations": relations})
    return pd.DataFrame(data)


def visualize_ents(doc):
    options = {"ents": ["ENTITY"], "colors": {"ENTITY": "lightblue"}}
    displacy.serve(doc, style="ent", options=options)


if __name__ == "__main__":
    text = next(read_md())
    doc, linker = entity_linking(text)
    
    df_with_rels = linked_entities_with_relations(doc, linker, api_key=UMLS_API_KEY)   #API is faster
    df_with_rels.to_csv("tmp/linked_entities.csv", index=False)
    print(f"Saved {len(df_with_rels)}")

   
   
   
   
#can hv issues
    #only extracting top entity match for now




# MRREL_PATH = "/home/nadun/wd/datasets/UMLS/extracted/MRREL.RRF.aa"
# 
# def get_relations_from_local(cui: str, mrrel_path: str = MRREL_PATH, max_relations: int = 50) -> pd.DataFrame:
#     """Query local MRREL.RRF file for relationships.
    
#     Args:
#         cui: UMLS Concept Unique Identifier
#         mrrel_path: Path to MRREL.RRF file
#         max_relations: Maximum number of relations to return
        
#     Returns:
#         DataFrame with relationships
#     """
#     cols = ['CUI1', 'AUI1', 'STYPE1', 'REL', 'CUI2', 'AUI2', 'STYPE2', 
#             'RELA', 'RUI', 'SRUI', 'SAB', 'SL', 'RG', 'DIR', 'SUPPRESS', 'CVF', 'EMPTY']
    
#     if not Path(mrrel_path).exists():
#         return pd.DataFrame()
    
#     relations = []
#     total_found = 0
    
#     for chunk in pd.read_csv(mrrel_path, sep='|', header=None, names=cols,
#                              dtype=str, chunksize=100000, on_bad_lines='skip'):
#         mask = (chunk['CUI1'] == cui) | (chunk['CUI2'] == cui)
#         matched = chunk[mask]
#         relations.append(matched)
#         total_found += len(matched)
        
#         if total_found >= max_relations:
#             break
    
#     if relations:
#         df = pd.concat(relations, ignore_index=True).head(max_relations)
#         return df[['CUI1', 'REL', 'RELA', 'CUI2', 'SAB']]
#     return pd.DataFrame()