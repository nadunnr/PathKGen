import sys
from pathlib import Path
import scispacy
import spacy
from pprint import pprint
from spacy import displacy
from scispacy.linking import EntityLinker
from sentence_generator import get_prompt
from sentence_clusterer import SentenceClusterer
import warnings
warnings.filterwarnings("ignore")


class KnowledgeExtractor:
    def __init__(self, model_name: str = "en_core_sci_sm", linker_name: str = "mesh"):
        print(f"Loading scispacy model '{model_name}' and UMLS entity linker '{linker_name}'...")
        
        # Load scispacy model
        self.nlp = spacy.load(model_name)
        
        # Add entity linker to the pipeline
        self.linker = self.nlp.add_pipe(
            "scispacy_linker", 
            config={"resolve_abbreviations": True, "linker_name": linker_name}
        )
        
        # Initialize sentence clusterer
        self.clusterer = SentenceClusterer()
        print("KnowledgeExtractor initialized successfully!")
    
    def extract_knowledge(
        self, 
        report_text: str, 
        max_cuis: int = 20, 
        min_score: float = 0.9,
        max_sentences: int = 10
    ) -> str:
    
        # Process the report text with scispacy
        doc = self.nlp(report_text)
        
        # Extract unique CUIs
        seed_cuis = set()
        for ent in doc.ents:
            if ent._.kb_ents:
                cui, score = ent._.kb_ents[0]  # Get top match
                if score >= min_score:
                    seed_cuis.add(cui)
        
        # Limit to max_cuis
        seed_cuis = list(seed_cuis)[:max_cuis]
        
        if not seed_cuis:
            print("Warning: No UMLS entities found in the report text.")
            return ""
        
        # Generate sentences for each CUI
        all_sentences = []
        for cui in seed_cuis:
            sentences = get_prompt(cui)
            if sentences:
                all_sentences.extend(sentences)
        
        if not all_sentences:
            print("Warning: No knowledge sentences were generated.")
            return ""
        
        # Filter by clustering
        if len(all_sentences) > max_sentences:
            best_sentences = self.clusterer.cluster_sentences(all_sentences, max_sentences)
        else:
            best_sentences = all_sentences
        
        # Join sentences to create a paragraph
        knowledge_paragraph = "".join(best_sentences)
        return knowledge_paragraph


# Pre-compute knowledge for all reports in the dataset
if __name__ == "__main__":
    import json
    from tqdm import tqdm
    
    # Path to the JSON file
    json_file_path = "/home/nadun/wd/datasets/tcga_report_histgen/primary_site_splits/breast_dataset_split_0.7_0.1_0.2 copy.json"
    
    print(f"Loading dataset from: {json_file_path}")
    
    # Load the JSON file
    with open(json_file_path, 'r') as f:
        dataset = json.load(f)
    
    # Initialize the extractor once
    print("\nInitializing Knowledge Extractor...")
    extractor = KnowledgeExtractor()
    
    # Process each split (train, val, test)
    for split_name in ['train', 'val', 'test']:
        if split_name not in dataset:
            print(f"\nWarning: Split '{split_name}' not found in dataset, skipping...")
            continue
        
        items = dataset[split_name]
        print(f"\n{'='*80}")
        print(f"Processing {split_name} split: {len(items)} items")
        print(f"{'='*80}")
        
        # Process each item with progress bar
        for idx, item in enumerate(tqdm(items, desc=f"Generating knowledge for {split_name}")):
            # Skip if knowledge already exists
            if 'knowledge' in item and item['knowledge']:
                continue
            
            # Get the report text
            report_text = item.get('report', '')
            
            if not report_text:
                print(f"\nWarning: No report text found for item {item.get('id', 'unknown')}")
                item['knowledge'] = ""
                # Save after each update
                with open(json_file_path, 'w') as f:
                    json.dump(dataset, f, indent=4)
                continue
            
            # Generate knowledge paragraph
            try:
                knowledge_paragraph = extractor.extract_knowledge(report_text)
                item['knowledge'] = knowledge_paragraph
            except Exception as e:
                print(f"\nError processing item {item.get('id', 'unknown')}: {str(e)}")
                item['knowledge'] = ""
            
            # Save to file after each item is processed
            with open(json_file_path, 'w') as f:
                json.dump(dataset, f, indent=4)
    
    print(f"\n{'='*80}")
    print("Done! All knowledge paragraphs have been generated and saved.")
    print(f"{'='*80}")

