import sys
from pathlib import Path
import scispacy
import spacy
from pprint import pprint
from spacy import displacy
from scispacy.linking import EntityLinker
from .sentence_generator import get_prompt
from .sentence_clusterer import SentenceClusterer
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
        
        print(f"Identified {len(seed_cuis)} CUIs in the report text.")
        
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
                print(f"Generated {len(sentences)} new sentences for CUI {cui}")
                all_sentences.extend(sentences)
        
        if not all_sentences:
            print("Warning: No knowledge sentences were generated.")
            return ""
        
        # Filter by clustering
        best_sentences = self.clusterer.cluster_sentences(all_sentences, max_sentences)
        
        # Join sentences to create a paragraph
        knowledge_paragraph = "".join(best_sentences)
        return knowledge_paragraph


# Test script
if __name__ == "__main__":
    # Test report
    test_report = """
    The pathology report describes the specimens labeled A, B, and C, which include wide local excisions and an additional margin as part of the surgical treatment of a female patient with bilateral masses and a preoperative diagnosis of bilateral invasive ductal carcinoma. 
    The report reveals that the right breast excision (Part A) measures 2.4 cm and contains invasive ductal carcinoma, SBR grade 2, mixed with ductal carcinoma in situ, lobular carcinoma in situ, and biopsy site changes. The report also indicates that invasive tumor is present at the superior margins and within 1-mm from medial in parts A and B, respectively.
    The left breast excision (Part C) measures 4 x 4 x 3.5 cm and contains invasive ductal carcinoma, SBR grade 2, and lobular carcinoma in situ, but with negative surgical resection margins.
    The report further provides additional information, such as specimen processing, margin status, tumor characteristics, lymph node status, and ER/PR/HER2 results.
    """
    
    print(f"\nInput Report:\n{test_report}\n")
    
    # Initialize the extractor once
    extractor = KnowledgeExtractor()
    
    # Extract knowledge (can be called multiple times without reloading the model)
    knowledge_paragraph = extractor.extract_knowledge(test_report)
    
    print("\n" + "="*80)
    print("GENERATED KNOWLEDGE PARAGRAPH:")
    print("="*80)
    print(knowledge_paragraph)
    print("="*80)

