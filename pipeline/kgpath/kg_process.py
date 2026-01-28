from .cui_concept_encoder import CUIConceptEncoder
from .sentence_generator import get_prompt
from .sentence_clusterer import SentenceClusterer
import warnings
warnings.filterwarnings("ignore")

# Create encoder instance
encoder = CUIConceptEncoder(device="cpu")

# 1. Encode CUI concepts from CSV and save to H5 file
csv_path = "/home/nadun/wd/vlm_histo/kg/tmp/nodes.csv"
output_path = "/home/nadun/wd/vlm_histo/kg/tmp/cui_embeddings.h5"

# encoder.encode_cui_concepts(csv_path, output_path)

# 2. Find top-k similar CUI concepts for a slide
slide_path = "/home/nadun/wd/datasets/trident_processed/breast_wsi_dataset/20x_512px_0px_overlap/slide_features_titan/TCGA-5L-AAT0-01Z-00-DX1.5E171263-30BF-4C6B-88A1-E8EA0522A861.h5"
top_k_results = encoder.get_top_k_similar_concepts(slide_path, output_path, k=10)

# Print top similar CUI concepts
for cui_id, cui_name, score in top_k_results:
    print(f"{cui_id}: {cui_name} (similarity: {score:.4f})")

# Generate sentences for top CUI concepts
prompts = []
for cui_id, cui_name, score in top_k_results:
    # Generate larger sentence group
    prompt = get_prompt(cui_id)

    # Filter best sentences by clustering method
    clusterer = SentenceClusterer()

    # If there are more than 5 sentences select the best 5
    if len(prompt) > 5:
        best_sentences = clusterer.cluster_sentences(prompt, 5)
        print("Original sentences: \n")
        for sentence in prompt:
            print(sentence)
        print("Clusterring output: \n")
        for sentence in best_sentences:
            print(sentence)
    else:
        best_sentences = prompt

    # Add list elements to prompts
    prompts.extend(best_sentences)
