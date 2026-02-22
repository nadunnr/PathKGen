import json

descriptions_path = '/home/nadun/wd/datasets/wsi_bench/kg_data/descriptions.json'
morphs_path = '/home/nadun/wd/datasets/wsi_bench/kg_data/top_cuis/top_10_cuis.json'

with open(descriptions_path) as f:
    descriptions = json.load(f)
    print(f"Loaded {len(descriptions)} descriptions from {descriptions_path}")

with open(morphs_path) as f:
    morphs = json.load(f)
    print(f"Loaded {len(morphs)} morphs from {morphs_path}")
data = {}
for key in descriptions:
    histology = key['histology']
    
    features = morphs[histology]['names']
    description = key['description']
    

    text = f"""{histology}: {description} Morphological features include {', '.join(features)}."""
    #add this to the dict
    data[histology] = text

#save this as a json file
output_path = '/home/nadun/wd/datasets/wsi_bench/kg_data/histology_descriptions_with_features.json'
with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)
print(f"Saved histology descriptions with features to {output_path}")
