import json
import pandas as pd


descriptions_path = '/home/nadun/wd/datasets/wsi_bench/kg_data/descriptions.json'
morphs_path = '/home/nadun/wd/datasets/wsi_bench/custom_kg/relations.csv'
output_path = "/home/nadun/wd/datasets/wsi_bench/kg_data/histology_descriptions_with_features.json"

with open(descriptions_path) as f:
    descriptions = json.load(f)
    print(f"Loaded {len(descriptions)} descriptions from {descriptions_path}")

histologies = {}
data = {}
df = pd.read_csv(morphs_path)

for _, row in df.iterrows():
    histology = row['reference_entity']
    relation = row['relation_name']
    if histology not in histologies:
        histologies[histology] = {}
    if relation not in histologies[histology]:
        histologies[histology][relation] = []
    
    feature = row['related_entity']
    histologies[histology][relation].append(feature)

r = {'found_in': 'found in', 
     'has_growth_pattern': 'has growth pattern', 
     'has_cytological_feature': 'has cytological feature',
     'has_invasion_pattern': 'has invasion pattern', 
     'has_background_feature': 'has background feature', 
     'has_nuclear_feature': 'has nuclear feature', 
     'has_mitotic_activity': 'has mitotic activity'}


for histology, features in histologies.items():
    knowledge_prompt = f"<knowledge>\nHistology: {histology}\n"
    for relation, feature_list in features.items():
        if feature_list:
            # paragraph_parts.append(f"{r.get(relation, relation)}: {', '.join(feature_list)}.")
            knowledge_prompt += f"{r.get(relation, relation)}: {', '.join(feature_list)}\n"
    knowledge_prompt += "</knowledge>"
    print(f"Histology: {histology}\n{knowledge_prompt}\n")
    data[histology] = knowledge_prompt

with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)
print(f"Saved histology descriptions with features to {output_path}")




# data = {}
# for histology_data in descriptions:

#     histology = histology_data['histology']
#     description = histology_data['description']

#     features = histologies.get(histology, {})
#     knowledge = f"""<knowledge>\nHistology: {histology}\nDescription: {description}\nMorphological Features: {features}\n</knowledge>"""
#     data[histology] = knowledge

# #save this as a json file
# with open(output_path, 'w') as f:
#     json.dump(data, f, indent=2)
# print(f"Saved histology descriptions with features to {output_path}")
