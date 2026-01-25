import re
# import ollama
from graphs import Neo4jQuery
from utils import entity_linking, linked_entities_with_relations, identify_entities


def get_response(f):
    response = ollama.chat(
        'deepseek-r1:8b',
        messages=[{
            'role': 'user',
            'content': f"""
                Here are the related facts of a disease that could be possible to 
                present in a patient whose whole slide image is provided to us.
            """
        }],
    )

    msg = response['message']['content']
    clean_text = re.sub(r"<think>.*?</think>", "", msg, flags=re.DOTALL).strip()
    return clean_text


if __name__ == "__main__":
    # NEO4J_URI = "bolt://localhost:7687"
    # NEO4J_USER = "neo4j"
    # NEO4J_PASSWORD = "11111111"  # Change this to your Neo4j password
    # query = Neo4jQuery(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    text = "A 45-year-old female patient presents with a palpable mass in her left breast. " \
           "Mammography reveals a suspicious lesion categorized as BI-RADS 4. " \
           "A core needle biopsy is performed, and histopathological examination shows invasive ductal carcinoma. " \
           "Immunohistochemistry indicates that the tumor is estrogen receptor-positive, progesterone receptor-positive, " \
           "and HER2-negative. The patient has no significant family history of breast cancer. " \
           "Based on these findings, the oncology team is considering neoadjuvant chemotherapy followed by surgery."
    print("Original Text:")
    doc, linker = entity_linking(text)
    print("Linked")
    # df_with_rels = linked_entities_with_relations(doc, linker)   #API is faster
    modified_text, entity_list = identify_entities(doc, linker)
    print("Modified Text with CUIs:")
    print(modified_text)

    