import re
import ollama

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


