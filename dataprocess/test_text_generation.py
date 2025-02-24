from openai import OpenAI
import json
from tqdm import tqdm

model_version = "gpt-3.5-turbo"
api_key='YOUR_OPENAI_KEY'
prompt_template = """
"Delineate the structural features, functional aspects, and applicable implementations of the molecules {NAME}. The commence with the introduction:"The molecule is ..." and reply to me in a sentence."
"""

def generate_with_prompt(prompt):
    client = OpenAI(api_key=api_key)
    message = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model = model_version,messages = message)
    answer = response.choices[0].message.content
    return answer

if __name__ == '__main__':
    file_name = './data/text_generation_test-data.json'
    with open(file_name, 'r') as f:
        dataset = json.load(f)

    dataset_with_text = []
    for item in tqdm(dataset):
        product = item['product']
        product_name = item['product_name']
        intermediates = item['intermidiates']
        intermediates_name = item['intermidiates_name']
        target = item['targets']
        depth = item['depth']
        intermediates_string = ""
        prompt = prompt_template.replace('{NAME}', product_name)
        text = generate_with_prompt(prompt)
        dataset_with_text.append({
            "product": product,
            "product_name": product_name,
            "intermediates": intermediates,
            "intermediates_name": intermediates_name,
            "targets": target,
            "depth": depth,
            "text": text
        })
    generated_split_file_name = f"test_dataset_with_text.json"
    with open(generated_split_file_name, "w") as json_w:
        json_w.write(json.dumps(dataset_with_text))
