import openai
from openai import OpenAI
import json
from tqdm import tqdm

model_version = "gpt-3.5-turbo"
api_key='YOUR_OPENAI_KEY'
prompt_template = """
"Describe the key transition states involved in the synthesis of {{products}} from the intermediates {{intermediates}}. Explain the structural changes and energy barriers for each transition state, and reply to me in a sentence".
"""

def generate_with_prompt(prompt):
    client = OpenAI(api_key=api_key)
    message = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model = model_version,messages = message)
    answer = response.choices[0].message.content
    return answer

if __name__ == '__main__':
    file_name = './data/text_generation_train-data.json'
    num_splits = 10
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
            for intermediate in item['intermidiates_name']:
                intermediates_string += intermediate + ","
            intermediates_string = intermediates_string[:-1]
            prompt = prompt_template.replace('{{intermediates}}', intermediates_string).replace('{{products}}', product_name)
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
        with open("./data/text_train_dataset", "w") as json_w:
            json_w.write(json.dumps(dataset_with_text))
