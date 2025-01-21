import json
from standalone_model_numpy import SCScorer
import os
from tqdm import tqdm

project_root = 'C:\\Multi-step'

file_name = 'train-dataset_text.json'
reaction_cost_file = 'reaction_cost.json'
dataset_with_cost = []
with open(file_name, 'r') as f:
    dataset = json.load(f)
with open(reaction_cost_file, 'r') as f:
    reaction_cost = json.load(f)

scscore_model = SCScorer()
scscore_model.restore(
    os.path.join(project_root, 'models', 'full_reaxys_model_1024bool', 'model.ckpt-10654.as_numpy.json.gz'))

for item in tqdm(dataset):
    cost = 0
    p = item['product']
    if '.' in p:
        continue
    p_name = item['product_name']
    intermediates = item['intermediates']
    intermediates_name = item['intermediates_name']
    targets = item['targets']
    depth = item['depth']
    text = item['text']
    for intermediate in intermediates:
        cost += 1-(scscore_model.get_score_from_smi(intermediate)[1]-1)/4
        cost += len(item['targets'])
        try:
            cost += reaction_cost[intermediate]
        except:
            continue
    dataset_with_cost.append({
        'product':p,
        'product_name':p_name,
        'intermediates':intermediates,
        'intermediates_name':intermediates_name,
        'targets':targets,
        'depth':depth,
        'text':text,
        'cost':cost})
with open("training_fusion-model.json", "w") as json_w:
    json_w.write(json.dumps(dataset_with_cost))

