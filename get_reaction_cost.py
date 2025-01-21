import argparse
import random
import json
import torch
import numpy as np
from tqdm import tqdm
# from preprocess import get_vocab_size, get_char_to_ix, get_ix_to_char
# from modeling import TransformerConfig, Transformer, get_padding_mask, get_mutual_mask, get_tril_mask, get_mem_tril_mask
from rdkit import Chem
from rdkit.rdBase import DisableLog
from transformers import AutoModel,AutoTokenizer,T5ForConditionalGeneration

DisableLog('rdApp.warning')


def get_beam(products, beam_size):
    tokenizer = AutoTokenizer.from_pretrained("model\MolT5", use_fast=False)
    model = T5ForConditionalGeneration.from_pretrained("model\MolT5")
    ins = "Please predict the reactant of the product:\n"
    final_beams = []
    inputs = tokenizer(ins + products[-1], return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        top_p=0.9,
        temperature=0.1,
        num_beams=beam_size,
        max_new_tokens=256,
        num_return_sequences=beam_size,
        output_scores=True,
        return_dict_in_generate=True
    )

    for tok, score,i in zip(outputs["sequences"], outputs["sequences_scores"],range(len(outputs["sequences"]))):
        generated_text = tokenizer.decode(tok, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        final_beams.append([generated_text, -score])

    final_beams = list(sorted(final_beams, key=lambda x: x[1]))
    answer = []
    aim_size = beam_size
    for k in range(len(final_beams)):
        if aim_size == 0:
            break
        reactants = set(final_beams[k][0].split("."))
        num_valid_reactant = 0
        sms = set()
        for r in reactants:
            m = Chem.MolFromSmiles(r)
            if m is not None:
                num_valid_reactant += 1
                sms.add(Chem.MolToSmiles(m))
        if num_valid_reactant != len(reactants):
            continue
        if len(sms):
            answer.append([sorted(list(sms)), final_beams[k][1]])
            aim_size -= 1

    return answer


def get_reaction_cost(task):
    reaction, _input = task
    product_list, ground_truth_reactants = _input
    ground_truth_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] for reactant in ground_truth_reactants.split('.')]) 
    for rank, solution in enumerate(get_beam(product_list, args.beam_size)):
        flag = False
        predict_reactants, cost = solution[0], solution[1]
        try:
            answer_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] for reactant in predict_reactants])
        except:
            return reaction, np.inf
        if answer_keys == ground_truth_keys:
            return reaction, cost
        if flag: break
    return reaction, np.inf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max_length', type=int, default=200, help='The max length of a molecule.')
    parser.add_argument('--max_depth', type=int, default=14, help='The max depth of a synthesis route.')
    parser.add_argument('--embedding_size', type=int, default=64, help='The size of embeddings')
    parser.add_argument('--hidden_size', type=int, default=640, help='The size of hidden units')
    parser.add_argument('--num_hidden_layers', type=int, default=3, help='Number of layers in encoder\'s module. Default 3.')
    parser.add_argument('--num_attention_heads', type=int, default=10, help='Number of attention heads. Default 10.')
    parser.add_argument('--intermediate_size', type=int, default=512, help='The size of hidden units of position-wise layer.')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--temperature', type=float, default=1.2, help='Temperature for decoding. Default 1.2')
    parser.add_argument('--beam_size', type=int, default=10, help='Beams size. Default 5. Must be 1 meaning greedy search or greater or equal 5.')
    parser.add_argument("--batch", help="batch", type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reaction_to_input = {}
    file_name = "train_canolize_dataset.json"
    with open(file_name, 'r') as f:
        dataset = json.load(f)
        for _, reaction_tree in dataset.items():
            retro_routes = reaction_tree['retro_routes']
            for retro_route in retro_routes:
                products, reactants = [], []
                for reaction in retro_route:
                    products.append(reaction.split('>')[0])
                    reactants.append(reaction.split('>')[-1])
                for i in range(len(products)):
                    reaction = products[i] + ">>" + reactants[i]
                    if reaction not in reaction_to_input.keys():
                        _input = (products[:i+1], reactants[i])
                        reaction_to_input[reaction] = _input

    tasks = []
    for reaction, _input in reaction_to_input.items():
        tasks.append((reaction, _input))

    reaction_cost = {}
    for task in tqdm(tasks):
        result = get_reaction_cost(task)
        reaction, cost = result
        if cost != np.inf:
            reaction_cost[reaction] = cost.item()
    with open('reaction_cost.json', 'w') as f:
        f.write(json.dumps(reaction_cost, indent=4))



















