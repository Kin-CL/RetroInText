import random
import json
import logging
import pandas as pd
import torch
from rdkit.Chem import AllChem
from copy import deepcopy
from tqdm import trange
from rdkit import Chem
from rdkit.rdBase import DisableLog
import argparse
import os
from model.Molecule_representation.datasets.bace_geomol_feat import featurize_mol_from_smiles
from icecream import install
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import seaborn

import yaml
from model.Molecule_representation.datasets.custom_collate import *  # do not remove
from model.Molecule_representation.models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from model.Molecule_representation.commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
from model.Molecule_representation.datasets.samplers import *  # do not remove

from torch_geometric.utils import to_dgl
# turn on for debugging C code like Segmentation Faults
import faulthandler
from transformers import AutoModel,AutoTokenizer,T5ForConditionalGeneration
faulthandler.enable()
install()
seaborn.set_theme()
DisableLog('rdApp.warning')

class ValueMLP(nn.Module):
    def __init__(self, n_layers, fp_dim, latent_dim, dropout_rate):
        super(ValueMLP, self).__init__()
        self.n_layers = n_layers
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        logging.info('Initializing value model: latent_dim=%d' % self.latent_dim)

        layers = []
        layers.append(nn.Linear(fp_dim, latent_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(latent_dim, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, fps):
        x = fps
        x = self.layers(x)
        x = torch.log(1 + torch.exp(x))

        return x

class AttentionFusionModel(nn.Module):
    def __init__(self, d_molecule, d_text, d_model):

        super(AttentionFusionModel, self).__init__()
        self.d_molecule = d_molecule
        self.d_text = d_text
        self.d_model = d_model
        self.scale = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        self.W_Q = nn.Linear(d_text, d_model)
        self.W_K = nn.Linear(d_text, d_model)
        self.W_V = nn.Linear(d_molecule, d_model)
    def forward(self, molecule_embedding, text_embedding):

        Q = self.W_Q(text_embedding)
        K = self.W_K(text_embedding)
        V = self.W_V(molecule_embedding)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        attention_weights = F.softmax(attention_scores, dim=-1)
        try:
            output = torch.matmul(attention_weights, V)
            return output
        except:
            output = torch.matmul(attention_weights, V[0].unsqueeze(0))
            for i in range(1, V.shape[0]):
                output = torch.concat((output, torch.matmul(attention_weights, V[i].unsqueeze(0))), 0)
            return output

class FusionModel(nn.Module):
    def __init__(self, d_molecule, d_text, d_model, n_layers, latent_dim, dropout_rate):
        super(FusionModel, self).__init__()
        self.attention_fusion = AttentionFusionModel(d_molecule, d_text, d_model)
        self.value_mlp = ValueMLP(n_layers, d_model, latent_dim, dropout_rate)

    def forward(self, molecule_embedding, text_embedding):
        fused_output = self.attention_fusion(molecule_embedding, text_embedding)
        value_output = self.value_mlp(fused_output)
        return value_output.sum()

def smiles_to_fp(s, fp_dim=600, pack=False):
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[onbits] = 1

    if pack:
        arr = np.packbits(arr)
    fp = 1 * np.array(arr)

    return fp

def value_fn(smi,text):
    if '.' in smi:
        represent = []
        for s in smi.split('.'):
            try:
                g = featurize_mol_from_smiles(s)
                temp = to_dgl(g)
                temp.edata['feat'] = g.edge_attr.long()
                temp.ndata['feat'] = g.z.long()
                v = value_model(temp).to(device)
                represent.append(v)
            except:
                fp = smiles_to_fp(s)
                fp_tensor = torch.from_numpy(fp)
                fp_tensor = fp_tensor.unsqueeze(0)
                fp_tensor = fp_tensor.float().to(device)
                represent.append(fp_tensor)
        t = represent[0]
        for ind in range(1, len(represent)):
            t = torch.concat((t, represent[ind]), 0)
        text = text_embedding(text).to(device)
        t = fusion_model(t, text).to(device)
        return t.sum().item()
    else:
        try:
            g = featurize_mol_from_smiles(smi)
            temp = to_dgl(g)
            temp.edata['feat'] = g.edge_attr.long()
            temp.ndata['feat'] = g.z.long()
            v = value_model(temp).to(device)
            text = text_embedding(text).to(device)
            value = fusion_model(v,text).to(device)
            return value.sum().item()
        except:
            fp = smiles_to_fp(smi)
            fp_tensor = torch.from_numpy(fp)
            fp_tensor = fp_tensor.unsqueeze(0)
            fp_tensor = fp_tensor.float().to(device)
            text = text_embedding(text).to(device)
            value = fusion_model(fp_tensor,text).to(device)
            return value.sum().item()

def get_beam(products, beam_size):
    tokenizer = AutoTokenizer.from_pretrained("model/MolT5", use_fast=False)
    model = T5ForConditionalGeneration.from_pretrained("model/MolT5")
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

def get_arguments():
    args = parse_arguments()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    else:
        config_dict = {}

    if args.checkpoint:  # overwrite args with args from checkpoint except for the args that were contained in the config file
        arg_dict = args.__dict__
        with open(os.path.join(os.path.dirname(args.checkpoint), "model/Molecule_representation/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/train_arguments.yaml"), 'r') as arg_file:
            checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        for key, value in checkpoint_dict.items():
            if key not in config_dict.keys():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value

    return args

def cano_smiles(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
        if tmp is None:
            return None, smiles
        tmp = Chem.RemoveHs(tmp)
        if tmp is None:
            return None, smiles
        [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
        return tmp, Chem.MolToSmiles(tmp)
    except:
        return None, smiles


def load_dataset(split):
    file_name = "%s_dataset.json" % split
    text_file_name = "text_dataset.json"
    dataset = []  # (product_smiles, materials_smiles, depth)
    with open(file_name, 'r') as f:
        _dataset = json.load(f)
    with open(text_file_name, 'r') as f:
        dataset_test = json.load(f)
        for _, reaction_trees in _dataset.items():
            product = reaction_trees['1']['retro_routes'][0][0].split('>')[0]
            product_mol = Chem.MolFromInchi(Chem.MolToInchi(Chem.MolFromSmiles(product)))
            product = Chem.MolToSmiles(product_mol)
            _, product = cano_smiles(product)
            for item in dataset_test:
                if item['product'] == product:
                    text = item['text']
                    intermediates = item['intermediates']
            materials_list = []
            for i in range(1, int(reaction_trees['num_reaction_trees']) + 1):
                materials_list.append(reaction_trees[str(i)]['materials'])
            dataset.append({
                "product": product,
                "targets": materials_list,
                "depth": reaction_trees['depth'],
                "text": text,
                "intermediates":intermediates
            })

    return dataset


def check_reactant_is_material(reactant):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] in stock_inchikeys


def check_reactants_are_material(reactants):
    for reactant in reactants:
        if Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] not in stock_inchikeys:
            return False
    return True

def get_route_result(task):
    max_depth = task["depth"]
    text = task['text']
    # Initialization
    answer_set = []
    queue = []
    queue.append({
        "score": 0.0,
        "routes_info": [{"route": [task["product"]], "depth": 0}],  # List of routes information
        "starting_materials": [],
    })
    while True:
        if len(queue) == 0:
            break
        nxt_queue = []
        for item in queue:
            score = item["score"]
            routes_info = item["routes_info"]
            starting_materials = item["starting_materials"]
            first_route_info = routes_info[0]
            first_route, depth = first_route_info["route"], first_route_info["depth"]
            if depth > max_depth:
                continue
            expansion_mol = first_route[-1]
            for expansion_solution in get_beam(first_route, args.beam_size):
                iter_routes = deepcopy(routes_info)
                iter_routes.pop(0)
                iter_starting_materials = deepcopy(starting_materials)
                expansion_reactants, reaction_cost = expansion_solution[0], expansion_solution[1]
                expansion_reactants = sorted(expansion_reactants)
                try:
                    check_reactants_are_material(expansion_reactants)
                except:
                    continue
                if check_reactants_are_material(expansion_reactants) and len(iter_routes) == 0:
                    answer_set.append({
                        "score": score + reaction_cost - value_fn(expansion_mol,text),
                        "starting_materials": iter_starting_materials + expansion_reactants,
                    })
                else:
                    estimation_cost = 0
                    for reactant in expansion_reactants:
                        if check_reactant_is_material(reactant):
                            iter_starting_materials.append(reactant)
                        else:
                            estimation_cost += value_fn(reactant,text)
                            iter_routes = [{"route": first_route + [reactant], "depth": depth + 1}] + iter_routes

                    nxt_queue.append({
                        "score": score + reaction_cost + estimation_cost - value_fn(expansion_mol,text),
                        "routes_info": iter_routes,
                        "starting_materials": iter_starting_materials
                    })
        queue = sorted(nxt_queue, key=lambda x: x["score"])[:args.beam_size]
    answer_set = sorted(answer_set, key=lambda x: x["score"])
    record_answers = set()
    final_answer_set = []
    for item in answer_set:
        score = item["score"]
        starting_materials = item["starting_materials"]
        answer_keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(m))[:14] for m in starting_materials]
        if '.'.join(sorted(answer_keys)) not in record_answers:
            record_answers.add('.'.join(sorted(answer_keys)))
            final_answer_set.append({
                "score": score,
                "answer_keys": answer_keys
            })
    final_answer_set = sorted(final_answer_set, key=lambda x: x["score"])[:args.beam_size]

    # Calculate answers
    ground_truth_keys_list = [
        set([
            Chem.MolToInchiKey(Chem.MolFromSmiles(target))[:14] for target in targets
        ]) for targets in task["targets"]
    ]
    for rank, answer in enumerate(final_answer_set):
        answer_keys = set(answer["answer_keys"])
        for ground_truth_keys in ground_truth_keys_list:
            if ground_truth_keys == answer_keys:
                return max_depth, rank

    return max_depth, None

def text_embedding(text):
    input_text = text
    token_text = tokenizer(input_text,max_length=512,return_tensors = 'pt').to(device)
    embedding = text_model(token_text['input_ids'],attention_mask = token_text['attention_mask'])['pooler_output']
    return embedding

def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument('--config', type=argparse.FileType(mode='r'), default="/model/Molecule_representation/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/12.yml")
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum numer of epochs to run')
    p.add_argument('--dataset', type=str, default='bace_geomol', help='[qm9, zinc, drugs, geom_qm9, molhiv]')
    p.add_argument('--num_train', type=int, default=-1, help='n samples of the model samples to use for train')
    p.add_argument('--seed', type=int, default=123, help='seed for reproducibility')
    p.add_argument('--num_val', type=int, default=None, help='n samples of the model samples to use for validation')
    p.add_argument('--multithreaded_seeds', type=list, default=[],
                   help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
    p.add_argument('--seed_data', type=int, default=123, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--critic_loss', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--critic_loss_params', type=dict, default={},
                   help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--lr_scheduler', type=str,
                   help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                   help='step every batch if true step every epoch otherwise')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--expensive_log_iterations', type=int, default=100,
                   help='frequency with which to do expensive logging operations')
    p.add_argument('--eval_per_epochs', type=int, default=0,
                   help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
    p.add_argument('--linear_probing_samples', type=int, default=500,
                   help='number of samples to use for linear probing in the run_eval_per_epoch function of the self supervised trainer')
    p.add_argument('--num_conformers', type=int, default=3,
                   help='number of conformers to use if we are using multiple conformers on the 3d side')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='mae_denormalized', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--val_per_batch', type=bool, default=True,
                   help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
    p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--checkpoint', type=str, default = "/model/Molecule_representation/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/best_checkpoint_35epochs.pt",help='path to directory that contains a checkpoint to continue training')
    p.add_argument('--pretrain_checkpoint', type=str, help='Specify path to finetune from a pretrained checkpoint')
    p.add_argument('--transfer_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--frozen_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--exclude_from_transfer', default=[],
                   help='parameters that usually should not be transferred like batchnorm params')
    p.add_argument('--transferred_lr', type=float, default=None, help='set to use a different LR for transfer layers')
    p.add_argument('--num_epochs_local_only', type=int, default=1,
                   help='when training with OptimalTransportTrainer, this specifies for how many epochs only the local predictions will get a loss')

    p.add_argument('--required_data', default=[],
                   help='what will be included in a batch like [dgl_graph, targets, dgl_graph3d]')
    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--use_e_features', default=True, type=bool, help='ignore edge features if set to False')
    p.add_argument('--targets', default=[], help='properties that should be predicted')
    p.add_argument('--device', type=str, default='cpu', help='What device to train on: cuda or cpu')

    p.add_argument('--dist_embedding', type=bool, default=False, help='add dist embedding to complete graphs edges')
    p.add_argument('--num_radial', type=int, default=6, help='number of frequencies for distance embedding')
    p.add_argument('--models_to_save', type=list, default=[],
                   help='specify after which epochs to remember the best model')

    p.add_argument('--model_type', type=str, default='PNA', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')

    p.add_argument('--model3d_type', type=str, default=None, help='Classname of one of the models in the models dir')
    p.add_argument('--model3d_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--critic_type', type=str, default=None, help='Classname of one of the models in the models dir')
    p.add_argument('--critic_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--trainer', type=str, default='contrastive', help='[contrastive, byol, alternating, philosophy]')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--force_random_split', type=bool, default=False, help='use random split for ogb')
    p.add_argument('--reuse_pre_train_data', type=bool, default=False, help='use all data instead of ignoring that used during pre-training')
    p.add_argument('--transfer_3d', type=bool, default=False, help='set true to load the 3d network instead of the 2d network')
    return p.parse_args()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlp_dim', type=int, default=600)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beams size. Default 5. Must be 1 meaning greedy search or greater or equal 5.')
    parser.add_argument('--pretrain_checkpoint', type=str, default='value_function_fusion-model.pkl')
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")
    value_model_mlp = ValueMLP(
        n_layers=args.n_layers,
        fp_dim=args.mlp_dim,
        latent_dim=args.latent_dim,
        dropout_rate=args.dropout
    ).to(device)
    value_model_mlp.load_state_dict(
        torch.load("value_function_mlp.pkl", map_location=device))
    value_model_mlp.eval()

    fusion_model = FusionModel(600, 768, 600, args.n_layers, args.latent_dim, args.dropout)
    fusion_model.load_state_dict(torch.load(args.pretrain_checkpoint))
    fusion_model.to(device)
    fusion_model.eval()

    args_value = get_arguments()
    checkpoint = torch.load("/model/Molecule_representation/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/best_checkpoint_35epochs.pt",
        map_location=device)
    value_model = globals()[args_value.model_type](node_dim=74, edge_dim=4,
                                             **args_value.model_parameters)
    value_model.load_state_dict(checkpoint['model_state_dict'])
    tokenizer = AutoTokenizer.from_pretrained('model/scibert')
    text_model = AutoModel.from_pretrained('model/scibert')
    text_model.to(device)

    stock = pd.read_hdf("zinc_stock_17_04_20.hdf5", key="table")
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    tasks = load_dataset('test')
    overall_result = np.zeros((args.beam_size, 2))
    depth_hit = np.zeros((2, 15, args.beam_size))

    for epoch in trange(0, len(tasks)):
        max_depth, rank = get_route_result(tasks[epoch])
        overall_result[:, 1] += 1
        depth_hit[1, max_depth, :] += 1
        if rank is not None:
            overall_result[rank:, 0] += 1
            depth_hit[0, max_depth, rank:] += 1
    print("overall_result: ", overall_result, 100 * overall_result[:, 0] / overall_result[:, 1], flush=True)
    print("depth_hit: ", depth_hit, 100 * depth_hit[0, :, :] / depth_hit[1, :, :])

