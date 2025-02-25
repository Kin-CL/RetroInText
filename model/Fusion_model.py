import torch
from torch.utils.data import Dataset, DataLoader
import json
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dgl
from transformers import AutoModel,AutoTokenizer
import argparse
import yaml
import os
import torch.nn as nn
import logging
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from Molecule_representation.commons.losses import *
from Molecule_representation.models import *
from Molecule_representation.datasets.samplers import *

bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
types = {'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
               'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
               'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
               'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34}
dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')
chirality = {ChiralType.CHI_TETRAHEDRAL_CW: -1.,
             ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
             ChiralType.CHI_UNSPECIFIED: 0,
             ChiralType.CHI_OTHER: 0}

def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


class MyDataset(Dataset):
    def __init__(self, file_name, tokenizer, text_model, value_model, device):
        with open(file_name, 'r') as f:
            self.dataset = json.load(f)
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.value_model = value_model
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def featurize_mol_from_smiles(self,smiles):
        # filter fragments
        if '.' in smiles:
            raise Exception
            return None

        # filter mols rdkit can't intrinsically handle
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()

        type_idx = []
        atomic_number = []
        atom_features = []
        chiral_tag = []
        neighbor_dict = {}
        ring = mol.GetRingInfo()
        for i, atom in enumerate(mol.GetAtoms()):
            type_idx.append(types[atom.GetSymbol()])
            n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
            if len(n_ids) > 1:
                neighbor_dict[i] = torch.tensor(n_ids)
            chiral_tag.append(chirality[atom.GetChiralTag()])
            atomic_number.append(atom.GetAtomicNum())
            atom_features.extend([atom.GetAtomicNum(),
                                  1 if atom.GetIsAromatic() else 0])
            atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
            atom_features.extend(one_k_encoding(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2]))
            atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
            atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
            atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                                  int(ring.IsAtomInRingOfSize(i, 4)),
                                  int(ring.IsAtomInRingOfSize(i, 5)),
                                  int(ring.IsAtomInRingOfSize(i, 6)),
                                  int(ring.IsAtomInRingOfSize(i, 7)),
                                  int(ring.IsAtomInRingOfSize(i, 8))])
            atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))

        z = torch.tensor(atomic_number, dtype=torch.long)
        chiral_tag = torch.tensor(chiral_tag, dtype=torch.float)

        row, col, edge_type, bond_features = [], [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bonds[bond.GetBondType()]]
            bt = tuple(
                sorted(
                    [bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()])), bond.GetBondTypeAsDouble()
            bond_features += 2 * [int(bond.IsInRing()),
                                  int(bond.GetIsConjugated()),
                                  int(bond.GetIsAromatic())]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
        # bond_features = torch.tensor(bond_features, dtype=torch.float).view(len(bond_type), -1)

        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_type = edge_type[perm]
        # edge_attr = torch.cat([edge_attr[perm], bond_features], dim=-1)
        edge_attr = edge_attr[perm]

        row, col = edge_index
        hs = (z == 1).to(torch.float)
        num_hs = scatter(hs[row], col, dim_size=N).tolist()

        x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
        x2 = torch.tensor(atom_features).view(N, -1)
        x = torch.cat([x1.to(torch.float), x2], dim=-1)

        data = Data(z=x, edge_index=edge_index, edge_attr=edge_attr, neighbors=neighbor_dict, chiral_tag=chiral_tag,
                    name=smiles, num_nodes=N)
        return data

    def represent(self, smi):
        if '.' in smi:
            represent = []
            for s in smi.split('.'):
                g = self.featurize_mol_from_smiles(s)
                temp = to_dgl(g)
                temp.edata['feat'] = g.edge_attr.long()
                temp.ndata['feat'] = g.z.long()
                v = self.value_model(temp).to(self.device)
                represent.append(v)
            t = represent[0]
            for ind in range(1, len(represent)):
                t = torch.concat((t, represent[ind]), 0)
            return t
        else:
            g = self.featurize_mol_from_smiles(smi)
            temp = to_dgl(g)
            temp.edata['feat'] = g.edge_attr.long()
            temp.ndata['feat'] = g.z.long()
            v = self.value_model(temp).to(self.device)
            return v

    def text_embedding(self, text):
        token_text = self.tokenizer(text, max_length=512, return_tensors='pt').to(self.device)
        embedding = self.text_model(token_text['input_ids'], attention_mask=token_text['attention_mask'])['pooler_output']
        return embedding

    def __getitem__(self, idx):
        item = self.dataset[idx]
        molecule_representation = torch.tensor(self.represent(item['product']),dtype=torch.float32)
        text_embedding = torch.tensor(self.text_embedding(item['text']),dtype=torch.float32)
        cost = torch.tensor(item['cost'], dtype=torch.float32)
        return molecule_representation, text_embedding, cost

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
        # layers.append(nn.BatchNorm1d(latent_dim,
        #                              track_running_stats=False))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            # layers.append(nn.BatchNorm1d(latent_dim,
            #                              track_running_stats=False))
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

        output = torch.matmul(attention_weights, V)
        return output

class FusionModel(nn.Module):
    def __init__(self, d_molecule, d_text, d_model, n_layers, latent_dim, dropout_rate):
        super(FusionModel, self).__init__()
        self.attention_fusion = AttentionFusionModel(d_molecule, d_text, d_model)
        self.value_mlp = ValueMLP(n_layers, d_model, latent_dim, dropout_rate)

    def forward(self, molecule_embedding, text_embedding):
        fused_output = self.attention_fusion(molecule_embedding, text_embedding)
        value_output = self.value_mlp(fused_output)
        return value_output

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

def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument('--config', type=argparse.FileType(mode='r'), default="model/Molecule_representation/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/12.yml")
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
    p.add_argument('--checkpoint', type=str, default = "model/Molecule_representation/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/best_checkpoint_35epochs.pt",help='path to directory that contains a checkpoint to continue training')
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    # ===================== model ====================== #
    parser.add_argument('--fp_dim', type=int, default=600)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=128)

    # ==================== training ==================== #
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--pretrain_checkpoint', type=str, default='value_function_fusion-model.pkl')


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('model/scibert')
    text_model = AutoModel.from_pretrained('model/scibert')
    text_model.to(device)

    args_value = get_arguments()
    checkpoint = torch.load("model/Molecule_representation/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52/best_checkpoint_35epochs.pt",
        map_location=device)
    value_model = globals()[args_value.model_type](node_dim=74, edge_dim=4,
                                                   **args_value.model_parameters)
    value_model.load_state_dict(checkpoint['model_state_dict'])

    file_name = "training_fusion-model.json"
    dataset = MyDataset(file_name, tokenizer, text_model, value_model, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    fusion_model = FusionModel(600, 768, 600, args.n_layers, args.latent_dim, args.dropout)
    # fusion_model.load_state_dict(torch.load(args.pretrain_checkpoint))
    fusion_model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=args.lr)

    best_loss = float('inf')
    for epoch in range(args.n_epochs):
        fusion_model.train()
        running_loss = 0.0
        for batch in tqdm(dataloader):
            molecule_embeddings, text_embeddings, costs = batch
            molecule_embeddings = molecule_embeddings.to(device)
            text_embeddings = text_embeddings.to(device)
            costs = costs.to(device)

            optimizer.zero_grad()
            outputs = fusion_model(molecule_embeddings, text_embeddings)
            loss = criterion(outputs, costs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"\nEpoch [{epoch + 1}/{args.n_epochs}], Loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(fusion_model.state_dict(), 'value_function_fusion-model.pkl')
            print(f"Best model saved with loss: {best_loss:.4f}")
