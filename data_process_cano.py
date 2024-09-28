import json
from rdkit import Chem
from rdkit.rdBase import DisableLog
import pubchempy as pcp
from tqdm import tqdm
DisableLog('rdApp.warning')

def smiles_to_name(smiles):
    try:
        compound_smiles = pcp.get_compounds(smiles, 'smiles')
        cpd_id = int(str(compound_smiles[0]).split("(")[-1][:-1])
        c = pcp.Compound.from_cid(cpd_id)
        if isinstance(c.iupac_name, str):
            return c.iupac_name
        else:
            return "None"
    except:
        return "None"

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

def load_dataset():
    file_name = "test_dataset_canolize.json"
    dataset = []  # (product_smiles, materials_smiles, depth)
    with open(file_name, 'r') as f:
        _dataset = json.load(f)
        for p, reaction_trees in tqdm(_dataset.items()):
            intermediates = []
            intermediates_name = []
            p_name = smiles_to_name(p)
            materials_list = reaction_trees['materials']
            for j in range(len(reaction_trees['retro_routes'])):
                if len(reaction_trees['retro_routes'][j]) != reaction_trees['depth']:
                    continue
                else:
                    for i in range(1,len(reaction_trees['retro_routes'][j])):
                        intermediates.append(reaction_trees['retro_routes'][j][i].split('>')[0])
                        intermediates_name.append(smiles_to_name(reaction_trees['retro_routes'][j][i].split('>')[0]))
            dataset.append({
                "product": p,
                "product_name": p_name,
                "intermidiates": intermediates,
                "intermidiates_name": intermediates_name,
                "targets": materials_list,
                "depth": reaction_trees['depth']
            })
    return dataset

if __name__ == '__main__':
    task = load_dataset()
    with open("test_dataset_with_name.json", "w") as json_w:
        json_w.write(json.dumps(task))

