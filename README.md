# RETROT: ADVANCING RETROSYNTHETIC PLANNING THROUGH IN-CONTEXT LEARNING AND LARGE LANGUAGE MODEL INTEGRATION

This repository contains an implementation of ["RETROT: ADVANCING RETROSYNTHETIC PLANNING THROUGH IN-CONTEXT LEARNING AND LARGE LANGUAGE MODEL INTEGRATION"](), which is a in-context learning for retrosynthesis plan.

## Fine tune MolT5 
You should download the origin MolT5 model before fine-tuning it at [here](https://huggingface.co/laituan245/molt5-base), then put it at the run_translation folder, and save the checkpoint in the model directory.

```bash
cd run_translation

# The checkpoint of MolT5 should be saved in the model directory.
python run_translation.py --Fine-tune.txt
```
You also can download the MolT5 model we used [here](https://drive.google.com/drive/folders/15qYBvDtfoWtVteaxav14VrPCBVQwRWBa).

## Model Training
You should download the Scibert model before testing at [here](https://github.com/allenai/scibert), and choose the first one in PyTorch HuggingFace Models, then put it in the model directory. Download the train_dataset and the zinc_stock_17_04_20 file [here](https://drive.google.com/drive/folders/15qYBvDtfoWtVteaxav14VrPCBVQwRWBa).

```bash
# Data process

python to_canilize.py --dataset train
python to_canolize.py --dataset test

# Retro Star Zero Search
python retro_star_0.py  --beam_size 5

# Retro Star Search w/o text in test phase
python get_reaction_cost.py
python get_cost.py
python fusion_data.py
python MLP-text.py
#We also provide value_function_mlp.pkl, you can skip the above commands
python retro_star_molt5-mlp.py

# Retro Star Search w/o text in test phase
python get_reaction_cost.py
python get_cost.py
python Fusion_model.py
#We also provide value_function_fusion-model.pkl, you can skip the above commands
python retro_star_molt5.py
```

## Reference  
FusionRetro: https://github.com/SongtaoLiu0823/FusionRetro  
3DInfomax: https://github.com/HannesStark/3DInfomax  
MolT5: https://github.com/blender-nlp/MolT5  
Scibert: https://github.com/allenai/scibert  
