# RETROT: ADVANCING RETROSYNTHETIC PLANNING THROUGH IN-CONTEXT LEARNING AND LARGE LANGUAGE MODEL INTEGRATION

This repository contains an implementation of ["RETROT: ADVANCING RETROSYNTHETIC PLANNING THROUGH IN-CONTEXT LEARNING AND LARGE LANGUAGE MODEL INTEGRATION"](), which is a in-context learning for retrosynthesis plan.

## Fine tune MolT5 
You should download the origin MolT5 model before fine-tuning it at [here](https://huggingface.co/laituan245/molt5-base), then put it at the run_translation folder.

```bash
cd run_translation

# The checkpoint of MolT5 should be saved in the model directory.
python run_translation.py --Fine-tune.txt
```
## Model Training
You should download the Scibert model before testing at [here](https://github.com/allenai/scibert), and choose the first one in PyTorch HuggingFace Models, then put it in the model directory.

```bash
cd RetroT

# Retro Star Zero
python retro_star_0.py

# Retro Star

```

## Reference  
FusionRetro: https://github.com/SongtaoLiu0823/FusionRetro  
3DInfomax: https://github.com/HannesStark/3DInfomax  
MolT5: https://github.com/blender-nlp/MolT5  
Scibert: https://github.com/allenai/scibert  
