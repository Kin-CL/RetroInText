# RETROT: ADVANCING RETROSYNTHETIC PLANNING THROUGH IN-CONTEXT LEARNING AND LARGE LANGUAGE MODEL INTEGRATION

This repository contains an implementation of ["RETROT: ADVANCING RETROSYNTHETIC PLANNING THROUGH IN-CONTEXT LEARNING AND LARGE LANGUAGE MODEL INTEGRATION"](), which is a in-context learning for retrosynthesis plan.

## Fine tune MolT5 
You should download the origin MolT5 model before fine-tuning it at [here](https://huggingface.co/laituan245/molt5-base), then put it at the run_translation folder.

```bash
cd run_translation 

python run_translation.py --Fine-tune.txt
```

## Reference  
FusionRetro: https://github.com/SongtaoLiu0823/FusionRetro  
3DInfomax: https://github.com/HannesStark/3DInfomax  
MolT5: https://github.com/blender-nlp/MolT5  
