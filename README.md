# RetroInText

## Abstact
Development of robust and effective strategies for retrosynthetic planning requires a deep understanding of the synthesis process. A critical step in achieving this goal is accurately identifying synthetic intermediates. Current machine learning-based methods often overlook the valuable context from the overall route, focusing only on predicting reactants from the product, requiring cost annotations for every reaction step, and ignoring the multi-faced nature of molecular, resulting in inaccurate synthetic route predictions. Therefore, we introduce RetroInText, an advanced end-to-end framework based on a multimodal Large Language Model (LLM), featuring in-context learning with TEXT descriptions of synthetic routes. First, RetroInText including ChatGPT presents detailed descriptions of the reaction procedure. It learns the distinct compound representations in parallel with corresponding molecule encoders to extract multi-modal representations including 3D features. Subsequently, we propose an attention-based mechanism that offers a fusion module to complement these multi-modal representations with in-context learning and a fine-tuned language model for a single-step model. As a result, RetroInText accurately represents and effectively captures the complex relationship between molecules and the synthetic route. In experiments on the USPTO pathways dataset RetroBench, RetroInText outperformed state-of-the-art methods, achieving up to a 5% improvement in Top-1 test accuracy, particularly for long synthetic routes. These results demonstrate the superiority of RetroInText by integrating with context information over routes. They also demonstrate its potential for advancing pathway design and facilitating the development of organic chemistry.

![image](./img/framework.png)

## Dependencies
The package depends on the Python==3.8.19:
```bash
dgl==2.1.0
einops==0.7.0
pandas==2.0.3
torch==2.2.0
torch-geometric==2.5.2
torch-scatter==2.1.2
torch-sparse==0.6.18
transformers==4.39.3
openai==1.30.1
scikit-learn==1.3.2
sentencepiece==0.2.0
```

We also provide the environment.yaml, you can create the environment below.
```bash
conda env create -f environment.yaml
```

## Dataset
We use the RetroBench dataset, you can find them in the data directory. You should download the trainning dataset and zinc_stock_17_04_20 file at the following link: [https://zenodo.org/records/14934820](https://zenodo.org/records/14915301) and put them in the folder ```./data```.

## Data processing
You can generate the text information for the train and test dataset as follow:
```bash
cd dataprocess

# Get the text information for the train_dataset.
python train_text_generation.py

# Get the text information for the test_dataset.
python test_text_generation.py
```
You also provide the text information in the ```./data/text_train_dataset``` and ```./data/text_test_dataset```. After getting the text information for the traindataset, you should get the cost as the following step:
```bash
# Get the reaction_cost.
python get_reaction_cost.py

# Get the total_cost.
python get_cost.py
```
We also provide the dataset in the following link: [https://zenodo.org/records/14934820](https://zenodo.org/records/14915301), you can email the corresponding author for dataset.

## Model Training
### Fine tune MolT5 
You should download the origin MolT5 model before fine-tuning it at [https://huggingface.co/laituan245/molt5-base](https://huggingface.co/laituan245/molt5-base), then put it at the run_translation folder, and save the checkpoint in the model directory.

```bash
cd run_translation

# The checkpoint of MolT5 should be saved in the model directory.
python run_translation.py --Fine-tune.txt
```
You also can download the MolT5 model we used in the following link: [https://zenodo.org/records/14915347](https://zenodo.org/records/14915347).

### Train fusion model 
You should download the Scibert model before testing at the following link: [https://zenodo.org/records/14922390](https://zenodo.org/records/14922390), and put it in  ```./model/scibert```. You also should download the best_checkpoint_35epochs.pt at the following link: [https://zenodo.org/records/14923575](https://zenodo.org/records/14923575) and put it in ```./model/Molecule_representation/runs/PNA_qmugs_NTXentMultiplePositives_620000_123_25-08_09-19-52```.

```bash
python Fusion_model.py
```
We also provide value_function_fusion-model.pkl, you can skip the above commands.

## Running the Experiment
To run our model in the RetroBench dataset, followed the setting used in FusionRetro, which set the beam size as 5, random seed as 42:
```bash
# Get the main result
python main.py

# Get the Greedy DFS result
python Greedy_DFS.py
```

## Reference    
```bash
@inproceedings{liu2023fusionretro,
  title={FusionRetro: Molecule Representation Fusion via In-Context Learning for Retrosynthetic Planning},
  author={Liu, Songtao and Tu, Zhengkai and Xu, Minkai and Zhang, Zuobai and Lin, Lu and Ying, Rex and Tang, Jian and Zhao, Peilin and Wu, Dinghao},
  booktitle={International Conference on Machine Learning},
  year={2023}
}

@inproceedings{stark20223d,
  title={3d infomax improves gnns for molecular property prediction},
  author={St{\"a}rk, Hannes and Beaini, Dominique and Corso, Gabriele and Tossou, Prudencio and Dallago, Christian and G{\"u}nnemann, Stephan and Li{\`o}, Pietro},
  booktitle={International Conference on Machine Learning},
  pages={20479--20502},
  year={2022},
  organization={PMLR}
}

@inproceedings{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Cho, Kyunghyun and Ji, Heng},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages={375--413},
  year={2022}
}

@inproceedings{beltagy2019scibert,
  title={SciBERT: A Pretrained Language Model for Scientific Text},
  author={Beltagy, Iz and Lo, Kyle and Cohan, Arman},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={3615--3620},
  year={2019}
}
```
