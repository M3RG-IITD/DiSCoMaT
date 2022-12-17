# Tabbie: Tabular Information Embedding
This repository includes scripts for Tabbie(Tabular Information Embedding) model. 
The link to the paper is as follows.
https://arxiv.org/pdf/2105.02584.pdf
Run this on Quadro P5000.

## (Setup 1): update cuda version from 10.0 to 10.1 (for AWS deep learning ami)
```
# https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-base.html
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.1 /usr/local/cuda
```

## Setup 2: create conda env (default env name: "table_emb_dev")
```
git clone https://github.com/SFIG611/tabbie.git
cd tabbie
conda env create --file env/env.yml
conda activate table_emb_dev
conda install pytorch==1.5.1 cudatoolkit=10.1 -c pytorch
```

## Setup 3: installing apex (gcc==6.5, cuda==10.1.168, cudnn==7.6-cuda_10.1)
```
conda activate table_emb_dev
mkdir -p third_party
git clone -q https://github.com/NVIDIA/apex.git third_party/apex
cd third_party/apex
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" ./
```
## Setup 4: downloading MATSCIBERT
```
cd tabbie
Download files from https://huggingface.co/m3rg-iitd/matscibert/tree/main and place it in folder named saved_matscibert
```

# Create data required for training and inference in TABBIE
```
python create_data_for_tabbie.py
mkdir ./data/ft_cell/discomat_data/discomat_all_csv
cd data/ft_cell/discomat_data/discomat_all_csv
cp -a ./../discomat_train_csv/. ./../discomat_val_csv/. ./../discomat_test_csv/. .
cd ../../../../
```

# Make directories to save logits
```
mkdir row_col_gid_logit
mkdir edge_logit
```

# Seeds to replicate the result
```
Edit the seeds present in tabbie/exp/ft_cell/cell.jsonnet file:
1. For TABBIE_ADAPTED : [[seed used in TABBIE(comment out the seeds)], ["random_seed": 20050, "numpy_seed": 2005, "pytorch_seed": 200,], ["random_seed": 2, "numpy_seed": 2, "pytorch_seed": 2,]
2. For TABBIE_ADAPTED : [[seed used in TABBIE(comment out the seeds)], ["random_seed": 20050, "numpy_seed": 2005, "pytorch_seed": 200,], ["random_seed": 30000, "numpy_seed": 6000, "pytorch_seed": 1200,]]

Reason for using different 3rd seed: Obtained better ressults from baseline, wanted to be fair to the baselines by obtaining best possible results.
```

# For TABBIE
```
Edit the tabbie/exp/ft_cell/cell.jsonnet file:
1. Change the name of the model present in the dataset reader, tokenizer and bert_embedder of the model to "bert-base-uncased"
2. Comment out the requires_grad=true in bert_embedder, as done in TABBIE.
3. Comment line 385-387, paramter grouping of BERT Embedder for controlling its lr. Not required in TABBIE.
```

# For TABBIE_ADAPTED
```
Edit the tabbie/exp/ft_cell/cell.jsonnet file:
1. Change the name of the model present in the dataset reader, tokenizer to "allenai/scibert_scivocab_uncased", and bert_embedder of the model to './saved_matscibert'.
2. Make sure requires_grad=true in bert_embedder, as done in TABBIE.
3. Make sure paramter grouping of BERT Embedder is on with given lr in line 385-387.
```

# Training
```
conda activate table_emb_dev
cd tabbie

python train.py --train_csv_dir ./data/ft_cell/discomat_data/discomat_train_csv --train_label_path ./data/ft_cell/discomat_data/discomat_train_label.csv --valid_csv_dir ./data/ft_cell/discomat_data/discomat_val_csv --valid_label_path ./data/ft_cell/discomat_data/discomat_val_label.csv
```

# Inference from best validation model
```
python pred.py --test_csv_dir ./data/ft_cell/discomat_data/discomat_test_csv --test_label_path ./data/ft_cell/discomat_data/discomat_test_label.csv --model_path ./out_model/model.tar.gz
```
# Checking the results on test dataset
```
python pred_from_tabbie.py
```
