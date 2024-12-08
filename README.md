# CPSC583

This repo contains code for data preprocessing, model training, testing, and evaluation for Keyi Li's independent course project: ILS-GNN (Interpretable Locally Sparse Graph Neural Network).

## 1. Environment and Dependency

To set up the environment necessary for reproducing the test, use docker build -t conda_torch:11.7 .
To run jupyter lab using the docker image, use 
  docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 8877:8888 \
           -u "$(id -u):$(id -g)"
           -v ~:/home/user -w /home/user -e HOME=/home/user -v /banach1:/banach1 \
           -d conda_torch:11.7 jupyter lab --ip 0.0.0.0 --allow-root --no-browser --IdentityProvider.token=d9e1b5f5ac81dfda

## 2. Data preprocessing and Model Training

For data preprocessing, run ./data_preprocessing.py
For model training, run ./model_all.py

The model detail of SORBET can be found at here. https://pubmed.ncbi.nlm.nih.gov/38260586/.
The model detail of DANN can be found at here. https://github.com/fungtion/DANN

## 3. Data
The CODEX data cannot be make public available. 
