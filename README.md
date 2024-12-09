# ILS-GNN (Interpretable Locally Sparse Graph Neural Network)

## Project Overview
This repository contains the implementation of ILS-GNN (Interpretable Locally Sparse Graph Neural Network), an independent course project by Keyi Li for CPSC583. The codebase includes data preprocessing pipelines, model training workflows, testing procedures, and evaluation metrics.

## Environment Setup

### Using Docker
1. Build the Docker image:
```bash
docker build -t conda_torch:11.7 .
```

2. Run Jupyter Lab with the following configuration:
```bash
docker run --gpus all \
           --ipc=host \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           -p 8877:8888 \
           -u "$(id -u):$(id -g)" \
           -v ~:/home/user \
           -w /home/user \
           -e HOME=/home/user \
           -v /banach1:/banach1 \
           -d conda_torch:11.7 \
           jupyter lab --ip 0.0.0.0 --allow-root --no-browser \
           --IdentityProvider.token=d9e1b5f5ac81dfda
```

## Usage Instructions

### Data Preprocessing
Execute the preprocessing script:
```bash
python data_preprocessing.py
```

### Model Training
Run the training pipeline:
```bash
python Train_Trial.py
```

## Related Models

### SORBET
- Details available in the [published paper](https://pubmed.ncbi.nlm.nih.gov/38260586/)

### DANN
- Implementation reference: [DANN GitHub Repository](https://github.com/fungtion/DANN)

## Data Availability
Please note that the CODEX dataset used in this project is not publicly available due to data privacy restrictions.
