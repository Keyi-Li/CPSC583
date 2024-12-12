# ILS-GNN (Interpretable Locally Sparse Graph Neural Network)

## Project Overview
This repository contains the implementation of ILS-GNN (Interpretable Locally Sparse Graph Neural Network), an independent course project by Keyi Li for CPSC583. The codebase includes related code and demo code. Please note that the CODEX dataset used in this project is not publicly available due to data privacy restrictions. A demo usecase on MNIST dataset is shown.

## Environment Setup

Most relavent packages can be found in `env/environment.yml`

## Usage Instructions

### CODEX dataset

- Data Preprocessing: Execute the preprocessing script:
```bash
python data_preprocessing.py
```

- Model Training: Run the training pipeline:
```bash
python Train_Trial.py
```

### MNIST-related Dataset (only for demonstration purpose)
```bash
python MNIST_Demo.py
```

## Related Models

### SORBET
- Details available in the [published paper](https://pubmed.ncbi.nlm.nih.gov/38260586/)

### DANN
- Implementation reference: [DANN GitHub Repository](https://github.com/fungtion/DANN)

### LSPIN
- Implementation reference: [LSPIN GitHub Repository](https://github.com/jcyang34/lspin)
