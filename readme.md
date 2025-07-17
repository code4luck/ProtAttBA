# Sequence-Only Prediction of Binding Affinity Changes: A Robust and Interpretable Model for Antibody Engineering

## Introduction

ProtAttBA is a protein language model that predicts binding affinity changes based solely on the sequence information of antibody-antigen complexes.

## Usage

### Install

1. Create conda environment 

```bash
conda create -n protab python==3.10
```

2. Install environment dependency

```bash
# activate environment
source activate protab
# install pytorch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia 
(or use pip: pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118)

# install dependencies
pip install -r ./requirments.txt
```

### dataset
The source data are located in the ```source_data```(SKEMPI[1], AB-Bind[2]).

[1] Moal I H, Fernández-Recio J. SKEMPI: a structural kinetic and energetic database of mutant protein interactions and its use in empirical models[J]. Bioinformatics, 2012, 28(20): 2600-2607.

[2] Sirin S, Apgar J R, Bennett E M, et al. AB‐bind: antibody binding mutational database for computational affinity predictions[J]. Protein Science, 2016, 25(2): 393-409.

Cross validation dataset is located in the  ```cross_validation/data/csv``` folder  (Using the dataset processed by: [Jin et al., 2024](https://github.com/ruofanjin/AttABseq)). The results are located in ```cross_validation/results```

Sequence identity dataset is located in the ```seq-identity_sig-mul/data/identity_data``` folder (Use MMseqs with ```--min-seq-id 0.3```). The results are located in the ```seq-identity_sig-mul/result_idt```.

Single mutation training and multi-mutation testing dataset is located in the ```seq-identity_sig-mul/data/sigmul_data``` folder. The results are located in the ```seq-identity_sig-mul/result_sigmul```.

### Training

```bash
# For cross validation you can use cross_validation/scripts/bash_cross-validation.sh with different args
cp bash_cross-validation.sh ../
bash bash_cross-validation.sh 

# For Sequence identity you can use seq-identity_sig-mul/scripts/bash_seq_identity.sh with different args
cp bash_seq_identity.sh ../ 
bash bash_seq_identity.sh

# For Single mutation training and multi-mutation testing you can use seq-identity_sig-mul/scripts/bash_seq_sigmul.sh with different args
cp bash_seq_sigmul.sh ../ 
bash bash_seq_sigmul.sh
```

### Evaluation

```bash
# For evaluation you can use the seq-identity_sig-mul/eval.py to predict the result by change the args
python eval.py
```
