# LAVA
LAVA: Granular Neuron-Level Explainable AI for Alzheimer’s Disease Assessment from Fundus Images

## Overview
LAVA is an XAI framework that aims to exploite neuron-level explanation as auxiliary information during learnign process to make a high-resolution AD continuum prediction. This work is supported by the National Science Foundation under Grant No. (NSF 2123809). This repository is provided to reproduce experimental results for this research work. For more details refer to our paper: {link to the paper}
![alt text](Images/github.drawio.png)
## Citation
If you use LAVA in your research, please cite our preliminary work published in arXiv.
{TBD}

## Installation
The following command uses the file `environmental.yml` in this project's root folder to create a conda virtual environement namend `LAVA_env` with all dependencies installed.
```
conda env create -f environment.yml
```
Activate the environment.
```
conda activate LAVA_env
```

For more information on conda please refer to [conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

**Note:** The `requirements.txt` file lists a lot of dependencies. But most of them are requirements of just a handful of packages necessary for running this project. You might create your conda environment manualy and install the following requiements; it should produce a similar environment but may result in packages with different version numbers.
Required libraries:

torch 1.9.0
torchvision 0.10.0
pandas 1.2.4
numpy 1.20.
sklearn 0.23.2
scipy 1.5.2
scipy 1.9.0
matplotlib 3.6.1
plotly 5.3.1
scipy 1.9.0
seaborn 0.11.2



## Specify file paths
`LAVA.yml` lists all global constants that will be used in the repository. These constants include:
- `urls`: URLs where the data is to download data from
- `download_dir`: where data will be downloaded to on your machine
- `proc_dir`: where preprocessed data will be saved
- `exp_dir`: where experiment data will be saved
- `train_val_split`: fraction of the provided train data to be used for training (the remaining will be used for validation)
- Names of various `.npy` files used for training/validation/testing. These are all saved in `proc_dir` and will be read by dataloaders.

You will need to modify `download_dir`,`proc_dir`, and `exp_dir` according to your own machine.

## Download and preprocess data
The raw data in this research is downloaded from [UK Biobank](https://www.ukbiobank.ac.uk/) Resource under application number 48388.

## Train the model 
```
python -m train
```
Any models generated during training are saved in `models` directory
## Test the model
```
python -m test
```
## Prediction
The follwoing script is the main brain of the LAVA running XAI Knowledge discovery to predict AD continuum.
The model with the best validation performance during training can be loaded up and evaluated on the test set using
```
python -m rediction
```

Any outputs generated during training are saved in `results` directory

For default arguments the validation accuracies of the 5-fold cross validation should be in the ballpark of the following

|Test-Fold 1|Test-Fold 2|Test-Fold 3|Test-Fold 4|Test-Fold 5|
|---|---|---|---|---|
|0.80|0.68|0.80|0.75|0.72|

**A note on reproducibility:** Reproducing the above numbers is possible only if all of the following are true:
- random seeds in the scripts are set to .....
- `train_val_split`....
- default arguments are used during training
