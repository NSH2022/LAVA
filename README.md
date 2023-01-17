# LAVA
LAVA: Granular Neuron-Level Explainable AI for Alzheimer’s Disease Assessment from Fundus Images

## Overview
LAVA is an XAI framework that aims to exploite neuron-level explanation as auxiliary information during learnign process to make a high-resolution AD continuum prediction. This work is supported by the National Science Foundation under Grant No. (NSF 2123809). This repository is provided to reproduce experimental results for this research work. For more details refer to our paper: {link to the paper}
![alt text](Images/github.drawio.png)
## Citation
If you use LAVA in your research, please cite our preliminary work published in arXiv.
{TBD}

## Installation
Follow the following steps to re-produce experimental results 

1- Install [python 3.8.1](https://www.python.org/downloads) 

2- Install [conda](https://docs.conda.io/projects/conda/en/latest/index.html)

3- Create and activate virtual environement. The file `environmental.yml` in this project's root folder contians the list of requirements. Running following command wil create a conda virtual environement and will install all dependencies.
```
conda env create -f environment.yml
```
Activate the environment
```
conda activate LAVA_env
```

For more information on conda please refer to [conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

**Note:** The `environment.yml` file lists a lot of dependencies. But most of them are requirements of just a handful of packages that can be found in `install.sh`. This script also shows steps used to create the conda environment from scratch. Ideally executing this script using `bash install.sh` should produce a similar environment but may result in packages with different version numbers.


## Specify file paths
`LAVA.yml` lists all global constants that will be used in the repository. These constants include:
- `urls`: URLs where the data is to download data from
- `download_dir`: where data will be downloaded to on your machine
- `proc_dir`: where preprocessed data will be saved
- `exp_dir`: where experiment data will be saved
- `train_val_split`: fraction of the provided train data to be used for training (the remaining will be used for validation)
- Names of various `.npy` files used for training/validation/testing. These are all saved in `proc_dir` and will be read by dataloaders.

You will need to modify `download_dir`,`proc_dir`, and `exp_dir` according to your own machine.

4- Download and preprocess data
The raw data in this research is downloaded from [UK Biobank](https://www.ukbiobank.ac.uk/) Resource under application number 48388.

5- Train the model 
```
python -m train
```

his trains a model with default arguments. To see the arguments and their default values run 
```
python -m train --help
```
which should show the following
```
Options:
  --exp_name TEXT               Name of the experiment  [default: default_exp]
  --loss [cross_entropy|focal]  Loss used for training  [default: cross_entropy]
  --num_hidden_blocks INTEGER   Number of hidden blocks in the classifier [default: 2]
```
Any outputs generated during training are saved in `proc_dir/exp_name`

To visualize loss and accuracy curves on tensorboard, go to the experiment directory and run 
```
tensorboard --logdir=./
```

# Testing

The model with the best validation performance during training can be loaded up and evaluated on the test set using
```
python -m test
```
Note that this would work only when default arguments were used during training. For training with non-default arguments use
```
python -m test --exp_name <experiment name> --num_hidden_blocks <number of hidden blocks in the classifier>
```

For default arguments the accuracies on various data subsets should be in the ballpark of the following

|Train|Val|Test|
|---|---|---|
|86.17|85.05|84.65|

**A note on reproducibility:** Reproducing the above numbers is possible only if all of the following are true:
- random seeds in the `preprocess.py` and `train.py` scripts are set to `0`
- `train_val_split` in `LAVA.yml` is set to `0.8`
- default arguments are used during training
