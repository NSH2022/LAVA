# LAVA
LAVA: Granular Neuron-Level Explainable AI for Alzheimer’s Disease Assessment from Fundus Images

## Overview
LAVA is an XAI framework that aims to exploite neuron-level explanation as auxiliary information during the learning process to make a high-resolution AD continuum prediction. This work is supported by the National Science Foundation under Grant No. (NSF 2123809). This repository is provided to reproduce experimental results for this research work. For more details refer to our paper: {link to the paper}
![alt text](Images/github.drawio.png)
## Citation
If you use LAVA in your research, please cite our preliminary work published in [arXiv](https://arxiv.org/pdf/2302.03008.pdf).
Code release DOI: [![DOI](https://zenodo.org/badge/590054290.svg)](https://zenodo.org/badge/latestdoi/590054290)

## Installation
The following command uses the file `environmental.yml` in this project's root folder to create a conda virtual environement named `LAVA_env` with all dependencies installed.
```
conda env create -f environment.yml
```
Activate the environment.
```
conda activate LAVA_env
```

For more information on conda please refer to [conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

**Note:** The `requirements.txt` file lists a lot of dependencies. But most of them are requirements of just a handful of packages necessary for running this project. You might create your conda environment manualy and install the following requiements; it should produce a similar environment but may result in packages with different version numbers.

Required libraries:<br>
torch 1.9.0<br>
torchvision 0.10.0<br>
pandas 1.2.4<br>
numpy 1.20.2<br>
sklearn 0.23.2<br>
matplotlib 3.6.1<br>
plotly 5.3.1<br>
scipy 1.9.0<br>
seaborn 0.11.2<br>


## Specify file paths
You will need to modify the following paths according to your own machine.
- `data_dir`: where the segemented input images are stored
- `model_dir`: where the trained models will be saved
- `out_dir`: where the results of the experiment will be saved.

## Download and preprocess data
The raw data in this research is downloaded from [UK Biobank](https://www.ukbiobank.ac.uk/) Resource under application number 48388. 

##
The subject characteristics code from the UK Biobank cohort can be run in the notebook below.
```
AD_NC_Subject_Characteristics.ipynb
```
The notebook requires UK Biobank application csv's, which entails the subject demographics, clinical measures, and statistical tests (t-test and chi-square). 
##

## Data-preprocessing
The data will be cropped, vessel-segmented, along with post-processed morphological features. It is recommended to follow the original authors github of the AutoMorph pipeline https://github.com/rmaphoh/AutoMorph for the pre-processing code, details, and guidelines. 

The data in our project was manually selected rather than automated selection. In accordance, run the AutoMorph pipeline as follows:

- Run only stages M0 and M1 (pre-processing and quality assessment) by commenting all later blocks. The run command is: sh run.sh 
- The output folder will contain M1/Bad_quality and M1/Good_quality. Move all data from Bad_quality to Good quality.
- Next, run stage all later stages of the pipeline (M2, M3) by commenting out stage M0 and M1, and proceeding onwards with sh run.sh 
- The output folders will be M2 and M3 with the output results and morphological feature measurements

The data used in this project stems from
M2/binary_vessel/resize 
M3/binary_feature_measurements.csv 


## Model training and evaluation
Our code will employ a five-fold stratified cross validation for AD vs NC. binary classification. Please refer to

```
cv_train_eval.ipynb
```
A separate testing script independent of the cross-validation script is provided by loading the model and data dir:
```
test.ipynb
```
For our dataset, the range of accuracy values generally ranges between 68-80% for each fold, for example,

|Test-Fold 1|Test-Fold 2|Test-Fold 3|Test-Fold 4|Test-Fold 5|
|---|---|---|---|---|
|0.80|0.68|0.80|0.75|0.72|

The random seed in the code will generate consistency in the data randomization, but the exact result may vary due to stochastic optimization. The arguments provided in the script were the default arguments used for the paper. 

## Feature Attribution
A visualization script using Guided Backpropagation is provided in
```
GBP_visualizations.ipynb
```
The script requires a manual image path (preprocessed), vessel image path, and corresponding model. 

## Prediction
The following script is the main brain of the LAVA running XAI Knowledge discovery to predict AD continuum.
The models trained through 5-fold cross-validation paradigm are loaded up and evaluated on the test sets. During the evaluation process, top-k critical neurons are identified and ensembled through the network. 
```
python -m prediction --r <number of latent subclasses> --d <number of adjacent neighbors> --k <number of top-k neurons to select at each layer> --s <number of pruning neurons at each step of RFE> --data_dir  <the path to the data directory> --model_dir <the path to the models directory> --out_dir <the path to the results directory>
```
Alternatively, you can run the script with all the default parameters.
```
python -m prediction
```

Any outputs generated during training are saved in `results` directory.


