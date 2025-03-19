# PhosPPI-SEQ
PhosPPI-SEQ: A Deep Learning Network for Predicting the Function of Phosphorylation Sites in Regulating Protein - Protein Interactions (PPI).

PhosPPI-SEQ requires only the raw protein sequences of the concerned PPI, and concerned phosphosites as input, avoiding the problem of losing information for a large amount of phosphosites due to the limited number of protein crystal structures. It is computationally efficient and has a broader scope of application for biologists.
![image](https://github.com/zjliangchina/PhosPPI-SEQ/blob/main/IMG/PhosPPI_SEQ_model.png)

# System requirement
PhosPPI-SEQ is develpoed under Linux environment with:
* Python (3.8.13):
    - keras==2.4.3
    - networkx==3.1
    - scipy==1.10.1 
    - scikit-learn==0.24.2
    - numpy==1.19.5
    - tensorflow==2.4.0
* You can install the dependent packages by the following commands:
    - pip install python==3.8.13
    - pip install numpy==1.19.5
    - pip install keras==2.4.3
    - pip install tensorflow==2.4.0

# Dataset
We provide phosphosite data, collected from four databases - PSP, PTMint, iPTMnet and PTMD - detailing information on phosphosites and their regulation of molecular functions, biological processes, and intermolecular interactions, as well as their associations with disease information.

# Predict test data
If you want to use this model to predict the function of phosphorylation sites in regulating Protein-Protein Interactions (PPI), please run the following command (The test data required for running can be found in the "Datasets/example" folder; you can also provide your own data, but the data format needs to be the same as that of the test data):
- python ./src/model/PhosPPI_SEQ.py
