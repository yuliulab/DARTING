# Section 1: Setup Environment
You can follow this instructions to setup the environment  
We use conda here, you can install DARTING via conda yaml files and conda commands:
```
conda env create -f environment.yml -n DARTING
conda activate DARTING
conda install pytorch::pytorch -c pytorch
conda install numpy pandas scikit-learn
pip install rdkit-pypi
```
* pandas>=1.0.3
* numpy>=1.18.1
* rdkit>=2019.09.3
* joblib>=0.14.1
* scikit-learn>=0.22.1
* python==3.8.19

# Section 2: Usage
## Step 1: Train the VAE model in DARTING  
Using the following commands to train the model
```
python run.py train \
	--train_data ./data/guacamol_v1_train.smiles \
	--log_file log.txt \
	--save_frequency 25 \
	--model_save model.pt \
	--n_epoch 200 \
	--n_batch 1024 \
	--debug \
	--d_dropout 0.2 \
	--device cpu
```
Or download the pretrained model weights directly:
```
wget https://github.com/yuliulab/DARTING/releases/download/v1.0/model.pt

```
## Step 2: Train ligand-binding prediction models for downstream generation
Here, we use MTOR and MEK1 as examples.
```
## target A: MTOR
# download BindingDB dataset and save it as csv-format file, eg, step1_key_info.csv
python run.py train_ligand_binding_model \
--binding_db_path [your path to BindingDB dataset] \
--uniprot_id "P42345" --output_path "MTOR.pkl"
## target B: MEK1
python run.py train_ligand_binding_model \
--binding_db_path [your path to BindingDB dataset] \
--uniprot_id "Q02750" --output_path "MEK1.pkl"
```
## Step 3: Identify activity hotspots
```bash
cd hotspot_finder
```
Here, we use MTOR and MEK1 as examples.
Open and run the notebook `get_hotspot_for_MEK1_MTOR.ipynb` to identify activity hotspots for MTOR and MEK1. 
This notebook generates the starting population file `MTOR_region1_MEK1_region3_start_population.txt`, which is used in Step 4.

## Step 4: Run molecular generation
Here, we use MTOR and MEK1 as examples.
```
python run.py generate --model_path ./model.pt \
--scoring_definition ./data/scoring_definition.csv \
 --max_len 100 \
 --n_epochs 50 \
 --mols_to_sample 4096  \
 --optimize_batch_size 512    \
 --optimize_n_epochs 2   \
 --keep_top 4096   \
 --opti gauss   \
 --outF molecular_generation_v4   \
 --device cpu  \
 --save_payloads   \
 --n_jobs 4 \
 --save_frequency 1 \
 --save_individual_scores \
 --debug \
 --starting_population ./data/MTOR_region1_MEK1_region3_start_population.txt
```

