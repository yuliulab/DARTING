#DARTING
# create conda, python=3.8
conda env create -f environment.yml -n DARTING
conda activate DARTING
* pandas>=1.0.3
* numpy>=1.18.1
* rdkit>=2019.09.3
* joblib>=0.14.1
* scikit-learn>=0.22.1
* python==3.8.19

```
conda install pytorch::pytorch -c pytorch
conda install numpy pandas scikit-learn
pip install rdkit-pypi
```

# train
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


## run ligand binding model
## target A
python run.py train_ligand_binding_model \
--binding_db_path [your path to BindingDB dataset] \ #eg, "/multi_target/step1_extract/step1_key_info.csv"
--uniprot_id "P42345" --output_path "MTOR"
## target B
python run.py train_ligand_binding_model \
--binding_db_path [your path to BindingDB dataset] \
--uniprot_id "Q02750" --output_path "MEK1.pkl"

## run molecular generation

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
 --starting_population ./data/BTK_region3_and_LYN_region2.txt 
