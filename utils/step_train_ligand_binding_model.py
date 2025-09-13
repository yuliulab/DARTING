#! /user/bin/python
#Author:yingwang
import pickle
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
from pathlib import Path

import logging

def train_ligand_binding_model(target_unit_pro_id,
                               binding_db_path,output_path):
    binddb = pd.read_csv(binding_db_path,sep=",",
                         header = 0,low_memory=False,
                        on_bad_lines='skip')
    d=  binddb[binddb['UniProt (SwissProt) Primary ID of Target Chain']==target_unit_pro_id]
    #
    d = d[['Ligand SMILES','IC50 (nM)']]
    d.columns = ['smiles','ic50']
    #
    d = d.dropna(axis= 0 ,subset=['ic50'],how='all')
    d.to_csv(f"{target_unit_pro_id}_brefore1_traning_data.csv")
    logging.debug(f"Number of obs:{d.shape[0]}")
    logging.debug(f"d.head()")
    vs = []
    left_s = []
    for s,i in d[['smiles','ic50']].values:
        try:
            v= float(i)
        except ValueError:
            v = float(i[1:])
        t = pd.Series([v]).dropna().min()
        if int(t) != 0:
            #smaller in ic50, greater in t
            t = -np.log10(t*1E-9) 
            vs.append(t)
            left_s.append(s)
    
    my_data = pd.DataFrame({
"smiles": left_s,
"metric_value":vs
})
    #remove duplicate smile in data
    my_data= my_data.dropna(axis=0,subset = ["metric_value"])#remove na
    my_data.sort_values(by='metric_value', inplace=True,ascending   =False)
    my_data= my_data.drop_duplicates(subset='smiles') 
    

    logging.debug(f"Number of obs:{d.shape[0]}")

    if my_data.shape[0] < 10:
        logging.info("Less than 10 compound-target pairs, can't fitting a model")
        return 1
    my_data.to_csv(f"{target_unit_pro_id}_traning_data.csv")
    
    # deal smiles: get print
    fps = []
    values = []
    for x,y in my_data[['smiles','metric_value']].values:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x),2)
        except:
            continue
        fps.append(fp)
        values.append(y)

    X = np.array(fps)
    y = np.array(values)
    regr = RandomForestRegressor(n_estimators=1000,random_state=0,n_jobs=-1)
    regr.fit(X,y)
    ## cross validation
    kfold = KFold(n_splits=5)
    rf_mse = cross_val_score(estimator = regr, X = X,y=y, 
                        scoring = 'neg_mean_squared_error', cv = kfold, verbose = 1, n_jobs=6)

    if output_path is None:
        output_path =  f'{target_unit_pro_id}_rfr_ligand_model.pt'
    print('running random forest,mean of MSE: %.4f,std: %.4f' %(rf_mse.mean(), rf_mse.std()))
    rec = pd.DataFrame.from_dict({
        "MSE_avg":rf_mse.mean(), 
        "MSE_std":rf_mse.std(),
        "MSE":rf_mse},orient='index')
    rec.to_csv(f"{target_unit_pro_id}_train-log.csv")
    #save
    with open(f"{output_path}.pkl", 'wb') as handle:
        s = pickle.dump(regr, handle)

    return 1


