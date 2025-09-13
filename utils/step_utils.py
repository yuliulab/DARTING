#! /user/bin/python
#Author:yingwang
import pandas as pd
import re
import os
import glob
import sys
import multiprocessing
import numpy as np
import logging
import itertools
import pickle
import torch
import argparse
from collections import Counter, defaultdict
from typing import Optional, List, Iterable, Collection, Tuple

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors

#modules for score function
from  utils.step_score_function import MoleculewiseScoringFunction 
from utils.step_score_function import ScoringFunctionBasedOnRdkitMol
from utils.step_score_function import ArithmeticMeanScoringFunction
from utils.step_score_function import MinMaxGaussianModifier


# custom scoring
from utils.step_custom_scoring_fcn import QED_custom 
from  utils.step_custom_scoring_fcn import SAScorer
from  utils.step_custom_scoring_fcn import LatentDistance
from  utils.step_custom_scoring_fcn import LogP
from  utils.step_custom_scoring_fcn import MW
from  utils.step_custom_scoring_fcn import LigandEfficancy

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#functions for canoncalizing smiles
def canonicalize(smiles:str,include_stereocenters=True)-> Optional[str]:
    """Canonicalize the SMILE strings with RDKIT
    # The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543
        Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string

    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    
    """
    mol = Chem.MolFromSmiles(smiles)
    # some smiles may can't convert to smiles because of illegal format 
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None
def trim_bond(smiles:str)-> Optional[str]:
    """judge the bond number of trim bond"""
    m = Chem.MolFromSmiles(smiles)
    # 获取分子中的所有化学键
    if m is not None:
        bonds = m.GetBonds()
        types = []
        for bond in bonds:
            
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
          
            bond_type = str(bond.GetBondType())
    
            if bond_type == "TRIPLE":
                print(f"三键:f{smiles} ")
               
                types.append(str(bond_type))
    #    print(f'Atom indices: {begin_atom_idx}-{end_atom_idx}')
    #    print(f'Bond type: {bond_type}')
    #    print()
    return types
def trim_mw(smi):
    """400=< mw <= 600"""
    m=Chem.MolFromSmiles(smi)
    mw =  Descriptors.ExactMolWt( m )
    #get all bonds
    if m is not None:
        count = (mw >=400) &(mw <=600)
        #ture-> 1,False:0
        if count:
            return 1
        else:
            return 0
def trim_cfff(smiles:str)-> Optional[str]:
    """remove smiles those contain cFFF
    smiles: a single smile
    return: the number of CFFF motif
    """
    m=Chem.MolFromSmiles(smiles)
    smarts='FC(F)(F)'
    pattern = Chem.MolFromSmiles(smarts)
    if m is not None:
        counts = []
        matches = m.GetSubstructMatches(pattern) #return a truple(truple,)
        counts.append(len(matches)>0) # append true or False
    return sum(counts)


def canonicalize_list(smiles_list: Iterable[str], include_stereocenters=True) -> List[str]:
    """
    Canonicalize forwar step,run for a list of smiles. 
    Filters out repetitions and removes corrupted molecules.

    Args:
        smiles_list: molecules as SMILES strings
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES strings

    Returns:
        The canonicalized and filtered input smiles.
    """
    canonicalized_smiles = [canonicalize(smiles,include_stereocenters=True) for smiles in smiles_list]
    #remove None
    canonicalized_smiles = [s for s in canonicalized_smiles if s is not None]

    canonicalized_smiles2 = [s for s in canonicalized_smiles if len(trim_bond(s)) < 1]
  
    canonicalized_smiles2 = [s for s in canonicalized_smiles2 if trim_mw(s) == 1]
 
    canonicalized_smiles2 = [s for s in canonicalized_smiles2 if trim_cfff(s) ==0]
    return canonicalized_smiles2

##########
def load_model(model_class,model_definition,device,model_params = None,copy_to_cpu=True):
    """
    Args:
    model_class: class definition od model to load
    model_definition:path to model pickle 
    device:cuda or cpu
    copy_to_cpu:bool
    Return: an VAE model
    """
    # load model,optionally with configuration
    if model_params:
        model = model_class(**model_params).to(device)
    else:
        model = model_class().to(device)
    
    #load stat dict, load parameters in train-well result
    if "cpu" in device:
        model.load_state_dict(torch.load(model_definition,map_location="cpu"))
    else:
        model.load_state_dict(torch.load(model_definition))
    
    #set state to eval mode
    model.eval()
    return model
def save_model(model,output_path,device=None):
    """Save a model
    Args:
    model:a model class
    device:device to return the model to outpath
    output_path(str):fle location to save model to 
    Returns:
    None
    """
    if device is None:
        try:
            device = model.device
        except:
            raise RuntimeError("Module must have a `devive` paremeters if note is specfied")
    #convert to cpu for storage
    model = model.to('cpu')
    # save model to outdir
    torch.save(model.state_dict(),output_path)
    #convert back to proper device
    model = model.to(device)

def load_smiles_from_file(smi_file):
    with open(smi_file) as f:
        return [canonicalize(s.strip()) for s in f]
        
def pick_diverse_set(smiles,n_diverse=100):
    """
    Get diverse set of molecules from a list
    """
    logger = logging.getLogger()


    ms = smiles
    fps = [GetMorganFingerprint(Chem.MolFromSmiles(x),3) for x in ms]
    nfps = len(fps)
    picker = MaxMinPicker()
    logger.debug(f'Len FPS: {nfps}')

    if n_diverse>nfps:
        logger.warning("Exceeded Size")
        n_diverse=nfps

    f = lambda i,j:  1-DataStructs.DiceSimilarity(fps[i],fps[j])
    pickIndices = picker.LazyPick(f,nfps,n_diverse)
    picks = [ms[x] for x in pickIndices]
    
    #top = dff[dff['smiles'].isin(picks)]
    return picks

def get_fingerprint_similarity(scaffolds, return_long = False):
    ms = [Chem.MolFromSmiles(s) for s in scaffolds]
    fps = [FingerprintMols.FingerprintMol(x) for x in ms]
    pairs = itertools.permutations(range(len(ms)),r=2)
    sim = []
    for i,j in pairs:
        s = DataStructs.FingerprintSimilarity(fps[i],fps[j])
        sim.append((i,j,s))
    sim = pd.DataFrame(sim)
    if return_long:
        return sim
    sim_wide = sim.pivot_table(index=0,columns=1, values=2)
    np.fill_diagonal(sim_wide.values,1)
    return sim_wide

def get_fingerprint_similarity_pair(a,b):
    am = Chem.MolFromSmiles(a)
    bm = Chem.MolFromSmiles(b)
    af = FingerprintMols.FingerprintMol(am)
    bf = FingerprintMols.FingerprintMol(bm)
    return DataStructs.FingerprintSimilarity(af,bf)



##score#######



def build_scoring_function( scoring_definition,
                            fscores,
                            opti='gauss',
                            return_individual = False, 
                            vae_model=None):
    """ Build scoring function :
    scoring_definition: a parameter file for define score function
    
    """

    # scoring definition has columns:
    # category, name, minimize, mu, sigma, file, model, n_top
    df = pd.read_csv(scoring_definition, sep=",",header=0)
    scorers = {}

    for i,row in df.iterrows():
        #遍历每行
        name= row['name']

        if row.category == "qed":
            scorers[name] = QED_custom(score_modifier=MinMaxGaussianModifier(mu=row.mu,
                                                                            sigma=row.sigma,
                                                                            minimize=row.minimize))
        elif row.category == "sa":
             scorers[name] = SAScorer( 
                                    score_modifier=MinMaxGaussianModifier(mu=row.mu,
                                                                            sigma=row.sigma,
                                                                            minimize=row.minimize),
                                    fscores=fscores  
                                    )
             
        elif row.category == "latent_distance":
            if vae_model == None:
                raise RuntimeError("No vae class defined.  TODO: you need to fix this")
            # file of smiles for 2 target
            
          
            with open(row.file) as handle:
                smiles_targets = [line.rstrip() for line in handle]
                smiles_targets = canonicalize_list(smiles_targets)
            model = load_model(vae_model, row.model, "cpu") 
            scorers[name] = LatentDistance( smiles_targets=smiles_targets,
                                            model=model,
                                            n_top=None,
                                            agg="mean",
                                            score_modifier=MinMaxGaussianModifier( mu=row.mu,
                                                                                    sigma=row.sigma,
                                                                                    minimize=row.minimize),
                                            ) 
        elif row.category == 'ligand_efficiency':
            scorers[name] = LigandEfficancy(
                                    score_modifier=MinMaxGaussianModifier( mu=row.mu,
                                                                            sigma=row.sigma,
                                                                            minimize=row.minimize),
                                    model_path=row.file
                                    )                           
        else:
            print("WTF Did not understand category: {}".format(row.category))
    #define weight for several types of score, default is all same as 1
    scoring_function = ArithmeticMeanScoringFunction([scorers[i] for i in scorers])

    if return_individual:
        return scorers, scoring_function
    else:
        return scoring_function

#付过滤结果，这里我做了改动
def filter_results(df, mean=True, verbose=False):
    """def filter_results(df, mean=True, verbose=False):
    # qed min
    qed_min = 0.7
    df = df[df['qed']>qed_min]
    if verbose:
        print("After QED: ", df.shape)
    return df
"""

    # qed min
    qed_min = 0.7
    df = df[df['qed']>qed_min]
    if verbose:
        print("After QED: ", df.shape)
    return df


def get_raw_scores(molecules, scores, aggregate_scoring_function=None):
    """
    get scores of individal_scorers
    """
    #pass

    raw_scores = {}
    raw_scores['smiles'] = molecules

    for n, sf in scores.items():
      
        l = []
        for m in molecules:
            try:
              
                s = sf.raw_score(m)
            except:
                # some error occured
                s = -1
            l.append(s)
        raw_scores[n] = l
        #raw_scores[n] = [sf.raw_score(m) for m in molecules]
    df = pd.DataFrame(raw_scores)
    if aggregate_scoring_function:
        df['Aggregate'] = aggregate_scoring_function.score_list(df['smiles'])
        df = df.sort_values('Aggregate', ascending=False) #降序
    return df

def set_random_seed(seed, device):
    """
    Set the random seed for Numpy and PyTorch operations
    Args:
        seed: seed for the random number generators
        device: "cpu" or "cuda"
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'cuda' in device:
        torch.cuda.manual_seed(seed)

def torch_device(arg):
    """ from moses
    """
    if re.match('^(cuda(:[0-9]+)?|cpu)$', arg) is None:
        raise TypeError(
            'Wrong device format: {}'.format(arg)
        )
    if arg != 'cpu':
        splited_device = arg.split(':')
        if (not torch.cuda.is_available()) or \
                (len(splited_device) > 1 and
                 int(splited_device[1]) > torch.cuda.device_count()):
            raise TypeError(
                'Wrong device: {} is not available'.format(arg)
            )
    return arg



