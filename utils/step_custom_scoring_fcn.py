#! /user/bin/python
# Author:yingwang

import numpy as np
import pandas as pd
#from test import numBridgeheadsAndSpiro
import math
import gzip
import pickle

#import xgboost as xgb
import joblib as skjoblib


# load module from other file
from utils.step_score_function import MoleculewiseScoringFunction
from utils.step_score_function import ArithmeticMeanScoringFunction
from utils.step_score_function import ScoringFunctionBasedOnRdkitMol
from utils.step_score_function import MinMaxGaussianModifier


# rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Mol
from rdkit.six import iteritems
#from rdkit.six.moves import cPickle
import _pickle as cPickle
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import DataStructs

import torch


## define ligand efficancy
class LigandEfficancy(MoleculewiseScoringFunction):
    """
    inhert from MoleculewiseScoringFunction
    """
    def __init__(self, score_modifier,model_path):
        super().__init__(score_modifier=score_modifier)
        with open(model_path,'rb') as handle:
            # load train-well random-sedd-forest model 
            self.rfr = pickle.load(handle)
    
    def raw_score(self,smiles:str)-> float:
        # determine score from self.model and the given string
        m =Chem.MolFromSmiles(smiles)
        mph = Chem.AddHs(m)
        N = mph.GetNumAtoms() - mph.GetNumHeavyAtoms()
        #get fingure print for smile
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),2)
        fp = np.array([fp])
        # fti linear model
        pic50 = self.rfr.predict(fp)
        LE = 1.4*(pic50)/N
        return LE[0]

    
#score for QED
class QED_custom(MoleculewiseScoringFunction):
    def __init__(self, score_modifier):
        super().__init__(score_modifier=score_modifier)
    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        mol = Chem.MolFromSmiles(smiles)
        qed = Descriptors.qed(mol)
        return qed

class LatentDistance(MoleculewiseScoringFunction):
    """
    """
    def __init__(self, smiles_targets, model, score_modifier, n_top=None, agg="mean"):
        super().__init__(score_modifier=score_modifier)
        self.model = model
        self.collate_fn = model.get_collate_fn()
        self.x_targets = self.collate_fn(smiles_targets)
        self.z_targets = self.model.encode(self.x_targets)
        
        if n_top is None:
            self.n_top = len(smiles_targets)
        else:
            self.n_top = int(n_top)
        self.agg = agg
   
    def raw_score(self, smiles):
        """ Ge distance to set """

        x = self.collate_fn([smiles])
        z = self.model.encode(x)
        #求embding差值tensor的范数，评估生成的分子的embedding矩阵和已知的分子的embedding矩阵的差异
        norm = torch.norm(self.z_targets - z, dim=1)
        norm = norm.sort()[0][:self.n_top]
        norm = norm.detach().numpy()
        if self.agg == "mean":
            return norm.mean()
        if self.agg == "max":
            return norm.max()
        if self.agg == "min":
            return norm.min()
    
    def get_all(self, smiles):
        z = self.model.encode(smiles)
        norm = torch.norm(self.z_targets - z, dim=1)
        return norm

## other rdkit property
class SAScorer(MoleculewiseScoringFunction):
    def __init__(self, score_modifier, fscores=None):
        super().__init__(score_modifier=score_modifier)
        if fscores is None:
            # download from https://github.com/jensengroup/String-GA/tree/master
            fscores = '../../data/fpscores.pkl.gz'
        self.fscores = cPickle.load(gzip.open(fscores ))
        outDict = {}
        for i in self.fscores:
            for j in range(1, len(i)):
                outDict[i[j]] = float(i[0])
        self.fscores = outDict
    
    def numBridgeheadsAndSpiro(self, mol, ri=None):
        nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        return nBridgehead, nSpiro


    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        m = Chem.MolFromSmiles(smiles)
        fp = rdMolDescriptors.GetMorganFingerprint(m,2)  # <- 2 is the *radius* of the circular fingerprint
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        for bitId, v in iteritems(fps):
            nf += v
            sfp = bitId
            score1 += self.fscores.get(sfp, -4) * v
        score1 /= nf
        # features score
        nAtoms = m.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
        ri = m.GetRingInfo()
        nBridgeheads, nSpiro = self.numBridgeheadsAndSpiro(m, ri)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1
        
        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.
        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)
        score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5
        sascore = score1 + score2 + score3
        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        # smooth the 10-end
        if sascore > 8.:
            sascore = 8. + math.log(sascore + 1. - 9.)
        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0
        return sascore
        #return self.threshold/np.maximum(sascore, self.threshold)

class LogP(MoleculewiseScoringFunction):
    def __init__(self, score_modifier=None):
        super().__init__(score_modifier=score_modifier)
    def raw_score(self, smiles: str) -> float:
        # determine score from self.model and the given smiles string
        mol = Chem.MolFromSmiles(smiles)
        logp = Descriptors.MolLogP(mol)
        return logp

class MW(MoleculewiseScoringFunction):
    def __init__(self, score_modifier=None):
        super().__init__(score_modifier=score_modifier)
    def raw_score(self, smiles: str) -> float:
        try:
            mw=  Descriptors.ExactMolWt( Chem.MolFromSmiles(smiles) )
            return mw
        except:
            #print('we cant calculate molecular weight', smiles )
            return -1.


