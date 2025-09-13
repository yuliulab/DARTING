#!/user/bin/python
#Author: yingwang
from abc import abstractmethod
import logging
from typing import List, Optional
from functools import partial

import numpy as np
from rdkit import Chem


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def smiles_to_rdkit_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Converts a SMILES string to a RDKit molecule.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        RDKit Mol, None if the SMILES string is invalid
    """
    mol = Chem.MolFromSmiles(smiles)

    #  Sanitization check (detects invalid valence)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None

    return mol
### the modifiers for score module

class ScoreModifier:
    """
    interface to score modifiers
    """
    @abstractmethod
    def __call__(self, x):
        """
        Apply the modifier on x.

        Args:
            x: float or np.array to modify

        Returns:
            float or np.array (depending on the type of x) after application of the distance function.
        """
class ChainedModifier(ScoreModifier):

    """
    from ScoreModifier
    Calls several modifiers one after the other, for instance:
        score = modifier3(modifier2(modifier1(raw_score)))
    """

    def __init__(self, modifiers: List[ScoreModifier]) -> None:
        """
        Args:
            modifiers: modifiers to call in sequence.
                The modifier applied last (and delivering the final score) is the last one in the list.
        """
        self.modifiers = modifiers

    def __call__(self, x):
        score = x
        for modifier in self.modifiers:
            score = modifier(score)
        return score

class AbsoluteScoreModifier(ScoreModifier):
    """
    Score modifier that has a maximum at a given target value, and decreases
    linearly with increasing distance from the target value.
    """

    def __init__(self, target_value: float) -> None:
        self.target_value = target_value

    def __call__(self, x):
        return 1. - np.abs(self.target_value - x)


class GaussianModifier(ScoreModifier):
    """
    Score modifier that reproduces a Gaussian bell shape.
    """

    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        # scale x for a gaussian distribution
        return np.exp(-0.5 * np.power((x - self.mu) / self.sigma, 2.))
### use ScoreModifier to get some spefic function
class LinearModifier(ScoreModifier):
    """
    Score modifier that multiplies the score by a scalar (default: 1, i.e. do nothing).
    """

    def __init__(self, slope=1.0):
        self.slope = slope

    def __call__(self, x):
        #return: x itself
        return self.slope * x


class MinMaxGaussianModifier(ScoreModifier):
    """
    Score modifier that reproduces a half Gaussian bell shape.
    For minimize==True, the function is 1.0 for x <= mu and decreases to zero for x > mu.
    For minimize==False, the function is 1.0 for x >= mu and decreases to zero for x < mu.
    """

    def __init__(self, mu: float, sigma: float, minimize=False) -> None:
        self.mu = mu
        self.sigma = sigma
        self.minimize = minimize
        self._full_gaussian = GaussianModifier(mu=mu, sigma=sigma)

    def __call__(self, x):
        if self.minimize:

            mod_x = np.maximum(x, self.mu)
        else:
            
            mod_x = np.minimum(x, self.mu)
        return self._full_gaussian(mod_x)
MinGaussianModifier = partial(MinMaxGaussianModifier, minimize=True)
MaxGaussianModifier = partial(MinMaxGaussianModifier, minimize=False)



class ScoringFunction:
    """
    Base class for an objective function.

    In general, do not inherit directly from this class. 
    Prefer `MoleculewiseScoringFunction` or `BatchScoringFunction`.
    """

    def __init__(self, score_modifier: ScoreModifier = None) -> None:
        
        """
        `score_modifier: ScoreModifier` means inherit from ScoreModifier class
        Args:
            score_modifier: Modifier to apply to the score. If None, will be LinearModifier()
        """
        self.score_modifier = score_modifier
        self.corrupt_score = -1.0

    @property
    def score_modifier(self):
        return self._score_modifier
    # modify `score_modifier`
    @score_modifier.setter
    def score_modifier(self,modifier:Optional[ScoreModifier]):
        self._score_modifier = LinearModifier() if modifier is None else modifier

    def modify_score(self,raw_score:float)-> float:
        """forward for score_modifie"""
        return self._score_modifier(raw_score)
        
    # 定义抽象方法，用时需要重新写
    @abstractmethod
    def score(self, smiles: str) -> float:
        """
        Score a single molecule as smiles
        """
        raise NotImplementedError

    @abstractmethod
    def score_list(self, smiles_list: List[str]) -> List[float]:
        """
        Score a list of smiles.

        Args:
            smiles_list: list of smiles [smiles1, smiles2,...]

        Returns: a list of scores

        the order of the input smiles is matched in the output.

        """
        raise NotImplementedError



#modify `class ScoreModifier` 1
class MoleculewiseScoringFunction(ScoringFunction):
    """
    inhert from ScoringFunction
    Objective function that is implemented by calculating the score molecule after molecule.
    Rather use `BatchScoringFunction` than this if your objective function can process a batch of molecules
    more efficiently than by trivially parallelizing the `score` function.

    Derived classes must only implement the `raw_score` function.
    """
    def __init__(self, score_modifier:ScoreModifier = None)-> None:
        """
        Args:
            score_modifier: Modifier to apply to the score. If None, will be LinearModifier()
        """

        super().__init__(score_modifier=score_modifier)
    # get scores for smile list
    def score(self, smiles:str):
        return self.modify_score(self.raw_score(smiles))
    def score_list(self, smiles_list: List[str]) -> List[float]:
        return [self.score(smiles) for smiles in smiles_list]
    
    @abstractmethod
    def raw_score(self, smiles: str) -> float:
        """
        Get the objective score before application of the modifier.

        For invalid molecules, `InvalidMolecule` should be raised.
        For unsuccessful score calculations, `ScoreCannotBeCalculated` should be raised.
        """
        raise NotImplementedError



#modify `class ScoreModifier` 2
class BatchScoringFunction(ScoringFunction):
    """
    Objective function that is implemented by calculating the scores of molecules in batches.
    Rather use `MoleculewiseScoringFunction` than this if processing a batch is not faster than
    trivially parallelizing the `score` function for the distinct molecules.

    Derived classes must only implement the `raw_score_list` function.
    """

    def __init__(self, score_modifier: ScoreModifier = None) -> None:
        """
        Args:
            score_modifier: Modifier to apply to the score. If None, will be LinearModifier()
        """
        super().__init__(score_modifier=score_modifier)

    def score(self, smiles: str) -> float:
        return self.score_list([smiles])[0]

    def score_list(self, smiles_list: List[str]) -> List[float]:
        raw_scores = self.raw_score_list(smiles_list)

        scores = [self.corrupt_score if raw_score is None
                  else self.modify_score(raw_score)
                  for raw_score in raw_scores]

        return scores

    @abstractmethod
    def raw_score_list(self, smiles_list: List[str]) -> List[float]:
        """
        Calculate the objective score before application of the modifier for a batch of molecules.

        Args:
            smiles_list: list of SMILES strings to process

        Returns:
            A list of scores. For unsuccessful calculations or invalid molecules, `None` should be given as a value for
            the corresponding molecule.
        """
        raise NotImplementedError



#modify `MoleculewiseScoringFunction` 1
class ScoringFunctionBasedOnRdkitMol(MoleculewiseScoringFunction):
    """
    Base class for scoring functions that calculate scores based on rdkit.Chem.Mol instances.

    Derived classes must implement the `score_mol` function.
    """

    def raw_score(self, smiles: str) -> float:
        mol = smiles_to_rdkit_mol(smiles)

        if mol is None:
            raise InvalidMolecule

        return self.score_mol(mol)

    @abstractmethod
    def score_mol(self, mol: Chem.Mol) -> float:
        """
        Calculate the molecule score based on a RDKit molecule

        Args:
            mol: RDKit molecule
        """
        raise NotImplementedError

#modifiy ``BatchScoringFunction
class ArithmeticMeanScoringFunction(BatchScoringFunction):
    """
    Scoring function that combines multiple scoring functions linearly.
    """

    def __init__(self, scoring_functions: List[ScoringFunction], weights=None) -> None:
        """
        Args:
            scoring_functions: scoring functions to combine
            weights: weight for the corresponding scoring functions. If None, all will have the same weight.
        """
        super().__init__()

        self.scoring_functions = scoring_functions
        number_scoring_functions = len(scoring_functions)

        self.weights = np.ones(number_scoring_functions) if weights is None else weights
        assert number_scoring_functions == len(self.weights)

    def raw_score_list(self, smiles_list: List[str]) -> List[float]:
        scores = []

        for function, weight in zip(self.scoring_functions, self.weights):
            res = function.score_list(smiles_list)
         
            scores.append(weight * np.array(res))
       
        scores = np.array(scores).sum(axis=0) / np.sum(self.weights)

        return list(scores)


class ScoringFunctionWrapper(ScoringFunction):
    """
    Wraps a scoring function to store the number of calls to it.
    """

    def __init__(self, scoring_function: ScoringFunction) -> None:
        super().__init__()
        self.scoring_function = scoring_function
        self.evaluations = 0

    def score(self, smiles):
        self._increment_evaluation_count(1)
        return self.scoring_function.score(smiles)

    def score_list(self, smiles_list):
        self._increment_evaluation_count(len(smiles_list))
        return self.scoring_function.score_list(smiles_list)

    def _increment_evaluation_count(self, n: int):
        # Ideally, this should be protected by a lock in order to allow for multithreading.
        # However, adding a threading.Lock member variable makes the class non-pickle-able, which prevents any multithreading.
        # Therefore, in the current implementation there cannot be a guarantee that self.evaluations will be calculated correctly.
        self.evaluations += n

