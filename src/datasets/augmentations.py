import random
import numpy as np
from copy import copy
from abc import ABC, abstractmethod
from typing import List, Union


class SeqTransformationInterface(ABC):
    """
    Sequence transformation interface.
    """

    @staticmethod
    def check_probability(p: float) -> float:
        if not 0 <= p <= 1:
            raise ValueError(f"p should be a float number from diapasone [0, 1]!")
        return p
    
    @abstractmethod
    def transform(self, state: dict) -> dict:
        return state


class FieldsShuffler(SeqTransformationInterface):
    """
    Randomly shuffle fields values (ie {"one": 1, "two" 2} -> {"one": 2, "two" 1}).
    """

    def __init__(self, fields_to_shuffle: List[str], p: float = 0.5):
        """
        :param fields_to_shuffle: list of fields to use for shuffling 
            (fields shuffle not value of fields shuffle)
        :param p: probability threshold of shuffle
        """
        self.fields = fields_to_shuffle
        self.p = self.check_probability(p)

    def transform(self, state: dict) -> dict:
        if np.random.uniform() >= self.p:
            return state

        new_state = dict()
        cols = list(self.fields)
        random.shuffle(cols)
        for original, shuffled in zip(self.fields, cols):
            new_state[shuffled] = state[original]
        # fill with other values
        for col in state.keys():
            if col not in new_state:
                new_state[col] = state[col]
        return new_state


class SeqElementsShuffler(SeqTransformationInterface):
    """
    Randomly shuffle sequence values.
    """
    
    def __init__(self, field: str, p: float = 0.5):
        self.field = field
        self.p = self.check_probability(p)

    def transform(self, state: dict) -> dict:
        if np.random.uniform() >= self.p:
            return state
        
        random.shuffle(state[self.field])
        return state


class SeqValuesMapper(SeqTransformationInterface):
    """
    Randomly map values of sequence to value from scheme.
    """

    def __init__(self, feature: str, map_scheme: dict, p: float = 0.5):
        """
        :param feature: state field to use
        :param map_scheme: dict with data mappings, ie {0: 1, ...} means token 0 will be maped to 1 with some probability
        :param p: probability threshold of changing whole sequence 
            (if one value or more is changed then the whole sequence is changed)
        """
        self.feat: dict = feature
        self.elem_map: dict = map_scheme
        self.p: float = self.check_probability(p)

    def _find_prob_for_seq(self, seq: List[int]) -> float:
        """
        Compute probability of change for each token based on number of tokens founded in a sequence.

        :param seq: list of tokens (integers)
        :return: probability threshold of changing character
        """
        n = sum(1 for t in seq if t in self.elem_map)
        prob = 1 - np.power(1 - self.p, 1 / n, dtype=np.float32) if n > 0 else self.p
        return prob
    
    def transform(self, state: dict) -> dict:
        seq = state[self.feat]
        aug_seq = copy(seq)
        prob = self._find_prob_for_seq(seq)
        
        for idx, token in enumerate(seq):
            if token in self.elem_map:
                if np.random.uniform() < prob:
                    aug_seq[idx] = self.elem_map[token]

        state[self.feat] = aug_seq
        return state


class CombineSeqs(SeqTransformationInterface):
    """
    Combine fields of state into one field glued with token.
    """

    def __init__(self, cols: List[str], out_col: str, glue_token = None):
        """
        :param cols: list of features from state to use for combining into one
        :param out_col: name of feature to use for storing result
        :param glue_token: token to use for gluing features (line separator in join functions for list of strings)
        """
        if glue_token is not None and not isinstance(glue_token, (int, float)):
            raise ValueError("'glue_token' should be a number")

        self.cols: List[str] = cols
        self.out: str = out_col
        self.glue_token: list = [glue_token] if glue_token is not None else []

    def transform(self, state: dict) -> dict:
        res = []
        not_empty_items = [state[c] for c in self.cols if state[c]]
        last_idx = len(not_empty_items) - 1
        for idx, item in enumerate(not_empty_items):
            res.extend(item)
            if idx != last_idx:
                res.extend(self.glue_token)
        state[self.out] = res
        return state


class Compose(SeqTransformationInterface):
    """
    Execute passed transformations sequentialy.
    """

    def __init__(self, *args):
        self.transfom_sequence = args
    
    def transform(self, state: dict) -> dict:
        for augg in self.transfom_sequence:
            state = augg.transform(state)
        return state


__all__ = (
    "SeqTransformationInterface", "FieldsShuffler", "SeqElementsShuffler", 
    "SeqValuesMapper", "CombineSeqs", "Compose",
)
