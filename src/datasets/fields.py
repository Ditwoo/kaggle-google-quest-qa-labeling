import numpy as np
from pandas import DataFrame
import torch
from torch.utils.data import Dataset
from typing import List
from .utils import pad_sequences
from .augmentations import SeqTransformationInterface


class TextDataset(Dataset):
    def __init__(self, data, targets=None, shuffle_prob: float = 0.0):
        self.data = data
        self.targets = targets
        self.shuffle_prb = shuffle_prob
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = np.array(self.data[idx])

        if np.random.uniform(0.0, 1.0) < self.shuffle_prb:
            np.random.shuffle(seq)

        if self.targets is None:
            return seq
        lbl = self.targets[idx]
        return seq, lbl


class FieldsDataset(Dataset):
    def __init__(self, 
        df: DataFrame, 
        feature_cols: List[str], 
        target: List[str], 
        transforms: SeqTransformationInterface, 
        field: str = None
    ):
        self.df: DataFrame = df
        self.features: List[str] = feature_cols
        self.target: List[str] = target
        self.augs: SeqTransformationInterface = transforms
        self.field = field

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        index = self.df.index[idx]
        state = {c: self.df.at[index, c] for c in self.features}
        state = self.augs.transform(state) if self.augs is not None else state
        features = state[self.field] if self.field is not None else state
        target = [self.df.at[index, c] for c in self.target] 
        return features, target


class SequencesCollator:
    """
    Pack sequences by it's lengths.

    Main case of usage:

    ```
    >>> train_data = ClassificationDataset(train_seqs, train_labels)
    >>> train_loader = DataLoader(train_data, ..., collate_fn=SequencesCollator())
    ```
    ```
    >>> test_data = ClassificationDataset(test_seqs)
    >>> test_loader = DataLoader(test_data, ..., collate_fn=SequencesCollator(test=True))
    ```

    """

    def __init__(self, is_test: bool = False, percentile: int = 100, max_len: int = 1_000):
        self.is_test = is_test
        self.percentile = percentile
        self.max_len = max_len

    def __call__(self, batch):
        if self.is_test:
            sequences = batch
        else:
            sequences, labels = zip(*batch)
    
        lengths = np.array(list(map(len, sequences)))
        max_len = min(int(np.percentile(lengths, self.percentile)), self.max_len)
        sequences = torch.from_numpy(pad_sequences(sequences, max_len))
        sequences = sequences.long()

        if self.is_test:
            return sequences
        else:
            labels = torch.FloatTensor(labels)
            return sequences, labels


class FieldsCollator:
    def __init__(self, fields: list, is_test: bool = False, percentile: int = 100, max_len: int = 500):
        self.fields = fields
        self.is_test = is_test
        self.percentile = percentile
        self.max_len = max_len

    def __call__(self, batch):
        if self.is_test:
            sequences = batch
        else:
            sequences, labels = zip(*batch)

        res = {}
        for f in self.fields:
            seq = [item[f] for item in sequences]
            lengths = np.array(list(map(len, seq)))
            max_len = int(np.percentile(lengths, self.percentile))
            max_len = min(int(np.percentile(lengths, self.percentile)), self.max_len)
            seq = torch.from_numpy(pad_sequences(seq, max_len))
            seq = seq.long()
            res[f] = seq

        if self.is_test:
            return res
        else:
            res["targets"] = torch.FloatTensor(labels)
            return res
