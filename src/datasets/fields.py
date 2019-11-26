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
    def __init__(self, df: DataFrame, feature_cols: List[str], target: List[str], transforms: SeqTransformationInterface):
        self.df: DataFrame = df
        self.features: List[str] = feature_cols
        self.target: List[str] = target
        self.augs: SeqTransformationInterface = transforms

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        index = self.df.index[idx]
        state = {c: self.df.at[index, c] for c in self.features}
        aug_state = self.augs.transform(state)
        seq = aug_state["seq"]
        target = [self.df.at[index, c] for c in self.target] 
        return seq, target


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

    def __init__(self, 
                 is_test: bool = False, 
                 percentile: int = 100, 
                 max_len: int = 1_000):
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
