import numpy as np


def pad_sequences(sequences: list,
                  max_len: int,
                  value: int = 0,
                  padding: str = 'pre') -> np.ndarray:
    assert max_len > 0, '`max_len` should be greater than 0'
    assert padding in {'pre', 'post'}, '`padding` should be one of `pre` or `post`'

    features = np.full((len(sequences), max_len), value, dtype=np.int32)

    for idx, row in enumerate(sequences):
        if len(row):
            if padding == 'pre':
                features[idx, -len(row):] = np.array(row)[:max_len]
            else:
                features[idx, :len(row)] = np.array(row)[:max_len]

    return features