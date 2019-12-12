import numpy as np


def pad_sequences(sequences: list,
                  max_len: int,
                  value: int = 0,
                  padding: str = "pre",
                  dtype=np.int32) -> np.ndarray:
    """
    Pad sequences with specified value.
    
    Example of different padding strategies:

    >>> seqs = [[1, 2, 3], [4, 5], [6]]
    >>> pad_sequences(seqs, max_len=3, padding="post")
    array([[1, 2, 3],
       [4, 5, 0],
       [6, 0, 0]], dtype=int32)
    >>> pad_sequences(seqs, max_len=3, padding="pre")
    array([[1, 2, 3],
       [0, 4, 5],
       [0, 0, 6]], dtype=int32)
    """

    if not max_len > 0:
        raise ValueError("`max_len` should be greater than 0")

    if padding not in {"pre", "post"}:
        raise ValueError("`padding` should be one of `pre` or `post`")

    features = np.full(
        shape=(len(sequences), max_len),
        fill_value=value,
        dtype=dtype
    )

    for idx, row in enumerate(sequences):
        if len(row):
            if padding == "pre":
                features[idx, -len(row):] = np.array(row)[:max_len]
            else:
                features[idx, : len(row)] = np.array(row)[:max_len]

    return features
