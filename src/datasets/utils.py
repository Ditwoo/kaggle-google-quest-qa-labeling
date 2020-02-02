import re
import json
import numpy as np


AJAX_FUNCTION = re.compile(r"\$[A-z0-9\(\)\[\]\{\}\.\,\s\/\-\>\<\=\&\|\#\@\!\%\^\*\\\"\'\;\`]*\;")
HTML_TAGS = re.compile(r"\&lt\;[\s\S]+?\&gt\;")
JS_FUNCTION = re.compile(r"function[\s\S]+?\}")
LATEX_EXPRESSION = re.compile(r"\${1,2}[\s\S]+?\${1,2}")
JAVA_FUNCTION = re.compile(r"(public|static|protected) class[\s\S]+?\}")


def clear_text_from_code(text: str) -> str:
    pass


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


class DummyTokenizer:
    def __init__(self,
                 index2word,
                 index2host,
                 index2category,
                 text_fields,
                 host_field,
                 category_field,
                 unknown_token = "<unk>"):
        self.idx2word = index2word
        self.word2idx = {w: idx for idx, w in enumerate(index2word)}

        self.idx2host = index2host
        self.host2idx = {h: idx for idx, h in enumerate(index2host)}

        self.idx2category = index2category
        self.category2idx = {c: idx for idx, c in enumerate(index2category)}

        self.text_fields = text_fields
        self.host_field = host_field
        self.category_field = category_field

        self.separate_chars = [
            ',', '.', '"', ':', ')', '(', '-', '!', '?', 
            '|', ';', "'", '$', '&', '/', '[', ']', '>', 
            '%', '=', '#', '*', '+', '\\', '•',  '~', '@', 
            '£', '·', '_', '{', '}', '©', '^', '®', '`',
            '<', '→', '°', '€', '™', '›',  '♥', '←', '×', 
            '§', '″', '′', 'Â', '█', '½', 'à', '…', '\n', 
            '\xa0', '\t', '“', '★', '”', '–', '●', 'â', 
            '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±',
            '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—',
            '‹', '─', '\u3000', '\u202f', '▒', '：', '¼', 
            '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', 
            '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', 
            '¾', 'Ã', '⋅', '‘', '∞', '«', '∙', '）', '↓', 
            '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', 
            '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', 
            '¹', '≤', '‡', '√', 
        ]
        self.lower = True
        self.split = " "
        self.UNK = unknown_token

    def tokenize(self, state: dict) -> dict:
        """
        Return tokenized (for each field: str -> list[str]) state
        """
        for txt_field in self.text_fields:
            s = state[txt_field]

            if self.lower:
                s = s.lower()

            s = re.sub('[0-9]{5,}', '#####', s)
            s = re.sub('[0-9]{4}', '####', s)
            s = re.sub('[0-9]{3}', '###', s)
            s = re.sub('[0-9]{2}', '##', s)

            for c in self.separate_chars:
                s = s.replace(c, f" {c} ")
            
            state[txt_field] = s
        for field in (self.host_field, self.category_field):
            state[field] = [state[field]]
        return state
    
    def convert_tokens_to_ids(self, state: dict) -> dict:
        for txt_field in self.text_fields:
            state[txt_field] = [self.word2idx[token if token in self.word2idx else self.UNK] 
                                for token in state[txt_field]]
        
        state[self.host_field] = [self.host2idx[host if host in self.host2idx else self.UNK] 
                                  for host in state[self.host_field]]
        state[self.category_field] = [self.category2idx[category if category in self.category2idx else self.UNK] 
                                      for category in state[self.category_field]]
        return state

    @staticmethod
    def from_file(tokenizer_dir):
        with open(tokenizer_dir, 'r') as f:
            content = json.load(f)
        
        return DummyTokenizer(
            index2word=content["text"],
            index2host=content["host"],
            index2category=content["category"],
            text_fields=["question_title", "question_body", "answer"],
            host_field="host",
            category_field="category",
        )
