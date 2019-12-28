from math import floor, ceil
import numpy as np
import torch
from pandas import DataFrame
from transformers import BertTokenizer, XLNetTokenizer
from torch.utils.data import Dataset
from typing import List


MAX_LEN = 512
MAX_QUESTION_LEN = 250
MAX_ANSWER_LEN = 259
SEP_TOKEN_ID = 102


class TransformerFieldsDataset(Dataset):
    def __init__(self, 
                 df: DataFrame,
                 target: List[str],
                 tokenizer_dir: str,
                 field: str = None,
                 train_mode: bool = True,
                 **kwargs):
        self.df: DataFrame = df
        self.target = target
        self.field = field
        self.train_mode = train_mode
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        # self.tokenizer: XLNetTokenizer = XLNetTokenizer.from_pretrained(tokenizer_dir)
        self.PAD = 0 # self.tokenizer.vocab["[PAD]"]

    def __len__(self):
        return self.df.shape[0]

    def _select_tokens(self, tokens, max_num):
        if len(tokens) <= max_num:
            return tokens
        if self.train_mode:
            num_remove = len(tokens) - max_num
            remove_start = np.random.randint(0, len(tokens) - num_remove - 1)
            return tokens[:remove_start] + tokens[remove_start + num_remove:]
        else:
            return tokens[:max_num // 2] + tokens[-(max_num - max_num // 2):]

    def _build_tokens(self, title, question, answer):
        # TODO: check option to dynamicaly build sequence based on 
        #       sum of lengths of fields
        title_body = self._select_tokens(
            self.tokenizer.tokenize(title + "," + question), 
            max_num=MAX_QUESTION_LEN
        )
        ans = self._select_tokens(
            self.tokenizer.tokenize(answer), 
            max_num=MAX_ANSWER_LEN
        )
        tokens = ["[CLS]"] + title_body + ["[SEP]"] + ans + ["[SEP]"]
        return tokens

    # def _build_tokens(self, 
    #                   title, 
    #                   question, 
    #                   answer, 
    #                   t_max_len=30, 
    #                   q_max_len=239, 
    #                   a_max_len=239):

    #     t = self.tokenizer.tokenize(title)
    #     q = self.tokenizer.tokenize(question)
    #     a = self.tokenizer.tokenize(answer)
        
    #     t_len = len(t)
    #     q_len = len(q)
    #     a_len = len(a)

    #     if t_len + q_len + a_len + 4 > MAX_LEN: 
    #         if t_max_len > t_len:
    #             t_new_len = t_len
    #             a_max_len = a_max_len + floor((t_max_len - t_len)/2)
    #             q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
    #         else:
    #             t_new_len = t_max_len

    #         if a_max_len > a_len:
    #             a_new_len = a_len 
    #             q_new_len = q_max_len + (a_max_len - a_len)
    #         elif q_max_len > q_len:
    #             a_new_len = a_max_len + (q_max_len - q_len)
    #             q_new_len = q_len
    #         else:
    #             a_new_len = a_max_len
    #             q_new_len = q_max_len

    #         if t_new_len + a_new_len + q_new_len + 4 != MAX_LEN:
    #             raise ValueError(
    #                 "New sequence length should be %d, but is %d" % (MAX_LEN, (t_new_len + a_new_len + q_new_len + 4))
    #             )

    #         t = t[:t_new_len]
    #         q = q[:q_new_len]
    #         a = a[:a_new_len]

    #     tokens = ["[CLS]"] + t + ["[SEP]"] + q + ["[SEP]"] + a + ["[SEP]"]
    #     return tokens

    # def _build_segments(self, tokens):
    #     segments = []
    #     sep_num = 0
    #     current_segment_id = 0
    #     for token in tokens:
    #         segments.append(current_segment_id)
    #         if token == "[SEP]":
    #             if sep_num > 1:
    #                 current_segment_id = 1
    #             sep_num += 1
    #     return segments

    def _build_segments(self, tokens):
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments

    def __getitem__(self, idx):
        index = self.df.index[idx]
        title = self.df.at[index, "question_title"]
        body = self.df.at[index, "question_body"]
        answer = self.df.at[index, "answer"]
        # combine fields into one sequence
        tokens = self._build_tokens(title, body, answer)
        segments = self._build_segments(tokens)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # pad sequneces if needed
        if len(token_ids) < MAX_LEN:
            token_ids += [self.PAD] * (MAX_LEN - len(token_ids))
        if len(segments) < MAX_LEN:
            segments += [self.PAD] * (MAX_LEN - len(segments))
        # converting to tensors
        token_ids = torch.LongTensor(token_ids)
        segments = torch.LongTensor(segments) 
        target = torch.FloatTensor([self.df.at[index, c] for c in self.target])
        return {"sequences": token_ids, "segments": segments, "targets": target}


class TransformerMultipleFieldsDataset(Dataset):
    def __init__(self, 
                 df: DataFrame,
                 target: List[str],
                 tokenizer_dir: str,
                 field: str = None,
                 train_mode: bool = True,
                 **kwargs):
        self.df: DataFrame = df
        self.target = target
        self.field = field
        self.train_mode = train_mode
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        self.tok2id = self.tokenizer.convert_tokens_to_ids

        self.CLS = self.tokenizer.vocab["[CLS]"]
        self.PAD = self.tokenizer.vocab["[PAD]"]  # or 0 token

    def __len__(self):
        return self.df.shape[0]

    def _tokenize(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.tokenize(text)
        # for too long sequences - get subsequence
        if len(tokens) > MAX_LEN - 1:
            start_idx = np.random.randint(0, len(tokens) - MAX_LEN + 1)
            tokens = tokens[start_idx:start_idx + MAX_LEN - 1]
        tokens = ["[CLS]"] + tokens
        # switch to integers
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(ids) < MAX_LEN:
            ids += [self.PAD] * (MAX_LEN - len(ids))
        return torch.LongTensor(ids)
    
    def __getitem__(self, idx):
        index = self.df.index[idx]
        title = self.df.at[index, "question_title"]
        body = self.df.at[index, "question_body"]
        answer = self.df.at[index, "answer"]

        
        fields = dict()
        fields["question_title"] = self._tokenize(title)
        fields["question_body"] = self._tokenize(body)
        fields["answer"] = self._tokenize(answer)

        fields["targets"] = torch.FloatTensor(
            [self.df.at[index, c] for c in self.target]
        )
        return fields


class TransformersCollator:
    def __init__(self,
                 is_test: bool = False):
        self.is_test = is_test

    def __call__(self, batch):
        if self.is_test:
            sequences = batch
            return torch.stack(sequences)
        else: 
            sequences, labels = zip(*batch)
            return torch.stack(sequences), torch.stack(labels)
