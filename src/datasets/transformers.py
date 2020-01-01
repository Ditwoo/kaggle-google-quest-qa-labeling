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

HOST_MAP = {
    '<unk>': 0,
    'academia.stackexchange.com': 1,
    'android.stackexchange.com': 2,
    'anime.stackexchange.com': 3,
    'apple.stackexchange.com': 4,
    'askubuntu.com': 5,
    'bicycles.stackexchange.com': 6,
    'biology.stackexchange.com': 7,
    'blender.stackexchange.com': 8,
    'boardgames.stackexchange.com': 9,
    'chemistry.stackexchange.com': 10,
    'christianity.stackexchange.com': 11,
    'codereview.stackexchange.com': 12,
    'cooking.stackexchange.com': 13,
    'crypto.stackexchange.com': 14,
    'cs.stackexchange.com': 15,
    'dba.stackexchange.com': 16,
    'diy.stackexchange.com': 17,
    'drupal.stackexchange.com': 18,
    'dsp.stackexchange.com': 19,
    'electronics.stackexchange.com': 20,
    'ell.stackexchange.com': 21,
    'english.stackexchange.com': 22,
    'expressionengine.stackexchange.com': 23,
    'gamedev.stackexchange.com': 24,
    'gaming.stackexchange.com': 25,
    'gis.stackexchange.com': 26,
    'graphicdesign.stackexchange.com': 27,
    'judaism.stackexchange.com': 28,
    'magento.stackexchange.com': 29,
    'math.stackexchange.com': 30,
    'mathematica.stackexchange.com': 31,
    'mathoverflow.net': 32,
    'mechanics.stackexchange.com': 33,
    'meta.askubuntu.com': 34,
    'meta.christianity.stackexchange.com': 35,
    'meta.codereview.stackexchange.com': 36,
    'meta.math.stackexchange.com': 37,
    'meta.stackexchange.com': 38,
    'meta.superuser.com': 39,
    'money.stackexchange.com': 40,
    'movies.stackexchange.com': 41,
    'music.stackexchange.com': 42,
    'photo.stackexchange.com': 43,
    'physics.stackexchange.com': 44,
    'programmers.stackexchange.com': 45,
    'raspberrypi.stackexchange.com': 46,
    'robotics.stackexchange.com': 47,
    'rpg.stackexchange.com': 48,
    'salesforce.stackexchange.com': 49,
    'scifi.stackexchange.com': 50,
    'security.stackexchange.com': 51,
    'serverfault.com': 52,
    'sharepoint.stackexchange.com': 53,
    'softwarerecs.stackexchange.com': 54,
    'stackoverflow.com': 55,
    'stats.stackexchange.com': 56,
    'superuser.com': 57,
    'tex.stackexchange.com': 58,
    'travel.stackexchange.com': 59,
    'unix.stackexchange.com': 60,
    'ux.stackexchange.com': 61,
    'webapps.stackexchange.com': 62,
    'webmasters.stackexchange.com': 63,
    'wordpress.stackexchange.com': 64
}
CATEGORY_MAP = {
    '<unk>': 0,
    'CULTURE': 1,
    'LIFE_ARTS': 2,
    'SCIENCE': 3,
    'STACKOVERFLOW': 4,
    'TECHNOLOGY': 5
}


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
        title_body_tokens = self.tokenizer.tokenize(title + "," + question)
        title_body_tokens = self._select_tokens(
            title_body_tokens, 
            max_num=MAX_QUESTION_LEN
        )
        ans_tokens = self.tokenizer.tokenize(answer)
        ans_tokens = self._select_tokens(
            ans_tokens, 
            max_num=MAX_ANSWER_LEN
        )
        tokens = ["[CLS]"] + title_body_tokens + ["[SEP]"] + ans_tokens + ["[SEP]"]
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


class TransformerFieldsDatasetWithCategoricalFeatures(TransformerFieldsDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctg_col = "category"
        self.host_col = "host"

    def __getitem__(self, idx):
        index = self.df.index[idx]
        title = self.df.at[index, "question_title"]
        body = self.df.at[index, "question_body"]
        answer = self.df.at[index, "answer"]
        category = self.df.at[index, self.ctg_col]
        category = CATEGORY_MAP[category if category in CATEGORY_MAP else "<unk>"]
        host = self.df.at[index, self.host_col]
        host = HOST_MAP[host if host in HOST_MAP else "<unk>"]
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
        category_id = torch.LongTensor([category])
        host_id = torch.LongTensor([host])
        res = {
            "sequences": token_ids, 
            "segments": segments, 
            "category": category_id,
            "host": host_id,
            "targets": target,
        }
        return res 


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
