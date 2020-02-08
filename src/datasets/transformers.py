import html
import re
from math import floor, ceil
from itertools import product
import numpy as np
import torch
from functools import partial
from pandas import DataFrame
from transformers import BasicTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer
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
                 tokenizer: BasicTokenizer,
                 field: str = None,
                 train_mode: bool = True,
                 pre_pad: bool = False,
                 **kwargs):
        self.df: DataFrame = df
        self.target = target
        self.field = field
        self.train_mode = train_mode
        self.tokenizer = tokenizer
        self.PAD = self.tokenizer.pad_token_id
        self.PAD_TOKEN = self.tokenizer.pad_token
        self.pre_pad = pre_pad
        self.pad_foo = self._pad_foo

    def _pad_foo(self, tokens_list, max_size, pad_value=None) -> list:
        """
        Arguments:
            tokens_list - list of strings (tokens)
            max_size - int, maximal size of sequnece
            pad_value - str, value to use for padding, if not specified then will be used pad value from tokenizer
            pre_pad - use padding before sequence (<pad>, ... <pad>, begin, ..., end) or after sequence (begin, ..., end, <pad>, ..., <pad>)
        """
        pad_value = self.PAD if pad_value is None else pad_value
        _PAD = []
        if len(tokens_list) < max_size:
            _PAD = [pad_value] * (max_size - len(tokens_list))

        if self.pre_pad:
            tokens_list = _PAD + tokens_list 
        else:
            tokens_list = tokens_list + _PAD

        return tokens_list

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

    def build_tokens_and_segments(self, title, question, answer):
        """
        Input will be represented as: 
        <CLS> {title & question token ids} <SEP> {asnwer token ids} <SEP>

        With mask:
        0 ............................... 0  1 ...................... 1
        """

        # first part of input
        title_body_tokens = self.tokenizer.tokenize(title + " " + question)  # list of tokens (strings)
        title_body_tokens = self._select_tokens(title_body_tokens, MAX_QUESTION_LEN)  # list of tokens (strings)
        title_body_tokens_ids = self.tokenizer.convert_tokens_to_ids(title_body_tokens)  # list of integers
        # second part of input
        ans_tokens = self.tokenizer.tokenize(answer)  # list of tokens (strings)
        ans_tokens = self._select_tokens(ans_tokens, MAX_ANSWER_LEN)  # list of tokens (strings)
        ans_tokens_ids = self.tokenizer.convert_tokens_to_ids(ans_tokens)  # list of integers
        # outputs
        token_ids = self.tokenizer.build_inputs_with_special_tokens(title_body_tokens_ids, ans_tokens_ids)  # list of integers
        segments = self.tokenizer.create_token_type_ids_from_sequences(title_body_tokens_ids, ans_tokens_ids)  # list of integers
        return token_ids, segments

    def __getitem__(self, idx):
        index = self.df.index[idx]
        title = self.df.at[index, "question_title"]
        body = self.df.at[index, "question_body"]
        answer = self.df.at[index, "answer"]
        # combine fields into one sequence
        token_ids, segments = self.build_tokens_and_segments(title, body, answer)
        # pad sequneces if needed
        token_ids = self._pad_foo(token_ids, MAX_LEN, self.tokenizer.pad_token_id)
        segments = self._pad_foo(segments, MAX_LEN, 0)
        # converting to tensors
        token_ids = torch.LongTensor(token_ids)
        segments = torch.LongTensor(segments) 
        target = torch.FloatTensor([self.df.at[index, c] for c in self.target])
        res = {
            "sequences": token_ids, 
            "segments": segments, 
            "targets": target
        }
        return res


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
        token_ids, segments = self.build_tokens_and_segments(title, body, answer)
        # pad sequneces if needed
        token_ids = self.pad_foo(token_ids, MAX_LEN)
        segments = self.pad_foo(segments, MAX_LEN)
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


class TFDCFSF(TransformerFieldsDataset):
    """
    Transformer Dataset based on data fields, categorical features and statistical features.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctg_col = "category"
        self.host_col = "host"
        self.str_cols = ["question_title", "question_body", "answer"]
        self.pre_pad = False

    def build_tokens_and_segments(self, title, question, answer):
        """
        Input will be represented as: 
        <CLS> {title & question tokens} <SEP> {asnwer tokens} <SEP>

        With mask:
        0 ............................ 0  1 ................... 1
        """

        title_body_tokens = self.tokenizer.tokenize(title + "," + question)
        num_title_body_tokens = len(title_body_tokens)
        title_body_tokens = self._select_tokens(
            title_body_tokens, 
            max_num=MAX_QUESTION_LEN
        )
        ans_tokens = self.tokenizer.tokenize(answer)
        num_ans_tokens = len(ans_tokens)
        ans_tokens = self._select_tokens(
            ans_tokens, 
            max_num=MAX_ANSWER_LEN
        )
        first_part = [self.tokenizer.cls_token] + title_body_tokens + [self.tokenizer.sep_token] 
        second_part = ans_tokens + [self.tokenizer.sep_token]
        segments = [0] * len(first_part) + [1] * len(second_part)
        tokens = first_part + second_part
        return tokens, segments, (num_title_body_tokens, num_ans_tokens)

    def build_stats(self, 
                    title: str, body: str, answer: str, 
                    num_title_body_tokens: int, num_ans_tokens: int) -> list:
        # (val - mean) / std
        title_body_tokens = (num_title_body_tokens - 241) / 334
        ans_tokens = (num_ans_tokens - 217) / 289
        
        title_str_len = (len(title) - 53) / 20
        title_alpha_num = (sum(1 for c in title if c.isalpha()) - 43) / 16
        title_nums_num = (sum(1 for c in title if c.isnumeric()) - 0) / 1
        title_low_num = (sum(1 for c in title if c.islower()) - 40) / 16
        title_upp_num = (sum(1 for c in title if c.isupper()) - 2) / 2
        title_space_num = (sum(1 for c in title if c.isspace()) - 8) / 3
        title_words_num = (len(title.split()) - 9) / 3

        body_str_len = (len(body) - 834) / 1035
        body_alpha_num = (sum(1 for c in title if c.isalpha()) - 591) / 663
        body_nums_num = (sum(1 for c in title if c.isnumeric()) - 15) / 68
        body_low_num = (sum(1 for c in title if c.islower()) - 558) / 608
        body_upp_num = (sum(1 for c in title if c.isupper()) - 33) / 80
        body_space_num = (sum(1 for c in title if c.isspace()) - 165) / 263
        body_words_num = (len(body.split()) - 125) / 116
        
        answer_str_len = (len(body) - 843) / 1023
        answer_alpha_num = (sum(1 for c in title if c.isalpha()) - 624) / 727
        answer_nums_num = (sum(1 for c in title if c.isnumeric()) - 9) / 79
        answer_low_num = (sum(1 for c in title if c.islower()) - 597) / 659
        answer_upp_num = (sum(1 for c in title if c.isupper()) - 26) / 56
        answer_space_num = (sum(1 for c in title if c.isspace()) - 157) / 232
        answer_words_num = (len(answer.split()) - 133) / 156
        
        res_stats = [
            title_body_tokens, ans_tokens, title_str_len, body_str_len, answer_str_len,
            title_alpha_num, title_nums_num, title_low_num, title_upp_num, title_space_num, title_words_num,
            body_alpha_num, body_nums_num, body_low_num, body_upp_num, body_space_num, body_words_num,
            answer_alpha_num, answer_nums_num, answer_low_num, answer_upp_num, answer_space_num, answer_words_num,
        ] # 23 stat features

        return res_stats

    def __getitem__(self, idx) -> dict:
        index = self.df.index[idx]
        title = self.df.at[index, "question_title"]
        body = self.df.at[index, "question_body"]
        answer = self.df.at[index, "answer"]
        category = self.df.at[index, self.ctg_col]
        category = CATEGORY_MAP[category if category in CATEGORY_MAP else "<unk>"]
        host = self.df.at[index, self.host_col]
        host = HOST_MAP[host if host in HOST_MAP else "<unk>"]
        # combine fields into one sequence
        tokens, segments, (num_tb_tokens, num_a_tokens) = self.build_tokens_and_segments(title, body, answer)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # pad sequneces if needed
        token_ids = self.pad_foo(token_ids, MAX_LEN)
        segments = self.pad_foo(segments, MAX_LEN)
        # converting to tensors
        token_ids = torch.LongTensor(token_ids)
        segments = torch.LongTensor(segments)
        category_id = torch.LongTensor([category])
        host_id = torch.LongTensor([host])
        stats = torch.FloatTensor(self.build_stats(title, body, answer, num_tb_tokens, num_a_tokens))
        target = torch.FloatTensor([self.df.at[index, c] for c in self.target])
        res = {
            "sequences": token_ids, 
            "segments": segments, 
            "category": category_id,
            "host": host_id,
            "stats": stats,
            "targets": target,
        }
        return res


class RFDCFSF(TFDCFSF):
    """
    RoBERTa Dataset based on data fields, categorical features and statistical features.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_pad = False

    def build_tokens_and_segments(self, title, question, answer):
        """
        Input will be represented as: 
        <CLS> {title & question tokens} <SEP> <SEP> {asnwer tokens} <SEP>

        With mask:
        0 ............................... 0     0  1 ................ 1
        """
        title_body_tokens = self.tokenizer.tokenize(title + "," + question)
        num_title_body_tokens = len(title_body_tokens)
        title_body_tokens = self._select_tokens(
            title_body_tokens, 
            max_num=MAX_QUESTION_LEN
        )
        ans_tokens = self.tokenizer.tokenize(answer)
        num_ans_tokens = len(ans_tokens)
        ans_tokens = self._select_tokens(
            ans_tokens, 
            max_num=MAX_ANSWER_LEN - 1
        )
        first_part = [self.tokenizer.cls_token] + title_body_tokens + [self.tokenizer.sep_token, self.tokenizer.sep_token]
        second_part = ans_tokens + [self.tokenizer.sep_token]
        tokens = first_part + second_part
        segments = [0] * len(first_part) + [1] * len(second_part) 
        return tokens, segments, (num_title_body_tokens, num_ans_tokens)

    def __getitem__(self, idx) -> dict:
        index = self.df.index[idx]
        title = self.df.at[index, "question_title"]
        body = self.df.at[index, "question_body"]
        answer = self.df.at[index, "answer"]
        category = self.df.at[index, self.ctg_col]
        category = CATEGORY_MAP[category if category in CATEGORY_MAP else "<unk>"]
        host = self.df.at[index, self.host_col]
        host = HOST_MAP[host if host in HOST_MAP else "<unk>"]
        # combine fields into one sequence
        tokens, segments, (num_tb_tokens, num_a_tokens) = self.build_tokens_and_segments(title, body, answer)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # converting to tensors & pad sequneces if needed
        token_ids = torch.LongTensor(self.pad_foo(token_ids, MAX_LEN, self.tokenizer.pad_token_id))
        # segments = torch.LongTensor(self._pad_foo(segments, MAX_LEN, 0))
        category_id = torch.LongTensor([category])
        host_id = torch.LongTensor([host])
        stats = torch.FloatTensor(self.build_stats(title, body, answer, num_tb_tokens, num_a_tokens))
        target = torch.FloatTensor([self.df.at[index, c] for c in self.target])
        return {
            "sequences": token_ids, 
            # "segments": segments,
            "category": category_id,
            "host": host_id,
            "stats": stats,
            "targets": target,
        }


class XFDCFSF(TFDCFSF):
    """
    XLNet Dataset based on data fields, categorical features and statistical features.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_pad = True

    def build_tokens_and_segments(self, title, question, answer):
        """
        Input will be represented as: 
        {title & question token ids} <SEP> {asnwer token ids} <SEP> <CLS>

        With mask:
        0 .......................... 0 1  0 ................. 0 1     1
        """
        title_body_tokens = self.tokenizer.tokenize(title + "," + question)
        num_title_body_tokens = len(title_body_tokens)
        title_body_tokens = self._select_tokens(
            title_body_tokens,
            max_num=MAX_QUESTION_LEN
        )
        title_body_token_ids = self.tokenizer.convert_tokens_to_ids(title_body_tokens)

        ans_tokens = self.tokenizer.tokenize(answer)
        num_ans_tokens = len(ans_tokens)
        ans_tokens = self._select_tokens(
            ans_tokens,
            max_num=MAX_ANSWER_LEN
        )
        ans_token_ids = self.tokenizer.convert_tokens_to_ids(ans_tokens)
        
        token_ids = self.tokenizer.build_inputs_with_special_tokens(title_body_token_ids, ans_token_ids)
        segments = self.tokenizer.create_token_type_ids_from_sequences(title_body_token_ids, ans_token_ids)

        return token_ids, segments, (num_title_body_tokens, num_ans_tokens)

    def __getitem__(self, idx) -> dict:
        index = self.df.index[idx]
        title = self.df.at[index, "question_title"]
        body = self.df.at[index, "question_body"]
        answer = self.df.at[index, "answer"]
        category = self.df.at[index, self.ctg_col]
        category = CATEGORY_MAP[category if category in CATEGORY_MAP else "<unk>"]
        host = self.df.at[index, self.host_col]
        host = HOST_MAP[host if host in HOST_MAP else "<unk>"]
        # combine fields into one sequence
        token_ids, segments, (num_tb_tokens, num_a_tokens) = self.build_tokens_and_segments(title, body, answer)
    
        # converting to tensors & pad sequneces if needed
        token_ids = torch.LongTensor(self.pad_foo(token_ids, MAX_LEN, self.tokenizer.pad_token_id))
        segments = torch.LongTensor(self._pad_foo(segments, MAX_LEN, 0))
        category_id = torch.LongTensor([category])
        host_id = torch.LongTensor([host])
        stats = torch.FloatTensor(self.build_stats(title, body, answer, num_tb_tokens, num_a_tokens))
        target = torch.FloatTensor([self.df.at[index, c] for c in self.target])
        return {
            "sequences": token_ids, 
            "segments": segments,
            "category": category_id,
            "host": host_id,
            "stats": stats,
            "targets": target,
        }


class BertDataset(TransformerFieldsDataset):
    def __init__(self, 
                 df: DataFrame,
                 target: List[str],
                 tokenizer: BasicTokenizer,
                 field: str = None,
                 train_mode: bool = True,
                 **kwargs):
        super().__init__(
            df=df,
            target=target,
            tokenizer=tokenizer,
            field=field,
            train_mode=train_mode,
            pre_pad=False,
            **kwargs
        )

        def __getitem__(self, idx):
            index = self.df.index[idx]
            title = html.unescape(self.df.at[index, "question_title"])
            body = html.unescape(self.df.at[index, "question_body"])
            answer = html.unescape(self.df.at[index, "answer"])
            # combine fields into one sequence
            token_ids, segments = self.build_tokens_and_segments(title, body, answer)
            # pad sequneces if needed
            token_ids = self._pad_foo(token_ids, MAX_LEN, self.tokenizer.pad_token_id)
            segments = self._pad_foo(segments, MAX_LEN, 0)
            # converting to tensors
            token_ids = torch.LongTensor(token_ids)
            segments = torch.LongTensor(segments) 
            target = torch.FloatTensor([self.df.at[index, c] for c in self.target])
            res = {
                "sequences": token_ids, 
                "segments": segments, 
                "targets": target
            }
            return res


class XLNetDataset(TransformerFieldsDataset):
    def __init__(self, 
                 df: DataFrame,
                 target: List[str],
                 tokenizer: BasicTokenizer,
                 field: str = None,
                 train_mode: bool = True,
                 **kwargs):
        super().__init__(
            df=df,
            target=target,
            tokenizer=tokenizer,
            field=field,
            train_mode=train_mode,
            pre_pad=True,
            **kwargs
        )

    def __getitem__(self, idx):
        index = self.df.index[idx]
        title = html.unescape(self.df.at[index, "question_title"])
        body = html.unescape(self.df.at[index, "question_body"])
        answer = html.unescape(self.df.at[index, "answer"])
        # combine fields into one sequence
        token_ids, segments = self.build_tokens_and_segments(title, body, answer)
        # pad sequneces if needed
        token_ids = self._pad_foo(token_ids, MAX_LEN, self.tokenizer.pad_token_id)
        segments = self._pad_foo(segments, MAX_LEN, 0)
        # converting to tensors
        token_ids = torch.LongTensor(token_ids)
        segments = torch.LongTensor(segments) 
        target = torch.FloatTensor([self.df.at[index, c] for c in self.target])
        res = {
            "sequences": token_ids, 
            "segments": segments, 
            "targets": target
        }
        return res


class GPT2Dataset(TransformerFieldsDataset):
    def __init__(self, 
                 df: DataFrame,
                 target: List[str],
                 tokenizer: BasicTokenizer,
                 field: str = None,
                 train_mode: bool = True,
                 **kwargs):
        super().__init__(
            df=df,
            target=target,
            tokenizer=tokenizer,
            field=field,
            train_mode=train_mode,
            pre_pad=False,
            **kwargs
        )

    def __getitem__(self, idx):
        index = self.df.index[idx]
        title = html.unescape(self.df.at[index, "question_title"])
        body = html.unescape(self.df.at[index, "question_body"])
        answer = html.unescape(self.df.at[index, "answer"])
        # combine fields into one sequence
        token_ids, segments = self.build_tokens_and_segments(title, body, answer)
        # pad sequneces if needed
        token_ids = self._pad_foo(token_ids, MAX_LEN, self.tokenizer.pad_token_id)
        segments = self._pad_foo(segments, MAX_LEN, 0)
        # converting to tensors
        token_ids = torch.LongTensor(token_ids)
        segments = torch.LongTensor(segments) 
        target = torch.FloatTensor([self.df.at[index, c] for c in self.target])
        res = {
            "sequences": token_ids, 
            "segments": segments, 
            "targets": target
        }
        return res


class Stats:
    def __init__(self, config: dict = None):
        if config is None:
            config = {
                "num_title_body_tokens": {"mean": 0, "std": 1},
                "num_ans_tokens": {"mean": 0, "std": 1},
                "title": {
                    "text_len": {"mean": 0, "std": 1},
                    "alpha_num": {"mean": 0, "std": 1},
                    "nums_num": {"mean": 0, "std": 1},
                    "low_num": {"mean": 0, "std": 1},
                    "upp_num": {"mean": 0, "std": 1},
                    "space_num": {"mean": 0, "std": 1},
                    "words_num": {"mean": 0, "std": 1},
                },
                "body": {
                    "text_len": {"mean": 0, "std": 1},
                    "alpha_num": {"mean": 0, "std": 1},
                    "nums_num": {"mean": 0, "std": 1},
                    "low_num": {"mean": 0, "std": 1},
                    "upp_num": {"mean": 0, "std": 1},
                    "space_num": {"mean": 0, "std": 1},
                    "words_num": {"mean": 0, "std": 1},
                },
                "answer": {
                    "text_len": {"mean": 0, "std": 1},
                    "alpha_num": {"mean": 0, "std": 1},
                    "nums_num": {"mean": 0, "std": 1},
                    "low_num": {"mean": 0, "std": 1},
                    "upp_num": {"mean": 0, "std": 1},
                    "space_num": {"mean": 0, "std": 1},
                    "words_num": {"mean": 0, "std": 1},
                },
                "host": [],
                "category": [],
            }
        self.config = config

    @staticmethod
    def normalize(val, mean, std) -> float:
        return (val - mean) / std

    def _compute_pack_of_str_metrics(self, text: str, c: dict) -> list:

        str_len = self.normalize(len(text), **c["text_len"])
        alpha_num = self.normalize(sum(1 for c in text if c.isalpha()), **c["alpha_num"])
        nums_num = self.normalize(sum(1 for c in text if c.isnumeric()), **c["nums_num"])
        low_num = self.normalize(sum(1 for c in text if c.islower()), **c["low_num"])
        upp_num = self.normalize(sum(1 for c in text if c.isupper()), **c["upp_num"])
        space_num = self.normalize(sum(1 for c in text if c.isspace()), **c["space_num"])
        words_num = self.normalize(len(text.split()), **c["words_num"])

        return [str_len, alpha_num, nums_num, low_num, upp_num, space_num, words_num]


    def build_stats(self, 
                    title: str, body: str, answer: str, 
                    num_title_body_tokens: int, num_ans_tokens: int) -> list:
        c = self.config
        title_body_tokens = self.normalize(num_title_body_tokens, **c["num_title_body_tokens"])
        ans_tokens = self.normalize(num_ans_tokens, **c["num_ans_tokens"])

        title_stats = self._compute_pack_of_str_metrics(title, self.config["title"])
        body_stats = self._compute_pack_of_str_metrics(title, self.config["body"])
        ans_stats = self._compute_pack_of_str_metrics(title, self.config["answer"])

        stats = [title_body_tokens, ans_tokens] + title_stats + body_stats + ans_stats
        return stats  # 23 features

    def build_stats_and_categories(self,
        title: str, body: str, answer: str, host: str, category: str,
        num_title_body_tokens: int, num_ans_tokens: int) -> list:
        stats = self.build_stats(title, body, answer, num_title_body_tokens, num_ans_tokens)
        te_host = self.config["host"].get(host, self.config["host"]["<unk>"])
        te_category = self.config["category"].get(category, self.config["category"]["<unk>"])

        return stats + te_host + te_category  # 83


class FoldTFDCFSF(TransformerFieldsDataset):
    """
    Fold Transformer Dataset based on data fields, categorical features and statistical features.
    Stats normalization parameters will be loaded from 'stats_config' dict
    """

    def __init__(self, stats_config: dict, **kwargs):
        super().__init__(**kwargs)
        self.ctg_col = "category"
        self.host_col = "host"
        self.str_cols = ["question_title", "question_body", "answer"]
        self.pre_pad = False
        self.stats_cfg: Stats = Stats(stats_config)

    def build_stats(self, 
                    title: str, body: str, answer: str, 
                    num_title_body_tokens: int, num_ans_tokens: int) -> list:
        return self.stats_cfg.build_stats(title, body, answer, num_title_body_tokens, num_ans_tokens)

    def build_tokens_and_segments(self, title, question, answer):
        """
        Input will be represented as: 
        <CLS> {title & question token ids} <SEP> {asnwer token ids} <SEP>

        With mask:
        0 ............................... 0  1 ...................... 1
        """
        # first part of input
        title_body_tokens = self.tokenizer.tokenize(title + "," + question)  # list of tokens (strings)
        num_title_body_tokens = len(title_body_tokens)
        title_body_tokens = self._select_tokens(title_body_tokens, MAX_QUESTION_LEN)  # list of tokens (strings)
        title_body_tokens_ids = self.tokenizer.convert_tokens_to_ids(title_body_tokens)  # list of integers
        # second part of input
        ans_tokens = self.tokenizer.tokenize(answer)  # list of tokens (strings)
        num_ans_tokens = len(ans_tokens)
        ans_tokens = self._select_tokens(ans_tokens, MAX_ANSWER_LEN)  # list of tokens (strings)
        ans_tokens_ids = self.tokenizer.convert_tokens_to_ids(ans_tokens)  # list of integers
        # outputs
        token_ids = self.tokenizer.build_inputs_with_special_tokens(title_body_tokens_ids, ans_tokens_ids)  # list of integers
        segments = self.tokenizer.create_token_type_ids_from_sequences(title_body_tokens_ids, ans_tokens_ids)  # list of integers
        return token_ids, segments, (num_title_body_tokens, num_ans_tokens)

    def __getitem__(self, idx) -> dict:
        index = self.df.index[idx]
        title = self.df.at[index, "question_title"]
        body = self.df.at[index, "question_body"]
        answer = self.df.at[index, "answer"]
        category = self.df.at[index, self.ctg_col]
        category = CATEGORY_MAP[category if category in CATEGORY_MAP else "<unk>"]
        host = self.df.at[index, self.host_col]
        host = HOST_MAP[host if host in HOST_MAP else "<unk>"]
        # combine fields into one sequence
        token_ids, segments, (num_tb_tokens, num_a_tokens) = self.build_tokens_and_segments(title, body, answer)
        # pad sequneces if needed
        token_ids = self.pad_foo(token_ids, MAX_LEN)
        segments = self.pad_foo(segments, MAX_LEN)
        # converting to tensors
        token_ids = torch.LongTensor(token_ids)
        segments = torch.LongTensor(segments)
        category_id = torch.LongTensor([category])
        host_id = torch.LongTensor([host])
        stats = torch.FloatTensor(self.build_stats(title, body, answer, num_tb_tokens, num_a_tokens))
        target = torch.FloatTensor([self.df.at[index, c] for c in self.target])
        res = {
            "sequences": token_ids, 
            "segments": segments, 
            "category": category_id,
            "host": host_id,
            "stats": stats,
            "targets": target,
        }
        return res


class FoldTFDCSF(FoldTFDCFSF):
    """
    Fold Transformer Dataset based on data fields, categorical features as target encoded vectors 
    and statistical features.
    Stats normalization parameters will be loaded from 'stats_config' dict
    """

    def build_stats(self, 
                    title: str, body: str, answer: str, host: str, category: str,
                    num_title_body_tokens: int, num_ans_tokens: int) -> list:
        return self.stats_cfg.build_stats_and_categories(
            title, body, answer, host, category, num_title_body_tokens, num_ans_tokens
        ) # 83 numbers


    def __getitem__(self, idx) -> dict:
        index = self.df.index[idx]
        title = self.df.at[index, "question_title"]
        body = self.df.at[index, "question_body"]
        answer = self.df.at[index, "answer"]

        category = self.df.at[index, self.ctg_col]
        category = CATEGORY_MAP[category if category in CATEGORY_MAP else "<unk>"]

        host = self.df.at[index, self.host_col]
        host = HOST_MAP[host if host in HOST_MAP else "<unk>"]
        # combine fields into one sequence
        token_ids, segments, (num_tb_tokens, num_a_tokens) = self.build_tokens_and_segments(title, body, answer)
        # pad sequneces if needed
        token_ids = self.pad_foo(token_ids, MAX_LEN)
        segments = self.pad_foo(segments, MAX_LEN)
        # converting to tensors
        token_ids = torch.LongTensor(token_ids)
        segments = torch.LongTensor(segments)
        category_id = torch.LongTensor([category])
        host_id = torch.LongTensor([host])
        stats = torch.FloatTensor(
            self.build_stats(title, body, answer, host, category, num_tb_tokens, num_a_tokens)
        )
        target = torch.FloatTensor([self.df.at[index, c] for c in self.target])
        res = {
            "sequences": token_ids, 
            "segments": segments, 
            "category": category_id,
            "host": host_id,
            "stats": stats,
            "targets": target,
        }
        return res


class JoinedTransformerFieldsDataset(Dataset):
    def __init__(self, 
                 df: DataFrame,
                 target: List[str],
                 tokenizer: BasicTokenizer,
                 field: str = None,
                 train_mode: bool = True,
                 pre_pad: bool = False,
                 **kwargs):
        self.df: DataFrame = df
        self.target = target
        self.field = field
        self.train_mode = train_mode
        self.tokenizer = tokenizer
        self.PAD = self.tokenizer.pad_token_id
        self.PAD_TOKEN = self.tokenizer.pad_token
        self.pre_pad = pre_pad
        self.data = []
        self.init_data()

    def pad_foo(self, tokens_list, max_size, pad_value=None) -> list:
        """
        Arguments:
            tokens_list - list of strings (tokens)
            max_size - int, maximal size of sequnece
            pad_value - str, value to use for padding, if not specified then will be used pad value from tokenizer
            pre_pad - use padding before sequence (<pad>, ... <pad>, begin, ..., end) or after sequence (begin, ..., end, <pad>, ..., <pad>)
        """
        pad_value = self.PAD if pad_value is None else pad_value
        _PAD = []
        if len(tokens_list) < max_size:
            _PAD = [pad_value] * (max_size - len(tokens_list))

        if self.pre_pad:
            tokens_list = _PAD + tokens_list 
        else:
            tokens_list = tokens_list + _PAD

        return tokens_list

    def __len__(self):
        return len(self.data)

    def _select_tokens(self, tokens, max_num):
        if len(tokens) <= max_num:
            return tokens
        if self.train_mode:
            num_remove = len(tokens) - max_num
            remove_start = np.random.randint(0, len(tokens) - num_remove - 1)
            return tokens[:remove_start] + tokens[remove_start + num_remove:]
        else:
            return tokens[:max_num // 2] + tokens[-(max_num - max_num // 2):]

    def parts(self, sequence: list, step: int, stride: int) -> list:
        return [sequence[i:i + step] for i in range(0, len(sequence), stride)]

    def build_tokens_and_segments(self, title: str, question: str, answer: str):
        """
        Used only in test mode (train_mode == False).
        """
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
        first_part = [self.tokenizer.cls_token] + title_body_tokens + [self.tokenizer.sep_token] 
        second_part = ans_tokens + [self.tokenizer.sep_token]
        segments = [0] * len(first_part) + [1] * len(second_part)
        tokens = first_part + second_part
        return tokens, segments

    def init_data(self):
        data = []
        for index in self.df.index:
            title = self.df.at[index, "question_title"]
            body = self.df.at[index, "question_body"]
            answer = self.df.at[index, "answer"]
            target_vec = [self.df.at[index, c] for c in self.target]
            if self.train_mode:
                records = []
                # generating multiple records from 1 sample
                title_body_parts = self.parts(
                    self.tokenizer.tokenize(title + "," + body),
                    step=MAX_QUESTION_LEN,
                    stride=128,
                )
                ans_parts = self.parts(
                    self.tokenizer.tokenize(answer), 
                    step=MAX_ANSWER_LEN,
                    stride=128,
                )
                for tb, a in product(title_body_parts, ans_parts):
                    title_body_tokens = self._select_tokens(tb, max_num=MAX_QUESTION_LEN)
                    ans_tokens = self._select_tokens(a, max_num=MAX_ANSWER_LEN)
                    first_part = [self.tokenizer.cls_token] + title_body_tokens + [self.tokenizer.sep_token] 
                    second_part = ans_tokens + [self.tokenizer.sep_token]
                    segments = [0] * len(first_part) + [1] * len(second_part)
                    token_ids = self.tokenizer.convert_tokens_to_ids(first_part + second_part)
                    # pad sequneces if needed
                    token_ids = self.pad_foo(token_ids, MAX_LEN, self.tokenizer.pad_token_id)
                    segments = self.pad_foo(segments, MAX_LEN, self.tokenizer.pad_token_id)
                    # converting to tensors
                    token_ids = torch.LongTensor(token_ids)
                    segments = torch.LongTensor(segments) 
                    target = torch.FloatTensor(target_vec)
                    records.append({
                        "sequences": token_ids, 
                        "segments": segments, 
                        "targets": target
                    })
                data.extend(records)
            else:
                # combine fields into one sequence
                tokens, segments = self.build_tokens_and_segments(title, body, answer)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                # pad sequneces if needed
                token_ids = self.pad_foo(token_ids, MAX_LEN, self.tokenizer.pad_token_id)
                segments = self.pad_foo(segments, MAX_LEN, self.tokenizer.pad_token_id)
                # converting to tensors
                token_ids = torch.LongTensor(token_ids)
                segments = torch.LongTensor(segments) 
                target = torch.FloatTensor(target_vec)
                data.append({
                    "sequences": token_ids, 
                    "segments": segments, 
                    "targets": target
                })
        self.data = data

    def __getitem__(self, idx) -> dict:
        return self.data[idx]


class TFDCC(TransformerFieldsDataset):
    """
    Transformer Dataset with code cleaning.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        # # code in spaces
        # text = re.sub(r"(e\.g\.| |\?|this|:|;|output)(\n{2,3}([\s\S]+?\n)\n{2})", "\1 code ", text)
        
        text = html.unescape(text)
        text = re.sub(r"\<[\s\S]+?\>", " code block ", text)
        
        # text = re.sub(r"((if|for)\W*\([\s\S]+?\)\W\{[\s\S]+\}|\{[\s\S]+\})", " code block ", text)
        # text = re.sub(r"( \$[\S]+ |\$[\s\S]+?;\n)", " code ", text)
        # text = re.sub(r"\/\/[\s\S]+?\n", " code comment ", text)
        # text = re.sub(r"\#[\s\S]+?\n", " code comment ", text)
        # text = re.sub(r"\/\*[\s\S]+\/\*", " code comment ", text)
        # text = re.sub(r"\([\s\S\W]+?\)\W+\{[\s\S]+\}", " function ", text)

        # latex equations
        text = re.sub(r" ?(\$[\s\S]+?\$|\$\$[\s\S]+?\$\$) ", " equation ", text)

        return text

    @staticmethod
    def glue_title_and_question(t: str, q: str) -> str:
        t = t.strip()
        last_char = t[-1]
        sep_symbol = " " if last_char in {".", ",", ":", ";", "?", "!", "-"} else ","
        return t + sep_symbol + q

    def build_tokens_and_segments(self, title, question, answer):
        question = self.clean_text(question)
        answer = self.clean_text(answer)
        # first part of input
        title_body_tokens = self.tokenizer.tokenize(self.glue_title_and_question(title, question))  # list of tokens (strings)
        title_body_tokens = self._select_tokens(title_body_tokens, MAX_QUESTION_LEN)  # list of tokens (strings)
        title_body_tokens_ids = self.tokenizer.convert_tokens_to_ids(title_body_tokens)  # list of integers
        # second part of input
        ans_tokens = self.tokenizer.tokenize(answer)  # list of tokens (strings)
        ans_tokens = self._select_tokens(ans_tokens, MAX_ANSWER_LEN)  # list of tokens (strings)
        ans_tokens_ids = self.tokenizer.convert_tokens_to_ids(ans_tokens)  # list of integers
        # outputs
        token_ids = self.tokenizer.build_inputs_with_special_tokens(title_body_tokens_ids, ans_tokens_ids)  # list of integers
        segments = self.tokenizer.create_token_type_ids_from_sequences(title_body_tokens_ids, ans_tokens_ids)  # list of integers
        return token_ids, segments


class QuestionAnswerDataset(Dataset):
    def __init__(self, 
                 df: DataFrame,
                 target: List[str],
                 tokenizer: BasicTokenizer,
                 mode: str,  # "question" or "answer"
                 train_mode: bool = True,
                 pre_pad: bool = False,
                 **kwargs):
        self.df: DataFrame = df
        self.target = target
        self.train_mode = train_mode
        self.tokenizer = tokenizer
        self.PAD = self.tokenizer.pad_token_id
        self.PAD_TOKEN = self.tokenizer.pad_token
        self.pre_pad = pre_pad
        self.mode = mode

    def pad_foo(self, tokens_list, max_size, pad_value=None) -> list:
        """
        Arguments:
            tokens_list - list of strings (tokens)
            max_size - int, maximal size of sequnece
            pad_value - str, value to use for padding, if not specified then will be used pad value from tokenizer
            pre_pad - use padding before sequence (<pad>, ... <pad>, begin, ..., end) or after sequence (begin, ..., end, <pad>, ..., <pad>)
        """
        pad_value = self.PAD if pad_value is None else pad_value
        _PAD = []
        if len(tokens_list) < max_size:
            _PAD = [pad_value] * (max_size - len(tokens_list))
        if self.pre_pad:
            tokens_list = _PAD + tokens_list
        else: 
            tokens_list = tokens_list + _PAD
        return tokens_list

    def __len__(self):
        return self.df.shape[0]

    def select_tokens(self, tokens, max_num):
        if len(tokens) <= max_num:
            return tokens
        if self.train_mode:
            num_remove = len(tokens) - max_num
            remove_start = np.random.randint(0, len(tokens) - num_remove - 1)
            return tokens[:remove_start] + tokens[remove_start + num_remove:]
        else:
            return tokens[:max_num // 2] + tokens[-(max_num - max_num // 2):]

    @staticmethod
    def clean_text(text: str) -> str:
        # # code in spaces
        # text = re.sub(r"(e\.g\.| |\?|this|:|;|output)(\n{2,3}([\s\S]+?\n)\n{2})", "\1 code ", text)
        
        text = html.unescape(text)
        # text = re.sub(r"\<[\s\S]+?\>", " code block ", text)
        
        # text = re.sub(r"((if|for)\W*\([\s\S]+?\)\W\{[\s\S]+\}|\{[\s\S]+\})", " code block ", text)
        # text = re.sub(r"( \$[\S]+ |\$[\s\S]+?;\n)", " code ", text)
        # text = re.sub(r"\/\/[\s\S]+?\n", " code comment ", text)
        # text = re.sub(r"\#[\s\S]+?\n", " code comment ", text)
        # text = re.sub(r"\/\*[\s\S]+\/\*", " code comment ", text)
        # text = re.sub(r"\([\s\S\W]+?\)\W+\{[\s\S]+\}", " function ", text)

        # latex equations
        # text = re.sub(r" ?(\$[\s\S]+?\$|\$\$[\s\S]+?\$\$) ", " equation ", text)

        return text

    @staticmethod
    def glue_title_and_question(t: str, q: str) -> str:
        t = t.strip()
        last_char = t[-1]
        sep_symbol = " " if last_char in {".", ",", ":", ";", "?", "!", "-"} else ","
        return t + sep_symbol + q

    def __getitem__(self, idx):
        index = self.df.index[idx]

        if self.mode == "question":
            title = self.df.at[index, "question_title"]
            body = self.clean_text(self.df.at[index, "question_body"])
            text = self.glue_title_and_question(title, body)
        else:
            text = self.clean_text(self.df.at[index, "answer"])
        
        tokens = self.tokenizer.tokenize(text)
        tokens = self.select_tokens(tokens, MAX_LEN)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = self.pad_foo(token_ids, MAX_LEN, self.tokenizer.pad_token_id)

        token_ids = torch.LongTensor(token_ids)
        target = torch.FloatTensor([self.df.at[index, c] for c in self.target])

        res = {
            "sequences": token_ids,
            "targets": target
        }
        return res


class AllInSequenceDataset(TransformerFieldsDataset):

    @staticmethod
    def glue_title_and_question(t: str, q: str) -> str:
        t = t.strip()
        last_char = t[-1]
        sep_symbol = " " if last_char in {".", ",", ":", ";", "?", "!", "-"} else ","
        return t + sep_symbol + q

    def build_tokens_and_segments(self, title: str, question: str, answer: str, category: str, host: str):
        """
        Input will be represented as: 
        <CLS> {category tokens} <SEP> {host tokens} <SEP> {title & question tokens} <SEP> {asnwer tokens} <SEP>

        With mask:
        0 ....................................................................... 0   1 ................... 1
        """
        # first part of input
        category = self.tokenizer.tokenize(category) + [self.tokenizer.sep_token]    # list of tokens (strings)
        host = self.tokenizer.tokenize(host) + [self.tokenizer.sep_token]    # list of tokens (strings)
        title_question = self.glue_title_and_question(title, question)  # str
        title_body_tokens = self.tokenizer.tokenize(title_question)  # list of tokens (strings)
        title_body_tokens = self._select_tokens(title_body_tokens, MAX_QUESTION_LEN)  # list of tokens (strings)
        title_body_tokens_ids = self.tokenizer.convert_tokens_to_ids(title_body_tokens)  # list of integers
        # second part of input
        ans_tokens = self.tokenizer.tokenize(answer)  # list of tokens (strings)
        ans_tokens = self._select_tokens(ans_tokens, MAX_ANSWER_LEN)  # list of tokens (strings)
        ans_tokens_ids = self.tokenizer.convert_tokens_to_ids(ans_tokens)  # list of integers
        # outputs
        token_ids = self.tokenizer.build_inputs_with_special_tokens(title_body_tokens_ids, ans_tokens_ids)  # list of integers
        segments = self.tokenizer.create_token_type_ids_from_sequences(title_body_tokens_ids, ans_tokens_ids)  # list of integers
        return token_ids, segments

    def __getitem__(self, idx):
        index = self.df.index[idx]
        title = self.df.at[index, "question_title"]
        body = self.df.at[index, "question_body"]
        answer = self.df.at[index, "answer"]
        category = self.df.at[index, "category"]
        host = self.df.at[index, "host"]

        # combine fields into one sequence
        token_ids, segments = self.build_tokens_and_segments(title, body, answer, category, host)
        # pad sequneces if needed
        token_ids = self._pad_foo(token_ids, MAX_LEN, self.tokenizer.pad_token_id)
        segments = self._pad_foo(segments, MAX_LEN, self.tokenizer.pad_token_id)
        # converting to tensors
        token_ids = torch.LongTensor(token_ids)
        segments = torch.LongTensor(segments) 
        target = torch.FloatTensor([self.df.at[index, c] for c in self.target])
        res = {
            "sequences": token_ids, 
            "segments": segments, 
            "targets": target
        }
        return res


class TwoSidedTransformerFieldsDataset(Dataset):
    def __init__(self,
                 df: DataFrame,
                 target: List[str],
                 tokenizer: BasicTokenizer,
                 train_mode: bool = True,
                 pre_pad: bool = False,
                 **kwargs):
        self.df: DataFrame = df
        self.target = target
        self.train_mode = train_mode
        self.tokenizer = tokenizer
        self.PAD_ID = self.tokenizer.pad_token_id
        self.PAD_TOKEN = self.tokenizer.pad_token
        self.pre_pad = pre_pad

    def __len__(self) -> int:
        return len(self.df)

    def pad_foo(self, tokens_list, max_size, pad_value=0) -> list:
        """
        Arguments:
            tokens_list - list of strings (tokens)
            max_size - int, maximal size of sequnece
            pad_value - str, value to use for padding, if not specified then will be used pad value from tokenizer
            pre_pad - use padding before sequence (<pad>, ... <pad>, begin, ..., end) or after sequence (begin, ..., end, <pad>, ..., <pad>)
        """
        PAD = [pad_value] * (max_size - len(tokens_list)) if len(tokens_list) < max_size else []
        tokens_list = PAD + tokens_list if self.pre_pad else tokens_list + PAD
        return tokens_list
    
    def _select_tokens(self, tokens, max_num):
        if len(tokens) <= max_num:
            return tokens
        if self.train_mode:
            num_remove = len(tokens) - max_num
            remove_start = np.random.randint(0, len(tokens) - num_remove - 1)
            return tokens[:remove_start] + tokens[remove_start + num_remove:]
        else:
            return tokens[:max_num // 2] + tokens[-(max_num - max_num // 2):]

    def __getitem__(self, idx: int) -> dict:
        index = self.df.index[idx]
        title = self.df.at[index, "question_title"]
        body = self.df.at[index, "question_body"]
        answer = self.df.at[index, "answer"]

        title_body_tokens = self.tokenizer.tokenize(title + "," + body)
        title_body_tokens = self._select_tokens(
            title_body_tokens, 
            max_num=MAX_LEN - 2
        )
        ans_tokens = self.tokenizer.tokenize(answer)
        ans_tokens = self._select_tokens(
            ans_tokens, 
            max_num=MAX_LEN - 2
        )

        first_part = [self.tokenizer.cls_token] + title_body_tokens + [self.tokenizer.sep_token]
        first_part_ids = self.tokenizer.convert_tokens_to_ids(first_part)
        first_part_ids = self.pad_foo(first_part_ids, MAX_LEN, self.PAD_ID)

        second_part = [self.tokenizer.cls_token] + ans_tokens + [self.tokenizer.sep_token]
        second_part_ids = self.tokenizer.convert_tokens_to_ids(second_part)
        second_part_ids = self.pad_foo(second_part_ids, MAX_LEN, self.PAD_ID)

        first_part_ids = torch.LongTensor(first_part_ids)
        second_part_ids = torch.LongTensor(second_part_ids)
        target = torch.FloatTensor([self.df.at[index, c] for c in self.target])

        res = {
            "question": first_part_ids,
            "answer": second_part_ids,
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

    def __len__(self):
        return self.df.shape[0]

    def _tokenize(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.tokenize(text)
        # for too long sequences - get subsequence
        if len(tokens) > MAX_LEN - 1:
            start_idx = np.random.randint(0, len(tokens) - MAX_LEN + 1)
            tokens = tokens[start_idx:start_idx + MAX_LEN - 1]
        tokens = [self.tokenizer.cls_token] + tokens
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

        fields["targets"] = torch.FloatTensor([self.df.at[index, c] for c in self.target])
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
