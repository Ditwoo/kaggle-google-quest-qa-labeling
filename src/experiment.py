import os
import json
import pickle
from collections import OrderedDict
from copy import copy

import numpy as np
import torch
import torch.nn as nn
from catalyst.dl import ConfigExperiment
from sklearn.model_selection import train_test_split
from transformers import (
    AlbertTokenizer,
    BertTokenizer, 
    BertTokenizerFast, 
    DistilBertTokenizer, 
    XLNetTokenizer, 
    RobertaTokenizer,
    GPT2Tokenizer,
)

from .datasets import (
    FieldsDataset,
    TokenizedFieldsDataset,
    SequencesCollator, 
    FieldsCollator,
    TransformerFieldsDataset,
    TransformersCollator,
    TransformerMultipleFieldsDataset,
    TransformerFieldsDatasetWithCategoricalFeatures,
    TFDCFSF,
    RFDCFSF,
    XFDCFSF,
    FoldTFDCFSF,
    FoldTFDCSF,
    TFDCC,
    JoinedTransformerFieldsDataset,
    TwoSidedTransformerFieldsDataset,
    QuestionAnswerDataset,
    AllInSequenceDataset,
    BertDataset,
    XLNetDataset,
    GPT2Dataset,
)
from .datasets.augmentations import (
    CombineSeqs, 
    SeqElementsShuffler, 
    Compose
)
# from .datasets import VectorDataset


TEXT_COLS = ["question_title", "question_body", "answer", "category", "host"]
QUESTION_TARGETS = [
    "question_asker_intent_understanding", 
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
]  # 21 variables
ANSWER_TARGETS = [
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]  # 9 variables
TARGETS = QUESTION_TARGETS + ANSWER_TARGETS  # 30 variables
TARGETS_ENV_MAP = {
    "QUESTION": QUESTION_TARGETS,
    "ANSWER": ANSWER_TARGETS,
    "BOTH": TARGETS,
}


class Experiment(ConfigExperiment):

    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):
        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if hasattr(model_, "bert"):
            if stage.startswith("stage_freezed"):
                for param in model_.bert.parameters():
                    param.requires_grad = False
            else:
                for param in model_.bert.parameters():
                    param.requires_grad = True

        return model_

    def transformer_get_datasets(self,
                                 stage: str,
                                 tokenizer: str,
                                 train_pickle: str = None,
                                 valid_pickle: str = None,
                                 **kwargs):
        # if "STATS_CONFIG" in os.environ and os.environ["STATS_CONFIG"]:
        #     stats_config = os.environ["STATS_CONFIG"]
        # else:
        #     raise ValueError("There is no specified 'STATS_CONFIG' env variable!")
            
        # with open(stats_config, "r") as f:
        #     stats_config = json.load(f)

        # from pprint import pprint
        # print()
        # pprint(stats_config)
        # print(flush=True)
        
        if "TRAIN_PICKLE" in os.environ and os.environ["TRAIN_PICKLE"]:
            train_pickle = os.environ["TRAIN_PICKLE"]

        # if "TARGETS" in os.environ and os.environ["TARGETS"] in {"QUESTION", "ANSWER", "BOTH"}:
        #     targets = os.environ["TARGETS"]
        # else:
        #     targets = "BOTH"

        # if targets == "QUESTION":
        #     dataset_mode = "question"
        # elif targets == "ANSWER":
        #     dataset_mode = "answer"
        # else:
        #     raise KeyError("Expected one of two modes - QUESTION or ANSWER !")

        # targets = TARGETS_ENV_MAP.get(targets, TARGETS_ENV_MAP["BOTH"])
        targets = TARGETS
        tokenizer_cls = XLNetTokenizer
        dataset_cls =  XLNetDataset #AllInSequenceDataset # TFDCC # FoldTFDCSF # TransformerFieldsDataset # TwoSidedTransformerFieldsDataset #JoinedTransformerFieldsDataset # FoldTFDCSF # FoldTFDCFSF # XFDCFSF # RFDCFSF # TFDCFSF

        with open(train_pickle, "rb") as f:
            df = pickle.load(f)

        # print(f" * Training dataset mode: '{dataset_mode}'")
        print(f" * Targets: {', '.join(targets)}")

        datasets = OrderedDict()
        datasets["train"] = dict(
            dataset=dataset_cls(
                # stats_config=stats_config,
                df=df, 
                target=targets,
                # mode=dataset_mode,
                tokenizer=tokenizer_cls.from_pretrained(tokenizer),
            ),
            shuffle=True,
        )
        print(" * Train size -", len(datasets["train"]["dataset"]), flush=True)
        
        if "VALID_PICKLE" in os.environ and os.environ["VALID_PICKLE"]:
            valid_pickle = os.environ["VALID_PICKLE"]

        with open(valid_pickle, "rb") as f:
            df = pickle.load(f)

        datasets["valid"] = dict(
            dataset=dataset_cls(
                # stats_config=stats_config,
                df=df, 
                target=targets,
                # mode=dataset_mode,
                tokenizer=tokenizer_cls.from_pretrained(tokenizer),
            ),
            shuffle=False,
        )
        print(" * Valid size -", len(datasets["valid"]["dataset"]), flush=True)
        return datasets

    def rnn_get_datasets(self,
                         stage: str,
                         tokenizer_dir: str,
                         seq_percentile: int = 75,
                         train_pickle: str = None,
                         valid_pickle: str = None,
                         **kwargs):

        if "TRAIN_PICKLE" in os.environ and os.environ["TRAIN_PICKLE"]:
            train_pickle = os.environ["TRAIN_PICKLE"]

        with open(train_pickle, "rb") as f:
            df = pickle.load(f)

        max_len = 1500
        print(f"Train size - {df.shape}")
        datasets = OrderedDict()
        datasets["train"] = dict(
            dataset=TokenizedFieldsDataset(
                df=df, 
                feature_cols=TEXT_COLS, 
                target=TARGETS, 
                tokenizer_dir=tokenizer_dir,
            ),
            collate_fn=FieldsCollator(
                fields=TEXT_COLS,
                ignore_fields=["category", "host"],
                max_len=max_len, 
                percentile=seq_percentile
            ),
            shuffle=True,
        )
        
        if "VALID_PICKLE" in os.environ and os.environ["VALID_PICKLE"]:
            valid_pickle = os.environ["VALID_PICKLE"]

        with open(valid_pickle, "rb") as f:
            df = pickle.load(f)

        print(f"Valid shapes - {df.shape}")
        datasets["valid"] = dict(
            # dataset=FieldsDataset(df, text_cols, TARGETS, CombineSeqs(text_cols, "seq", glue_token=0)),
            dataset=TokenizedFieldsDataset(
                df=df, 
                feature_cols=TEXT_COLS, 
                target=TARGETS, 
                tokenizer_dir=tokenizer_dir,
            ),
            collate_fn=FieldsCollator(
                fields=TEXT_COLS, 
                ignore_fields=["category", "host"],
                max_len=max_len, 
                percentile=seq_percentile
            ),
        )
        return datasets

    def get_datasets(self, stage: str, **kwargs):
        return self.transformer_get_datasets(stage, **kwargs)
