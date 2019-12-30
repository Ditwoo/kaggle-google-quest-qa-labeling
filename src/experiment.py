import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from catalyst.dl import ConfigExperiment
from sklearn.model_selection import train_test_split

from .datasets import (
    FieldsDataset,
    TokenizedFieldsDataset,
    SequencesCollator, 
    FieldsCollator,
    TransformerFieldsDataset,
    TransformersCollator,
    TransformerMultipleFieldsDataset
)
from .datasets.augmentations import (
    CombineSeqs, 
    SeqElementsShuffler, 
    Compose
)
# from .datasets import VectorDataset


TEXT_COLS = ["question_title", "question_body", "answer", "category", "host"]
TARGETS = [
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
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]


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
                                 tokenizer_dir: str,
                                 train_pickle: str = None,
                                 valid_pickle: str = None,
                                 **kwargs):
        if "TRAIN_PICKLE" in os.environ and os.environ["TRAIN_PICKLE"]:
            train_pickle = os.environ["TRAIN_PICKLE"]

        with open(train_pickle, "rb") as f:
            df = pickle.load(f)

        print(f"Train shapes - {df.shape}")
        datasets = OrderedDict()
        datasets["train"] = dict(
            dataset=TransformerFieldsDataset(
                df=df, 
                target=TARGETS, 
                tokenizer_dir=tokenizer_dir,
            ),
            shuffle=True,
        )
        
        if "VALID_PICKLE" in os.environ and os.environ["VALID_PICKLE"]:
            valid_pickle = os.environ["VALID_PICKLE"]

        with open(valid_pickle, "rb") as f:
            df = pickle.load(f)

        print(f"Valid shapes - {df.shape}")
        datasets["valid"] = dict(
            dataset=TransformerFieldsDataset(
                df=df, 
                target=TARGETS, 
                tokenizer_dir=tokenizer_dir,
            ),
            shuffle=False,
        )
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
