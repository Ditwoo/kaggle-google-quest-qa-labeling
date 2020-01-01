#!/bin/bash
CONF=configs/transformers_base_with_categorical_features.yml
LOGDDIR=logs/folds/bert_base_uncased_cat_features_00

# specify env vars (if specified - passed args to config will be ignored)
# NOTE: used models with split usign group kfold
export TRAIN_PICKLE=data/folds/tgkf_train_00.pkl
export VALID_PICKLE=data/folds/tgkf_valid_00.pkl

catalyst-dl run --expdir src --logdir ${LOGDDIR} --config ${CONF} --verbose
