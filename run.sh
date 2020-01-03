#!/bin/bash
CONF=configs/transformers_base_with_cfs.yml
LOGDDIR=logs/folds/bert_base_uncased_cfs_04

# will be used this files insted files in config
export TRAIN_PICKLE=data/folds/tgkf_train_04.pkl
export VALID_PICKLE=data/folds/tgkf_valid_04.pkl

catalyst-dl run --expdir src --logdir ${LOGDDIR} --config ${CONF} --verbose
