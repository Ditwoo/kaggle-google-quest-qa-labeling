#!/bin/bash

FOLD_IDX=2
# will be used this files insted files in config
export TRAIN_PICKLE=data/folds/tgkf_train_${FOLD_IDX}.pkl
export VALID_PICKLE=data/folds/tgkf_valid_${FOLD_IDX}.pkl

CONF=configs/roberta_large_with_cfs.yml
LOGDDIR=logs/folds/roberta_large_cfs_${FOLD_IDX}_0
catalyst-dl run --expdir src --logdir ${LOGDDIR} --config ${CONF} --verbose

CONF=configs/roberta_large_with_cfs_continue.yml
NEW_LOGDDIR=logs/folds/roberta_large_cfs_${FOLD_IDX}_1
export BEST_STATE=${LOGDDIR}/checkpoints/best.pth
catalyst-dl run --expdir src --logdir ${NEW_LOGDDIR} --config ${CONF} --verbose
