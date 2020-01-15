#!/bin/bash

FOLD_IDX=04
# will be used this files insted files in config
export TRAIN_PICKLE=data/folds/tgkf_train_${FOLD_IDX}.pkl
export VALID_PICKLE=data/folds/tgkf_valid_${FOLD_IDX}.pkl
export STATS_CONFIG=data/folds/tgkf_train_${FOLD_IDX}.json  

NAME=bert_base_uncased_stats_and_te
CONF=configs/bert_base_uncased_with_stats_and_te.yml
LOGDDIR=logs/folds/${NAME}_${FOLD_IDX}
# remove if exists
[ -e ${LOGDDIR} ] && rm -rf ${LOGDDIR} && echo "Removed existed dir with logs - '${LOGDDIR}'"
catalyst-dl run --expdir src --logdir ${LOGDDIR} --config ${CONF} --verbose