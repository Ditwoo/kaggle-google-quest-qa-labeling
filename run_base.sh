#!/bin/bash

FOLD_IDX=04
# will be used this files insted files in config
export TRAIN_PICKLE=data/folds/tgkf_train_${FOLD_IDX}.pkl
export VALID_PICKLE=data/folds/tgkf_valid_${FOLD_IDX}.pkl

NAME=bert_base_cased_cfs
CONF=configs/bert_base_cased_with_cfs.yml
LOGDDIR=logs/folds/${NAME}_${FOLD_IDX}
# remove if exists
[ -e ${LOGDDIR} ] && rm -rf ${LOGDDIR} && echo "Removed existed dir with logs - '${LOGDDIR}'"
catalyst-dl run --expdir src --logdir ${LOGDDIR} --config ${CONF} --verbose
