CONF=configs/transformers_large_2.yml
LOGDDIR=logs/folds/bert_large_uncased_00_2

# specify env vars (if specified - passed args to config will be ignored)
# NOTE: used models with split usign group kfold
export TRAIN_PICKLE=data/folds/tgkf_train_00.pkl
export VALID_PICKLE=data/folds/tgkf_valid_00.pkl

catalyst-dl run --expdir src --logdir ${LOGDDIR} --config ${CONF} --verbose
