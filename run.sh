CONF=configs/transformers_large_2.yml
LOGDDIR=logs/folds/tlgkf_2_2

# specify env vars (if specified - passed args to config will be ignored)
export TRAIN_PICKLE=data/folds/tgkf_train_2.pkl
export VALID_PICKLE=data/folds/tgkf_valid_2.pkl

catalyst-dl run --expdir src --logdir ${LOGDDIR} --config ${CONF} --verbose
