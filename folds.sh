CONF=configs/transformers_base.yml

echo "---------------------------------------------------------------------------------"

for (( i = 2; i < 5; i++ ))
do
    LOGDIR=logs/folds/bert_base_uncased_0${i}
    export TRAIN_PICKLE=data/folds/tgkf_train_0${i}.pkl
    export VALID_PICKLE=data/folds/tgkf_valid_0${i}.pkl

    catalyst-dl run --expdir src --logdir ${LOGDIR} --config ${CONF}
    echo "---------------------------------------------------------------------------------"
done
