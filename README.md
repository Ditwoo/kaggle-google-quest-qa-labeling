## [Kaggle Google QUEST Q&A Labeling](https://www.kaggle.com/c/google-quest-challenge/overview/notebooks-requirements)

Firstly need to generate data, example is [here](data/splits.ipynb).

I suggest to create bash file with content like this:

```bash
#!/bin/bash

# fold number to use for loading data
FOLD_IDX=123
# paths to data
export TRAIN_PICKLE=data/folds/tgkf_train_${FOLD_IDX}.pkl
export VALID_PICKLE=data/folds/tgkf_valid_${FOLD_IDX}.pkl
# notifying about training progress
export CATALYST_TELEGRAM_TOKEN="<bot token>"
export CATALYST_TELEGRAM_CHAT_ID="<chat id>"
# experiment variables
NAME=<experiment name>
CONF=configs/<config>.yml
LOGDDIR=logs/folds/<logdir>
# remove if exists
[ -e ${LOGDDIR} ] && rm -rf ${LOGDDIR} && echo "Removed existed dir with logs - '${LOGDDIR}'"
catalyst-dl run --expdir src --logdir ${LOGDDIR} --config ${CONF} --verbose
```

because then you need only to `bash <file>.sh` and experiment will run.
