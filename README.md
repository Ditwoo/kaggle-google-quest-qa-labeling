## [Kaggle Google QUEST Q&A Labeling](https://www.kaggle.com/c/google-quest-challenge/overview/notebooks-requirements)

Firstly need to generate data, example is [here](data/splits.ipynb).

I suggest to create bash file with content like this:

```bash
CONF=configs/<config>.yml
EXPDIR=src
LOGDDIR=logs/<logdir>

catalyst-dl run --expdir ${EXPDIR} --logdir ${LOGDDIR} --config ${CONF} --verbose
```

because then you need only to `bash <file>.sh` and experiment will run.
