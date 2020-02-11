# [Kaggle Google QUEST Q&A Labeling](https://www.kaggle.com/c/google-quest-challenge/overview/notebooks-requirements)


<h2> <img src=".readme/bronze.png" style="width:15px;"> 110/1571 Place (top 8%) <img src=".readme/bronze.png" style="width:15px;"> </h2>


## Training & Tracing

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
# training process
catalyst-dl run --expdir src --logdir ${LOGDDIR} --config ${CONF} --verbose
```

and bash file for tracing models:

```bash
#!/bin/bash

CONF=<path to .yml file with config>
INPUT_TYPE='<model input type>'
LOG_DIR_PREFIX=<prefix path to logs directory>
ONAME_PATTERN=<output name pattern>

CHECKPOINT=${LOG_DIR_PREFIX}_0/checkpoints/best.pth
OUTPUT=${ONAME_PATTERN}_0.pt
python3 -m src.trace --config ${CONF} \
--state ${CHECKPOINT} \
--input-type ${INPUT_TYPE} \
--out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_1/checkpoints/best.pth
OUTPUT=${ONAME_PATTERN}_1.pt
python3 -m src.trace --config ${CONF} \
--state ${CHECKPOINT} \
--input-type ${INPUT_TYPE} \
--out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_2/checkpoints/best.pth
OUTPUT=${ONAME_PATTERN}_2.pt
python3 -m src.trace --config ${CONF} \
--state ${CHECKPOINT} \
--input-type ${INPUT_TYPE} \
--out ${OUTPUT}

echo "Traced all models based on config - ${CONF}."

ARCHIVE_NAME=${ONAME_PATTERN}.zip
zip -r ${ARCHIVE_NAME} ${ONAME_PATTERN}_*
echo "Compressed to archive - ${ARCHIVE_NAME}."
```

## Submission

Final submission consist of 8 models - __5 base__ and __3 large__ models trained on different number of folds - 5 and 3 folds respectively.

## Base Models

At the end of competition I found a perfect finetuning congiguration:

```yaml
model_params:
  model: <model>
  pretrain_dir: <tokenizer>
  num_classes: 30
  pad_token: <pad token index>

stages:
  state_params:
    main_metric: spearman
    minimize_metric: False

  data_params:
    num_workers: &nw 6
    batch_size: *nw
    tokenizer: <tokenizer>

  criterion_params:
    criterion: BCEWithLogitsLoss

  callbacks_params:
    loss:
      callback: CriterionCallback
      input_key: targets

    optim:
      callback: OptimizerCallback

    spearman:
      callback: SpearmanScoreCallback
      classes: 30

    saver:
      callback: CheckpointCallback

    early_stopping:
      callback: EarlyStoppingCallback
      patience: 2
      metric: spearman
      minimize: False
    
    telegram_logger:
      callback: TelegramLogger
      log_on_stage_start: False
      log_on_loader_start: False
      metric_names:
        - loss
        - spearman

  stage1:
    state_params:
      num_epochs: 10
    
    optimizer_params:
      optimizer: RAdam
      lr: 0.00001
      betas: &betas [0.9, 0.98]
      eps: &eps 0.000000001

  stage2:
    state_params:
      num_epochs: 5
    
    optimizer_params:
      optimizer: RAdam
      lr: 0.000001
      betas: *betas
      eps: *eps

  stage3:
    state_params:
      num_epochs: 5
    
    optimizer_params:
      optimizer: RAdam
      lr: 0.0000001
      betas: *betas
      eps: *eps
```

### 1. Bert Base Uncased with fold text statistics and categorical features

Config file - [`bert_base_wit_te_stats_and_cats.yml`](configs/bert_base_wit_te_stats_and_cats.yml) <br>
Dataset class - __FoldTFDCFSF__ <br>
Folds:

| files | score |
|-------|-------|
| [`tgkf_train_00.pkl`](data/folds/tgkf_train_00.pkl), [`tgkf_train_00.json`](data/folds/tgkf_train_00.json) | 0.3948 |
| [`tgkf_train_01.pkl`](data/folds/tgkf_train_01.pkl), [`tgkf_train_01.json`](data/folds/tgkf_train_01.json) | 0.3800 |
| [`tgkf_train_02.pkl`](data/folds/tgkf_train_02.pkl), [`tgkf_train_02.json`](data/folds/tgkf_train_02.json) | 0.3996 |
| [`tgkf_train_03.pkl`](data/folds/tgkf_train_03.pkl), [`tgkf_train_03.json`](data/folds/tgkf_train_03.json) | 0.3929 |
| [`tgkf_train_04.pkl`](data/folds/tgkf_train_04.pkl), [`tgkf_train_04.json`](data/folds/tgkf_train_04.json) | 0.3837 |


### 2. Bert Base Uncased with text statistics based on all data and categorical features

Config file - [`bert_base_wit_te_stats_and_cats.yml`](configs/bert_base_wit_te_stats_and_cats.yml) <br>
Dataset class - __TFDCFSF__ <br>
Folds:

| files | score |
|-------|-------|
| [`tgkf_train_00.pkl`](data/folds/tgkf_train_00.pkl) | 0.3901 |
| [`tgkf_train_01.pkl`](data/folds/tgkf_train_01.pkl) | 0.3742 |
| [`tgkf_train_02.pkl`](data/folds/tgkf_train_02.pkl) | 0.4055 |
| [`tgkf_train_03.pkl`](data/folds/tgkf_train_03.pkl) | 0.3941 |
| [`tgkf_train_04.pkl`](data/folds/tgkf_train_04.pkl) | 0.3790 |


### 3. Bert Base Cased

Config file - [`bert_base_cased_with_cfs.yml`](configs/bert_base_cased_with_cfs.yml) <br>
Dataset class - __TransformerFieldsDataset__ <br>
Folds:

| files | score |
|-------|-------|
| [`tgkf_train_00.pkl`](data/folds/tgkf_train_00.pkl) | 0.3855 |
| [`tgkf_train_01.pkl`](data/folds/tgkf_train_01.pkl) | 0.3688 |
| [`tgkf_train_02.pkl`](data/folds/tgkf_train_02.pkl) | 0.3957 |
| [`tgkf_train_03.pkl`](data/folds/tgkf_train_03.pkl) | 0.3827 |
| [`tgkf_train_04.pkl`](data/folds/tgkf_train_04.pkl) | 0.3822 |


### 4. Roberta Base with text statistincs and categorical features

Config file - [`roberta_base_with_cfs.yml`](configs/roberta_base_with_cfs.yml) <br>
Dataset class - __RFDCFSF__ <br>

| files | score |
|-------|-------|
| [`tgkf_train_00.pkl`](data/folds/tgkf_train_00.pkl) | 0.3904 |
| [`tgkf_train_01.pkl`](data/folds/tgkf_train_01.pkl) | 0.3773 |
| [`tgkf_train_02.pkl`](data/folds/tgkf_train_02.pkl) | 0.3993 |
| [`tgkf_train_03.pkl`](data/folds/tgkf_train_03.pkl) | 0.3879 |
| [`tgkf_train_04.pkl`](data/folds/tgkf_train_04.pkl) | 0.3810 |

### 5. XLNet Base

Config file - [`xlnet_base_cased_less_stages.yml`](configs/xlnet_base_cased_less_stages.yml) <br>
Dataset class - __XLNetDataset__ <br>

| files | score |
|-------|-------|
| [`tgkf_train_00.pkl`](data/folds/tgkf_train_00.pkl) | 0.3830 |
| [`tgkf_train_01.pkl`](data/folds/tgkf_train_01.pkl) | 0.3754 |
| [`tgkf_train_02.pkl`](data/folds/tgkf_train_02.pkl) | 0.3953 |
| [`tgkf_train_03.pkl`](data/folds/tgkf_train_03.pkl) | 0.3840 |
| [`tgkf_train_04.pkl`](data/folds/tgkf_train_04.pkl) | 0.3744 |


## Large Models

Overall large models was trained in two stages:

```yaml
...
stages:
  state_params:
    main_metric: spearman
    minimize_metric: False

  data_params:
    num_workers: &nw 2
    batch_size: *nw
    tokenizer: <tokenizer>

  criterion_params:
    criterion: BCEWithLogitsLoss

  callbacks_params:
    loss:
      callback: CriterionCallback
      input_key: targets

    optim:
      callback: OptimizerCallback
      accumulation_steps: 16

    spearman:
      callback: SpearmanScoreCallback
      classes: 30

    saver:
      callback: CheckpointCallback
    
    early_stopping:
      callback: EarlyStoppingCallback
      patience: 3
      metric: spearman
      minimize: False

  stage1:
    state_params:
      num_epochs: 10
    
    optimizer_params:
      optimizer: Adam
      lr: 0.00003
```

and then best state was additionaly trained:

```yaml
stages:
  state_params:
    main_metric: spearman
    minimize_metric: False

  data_params:
    num_workers: &nw 2
    batch_size: *nw
    tokenizer: <tokenizer>

  criterion_params:
    criterion: BCEWithLogitsLoss

  callbacks_params:
    loss:
      callback: CriterionCallback
      input_key: targets

    optim:
      callback: OptimizerCallback
      accumulation_steps: 16

    spearman:
      callback: SpearmanScoreCallback
      classes: 30

    saver:
      callback: CheckpointCallback
    
    early_stopping:
      callback: EarlyStoppingCallback
      patience: 3
      metric: spearman
      minimize: False

  stage1:
    state_params:
      num_epochs: 10
    
    optimizer_params:
      optimizer: Adam
      lr: 0.000005
```

### 6. Bert Large Uncased

Config files - [`bert_large_with_cfs.yml`](configs/bert_large_with_cfs.yml) -> [`bert_large_with_cfs_continue.yml`](configs/xlnet_large_cased_continue.yml) <br>
Dataset class - __TransformerFieldsDataset__ <br>

| files | score |
|-------|-------|
| [`tgkf_train_0.pkl`](data/folds/tgkf_train_0.pkl) | 0.3940 |
| [`tgkf_train_1.pkl`](data/folds/tgkf_train_1.pkl) | 0.3908 |
| [`tgkf_train_2.pkl`](data/folds/tgkf_train_2.pkl) | 0.3847 |

### 7. Roberta Large with text statistincs and categorical features

Config files - [`roberta_large_with_cfs.yml`](configs/roberta_large_with_cfs.yml) -> [`roberta_large_with_cfs_continue.yml`](configs/roberta_large_with_cfs_continue.yml) <br>
Dataset class - __RFDCFSF__ <br>

| files | score |
|-------|-------|
| [`tgkf_train_0.pkl`](data/folds/tgkf_train_0.pkl) | 0.3909 |
| [`tgkf_train_1.pkl`](data/folds/tgkf_train_1.pkl) | 0.3878 |
| [`tgkf_train_2.pkl`](data/folds/tgkf_train_2.pkl) | 0.3900 |

### 8. XLNet Large

Config files - [`xlnet_large_cased.yml`](configs/xlnet_large_cased.yml) -> [`xlnet_large_cased_continue.yml`](configs/xlnet_large_cased_continue.yml) <br>
Dataset class - __XLNetDataset__ <br>

| files | score |
|-------|-------|
| [`tgkf_train_0.pkl`](data/folds/tgkf_train_0.pkl) | 0.3907 |
| [`tgkf_train_1.pkl`](data/folds/tgkf_train_1.pkl) | 0.3914 |
| [`tgkf_train_2.pkl`](data/folds/tgkf_train_2.pkl) | 0.3829 |

## Generating submission

All models predictions was averaged and then rounded to 2 decimals with additional heuristics:

```python
...
def is_stackexchange(url: str) -> bool:
    return ("ell.stackexchange.com" in url) or ("english.stackexchange.com" in url)


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


def th_round(val, threshold=0.5, decimals=2):    
    return round_up(val, decimals) if val > threshold else truncate(val, decimals)
...

sub = ps.read_csv(data_dir / "sample_submission.csv")
sub[targets] = avg_preds

for col in targets:
    sub.loc[sub[col] >= 0.99, col] = 0.99
    sub.loc[sub[col] <= 0.01, col] = 0.01
    sub[col] = sub[col].apply(th_round)

sub["question_type_spelling"] = test_df["url"].apply(lambda u: 0.5 if is_stackexchange(u) else 0.0)

```