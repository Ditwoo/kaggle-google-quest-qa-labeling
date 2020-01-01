#!/bin/bash
CONF=configs/transformers_base_with_categorical_features.yml
INPUT_TYPE='transformers-with-categories'
ONAME_PATTERN=bert_base_uncased_with_cat_features

CHECKPOINT=logs/folds/bert_base_uncased_cat_features_00/checkpoints/stage2.11.pth  # 0.3888
OUTPUT=${ONAME_PATTERN}_00.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=logs/folds/bert_base_uncased_cat_features_01/checkpoints/stage1.5.pth   # 0.3784
OUTPUT=${ONAME_PATTERN}_01.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=logs/folds/bert_base_uncased_cat_features_02/checkpoints/stage2.11.pth  # 0.3988
OUTPUT=${ONAME_PATTERN}_02.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=logs/folds/bert_base_uncased_cat_features_03/checkpoints/stage2.11.pth  # 0.3884
OUTPUT=${ONAME_PATTERN}_03.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=logs/folds/bert_base_uncased_cat_features_04/checkpoints/stage4.21.pth
OUTPUT=${ONAME_PATTERN}_04.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}
echo "Traced all models based on config - ${CONF}."

ARCHIVE_NAME=bert_base_uncased_with_cat_features.zip
zip -r ${ARCHIVE_NAME} bert_base_uncased_with_cat_features_*
echo "Compressed to archive - ${ARCHIVE_NAME}."
