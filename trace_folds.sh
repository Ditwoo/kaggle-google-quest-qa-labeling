#!/bin/bash
CONF=configs/transformers_base_with_cfs.yml
INPUT_TYPE='transformers-categories-stats'
ONAME_PATTERN=bert_base_uncased_cfs

CHECKPOINT=logs/folds/bert_base_uncased_cfs_00/checkpoints/stage4.21.pth  # 0.3901
OUTPUT=${ONAME_PATTERN}_00.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=logs/folds/bert_base_uncased_cfs_01/checkpoints/stage2.12.pth  # 0.3742
OUTPUT=${ONAME_PATTERN}_01.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=logs/folds/bert_base_uncased_cfs_02/checkpoints/stage2.11.pth  # 0.4055
OUTPUT=${ONAME_PATTERN}_02.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=logs/folds/bert_base_uncased_cfs_03/checkpoints/stage2.11.pth  # 0.3941
OUTPUT=${ONAME_PATTERN}_03.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=logs/folds/bert_base_uncased_cfs_04/checkpoints/stage1.5.pth  # 0.3790
OUTPUT=${ONAME_PATTERN}_04.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}
echo "Traced all models based on config - ${CONF}."

ARCHIVE_NAME=bert_base_uncased_cfs.zip
zip -r ${ARCHIVE_NAME} ${ONAME_PATTERN}_*
echo "Compressed to archive - ${ARCHIVE_NAME}."
