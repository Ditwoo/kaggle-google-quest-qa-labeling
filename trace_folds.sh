#!/bin/bash

CONF=configs/bert_base_uncased_with_stats_and_te.yml
INPUT_TYPE='transformers-stats'
LOG_DIR_PREFIX=logs/folds/bert_base_uncased_stats_and_te
ONAME_PATTERN=bb_u_ste

CHECKPOINT=${LOG_DIR_PREFIX}_00/checkpoints/stage4.13.pth  # 0.3948
OUTPUT=${ONAME_PATTERN}_00.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_01/checkpoints/stage4.12.pth  # 0.3800
OUTPUT=${ONAME_PATTERN}_01.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_02/checkpoints/stage1.5.pth  # 0.3996
OUTPUT=${ONAME_PATTERN}_02.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_03/checkpoints/stage2.7.pth  # 0.3929
OUTPUT=${ONAME_PATTERN}_03.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_04/checkpoints/stage2.7.pth  # 0.3837
OUTPUT=${ONAME_PATTERN}_04.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}
echo "Traced all models based on config - ${CONF}."

# ARCHIVE_NAME=${ONAME_PATTERN}.zip
# zip -r ${ARCHIVE_NAME} ${ONAME_PATTERN}_*
# echo "Compressed to archive - ${ARCHIVE_NAME}."
