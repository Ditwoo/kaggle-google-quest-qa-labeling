#!/bin/bash
CONF=configs/roberta_base_with_cfs.yml
INPUT_TYPE='roberta-categories-stats'
LOG_DIR_PREFIX=logs/folds/roberta_base_cfs
ONAME_PATTERN=rb_bcfs

CHECKPOINT=${LOG_DIR_PREFIX}_00/checkpoints/stage4.19.pth  # 0.3904
OUTPUT=${ONAME_PATTERN}_00.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_01/checkpoints/stage1.7.pth  # 0.3773
OUTPUT=${ONAME_PATTERN}_01.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_02/checkpoints/stage4.15.pth  # 0.3993
OUTPUT=${ONAME_PATTERN}_02.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_03/checkpoints/stage2.9.pth  # 0.3879
OUTPUT=${ONAME_PATTERN}_03.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_04/checkpoints/stage2.9.pth  # 0.3810
OUTPUT=${ONAME_PATTERN}_04.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}
echo "Traced all models based on config - ${CONF}."

ARCHIVE_NAME=${ONAME_PATTERN}.zip
zip -r ${ARCHIVE_NAME} ${ONAME_PATTERN}_*
echo "Compressed to archive - ${ARCHIVE_NAME}."
