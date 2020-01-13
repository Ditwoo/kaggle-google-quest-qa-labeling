#!/bin/bash

CONF=configs/bert_base_cased_with_cfs.yml
INPUT_TYPE='transformers'
LOG_DIR_PREFIX=logs/folds/bert_base_cased_cfs
ONAME_PATTERN=bb_c

CHECKPOINT=${LOG_DIR_PREFIX}_00/checkpoints/stage4.13.pth  # 0.3855
OUTPUT=${ONAME_PATTERN}_00.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_01/checkpoints/stage3.11.pth  # 0.3688
OUTPUT=${ONAME_PATTERN}_01.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_02/checkpoints/stage3.10.pth  # 0.3957
OUTPUT=${ONAME_PATTERN}_02.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_03/checkpoints/stage2.8.pth  # 0.3827
OUTPUT=${ONAME_PATTERN}_03.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_04/checkpoints/stage3.10.pth  # 0.3822
OUTPUT=${ONAME_PATTERN}_04.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}
echo "Traced all models based on config - ${CONF}."

ARCHIVE_NAME=${ONAME_PATTERN}.zip
zip -r ${ARCHIVE_NAME} ${ONAME_PATTERN}_*
echo "Compressed to archive - ${ARCHIVE_NAME}."
