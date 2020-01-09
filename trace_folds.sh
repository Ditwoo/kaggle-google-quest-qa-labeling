#!/bin/bash

CONF=configs/roberta_large_with_cfs.yml
INPUT_TYPE='roberta-categories-stats'
LOG_DIR_PREFIX=logs/folds/roberta_large_cfs
ONAME_PATTERN=rb_lcfs

CHECKPOINT=${LOG_DIR_PREFIX}_0_1/checkpoints/best.pth  # 0.3909
OUTPUT=${ONAME_PATTERN}_0.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_1_1/checkpoints/best.pth  # 0.3878
OUTPUT=${ONAME_PATTERN}_1.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_2_1/checkpoints/best.pth  # 0.3900
OUTPUT=${ONAME_PATTERN}_2.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

# CHECKPOINT=${LOG_DIR_PREFIX}_03/checkpoints/stage2.9.pth  # 0.3879
# OUTPUT=${ONAME_PATTERN}_03.pt
# python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

# CHECKPOINT=${LOG_DIR_PREFIX}_04/checkpoints/stage2.9.pth  # 0.3810
# OUTPUT=${ONAME_PATTERN}_04.pt
# python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}
echo "Traced all models based on config - ${CONF}."

ARCHIVE_NAME=${ONAME_PATTERN}.zip
zip -r ${ARCHIVE_NAME} ${ONAME_PATTERN}_*
echo "Compressed to archive - ${ARCHIVE_NAME}."
