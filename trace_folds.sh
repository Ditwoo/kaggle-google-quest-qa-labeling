#!/bin/bash


CONF=configs/xlnet_large_cased.yml
INPUT_TYPE='transformers'
LOG_DIR_PREFIX=logs/folds/xlnet_large_cased
ONAME_PATTERN=xlc_ls_3f

CHECKPOINT=${LOG_DIR_PREFIX}_0_1/checkpoints/best.pth  # 0.3907
OUTPUT=${ONAME_PATTERN}_0.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_1_1/checkpoints/best.pth  # 0.3914
OUTPUT=${ONAME_PATTERN}_1.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

CHECKPOINT=${LOG_DIR_PREFIX}_2_1/checkpoints/best.pth  # 0.3829
OUTPUT=${ONAME_PATTERN}_2.pt
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

# CHECKPOINT=${LOG_DIR_PREFIX}_03/checkpoints/stage3.12.pth  # 0.3840
# OUTPUT=${ONAME_PATTERN}_03.pt
# python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

# CHECKPOINT=${LOG_DIR_PREFIX}_04/checkpoints/stage3.11.pth  # 0.3744
# OUTPUT=${ONAME_PATTERN}_04.pt
# python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --input-type ${INPUT_TYPE} --out ${OUTPUT}

echo "Traced all models based on config - ${CONF}."

ARCHIVE_NAME=${ONAME_PATTERN}.zip
zip -r ${ARCHIVE_NAME} ${ONAME_PATTERN}_*
echo "Compressed to archive - ${ARCHIVE_NAME}."
