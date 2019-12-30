
CONF=configs/transformers_base.yml

CHECKPOINT=logs/folds/bert_base_uncased_00/checkpoints/stage2.11.pth
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --out bert_base_uncased_00.pt --transformers

CHECKPOINT=logs/folds/bert_base_uncased_01/checkpoints/stage1.5.pth
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --out bert_base_uncased_01.pt --transformers

CHECKPOINT=logs/folds/bert_base_uncased_02/checkpoints/stage1.5.pth
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --out bert_base_uncased_02.pt --transformers

CHECKPOINT=logs/folds/bert_base_uncased_03/checkpoints/stage2.11.pth
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --out bert_base_uncased_03.pt --transformers

CHECKPOINT=logs/folds/bert_base_uncased_04/checkpoints/stage4.21.pth
python3 -m src.trace --config ${CONF} --state ${CHECKPOINT} --out bert_base_uncased_04.pt --transformers
