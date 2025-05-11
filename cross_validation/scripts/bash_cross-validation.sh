#!/bin/bash
export PYTHONPATH="$PYTHONPATH:./"

DEVICES=1
NUM_NODES=1
SEED=3407
MAX_EPOCHS=120
ACC_BATCH=1
LR=3e-5
PATIENCE=20
STRATEGY="auto"
BATCH_SIZE=12
HIDDEN_SIZE=1280
NUM_HEADS=4
DROPOUT=0.1
DATA_PATH="./data/csv/AB1101.csv"
DATA_NAME="AB1101"
MODEL_LOCATE="./model/esm2_650m"
loss="mse"

python ./src_ab1101/trainer.py \
  --batch_size $BATCH_SIZE \
  --seed $SEED \
  --max_epochs $MAX_EPOCHS \
  --patience $PATIENCE \
  --lr $LR \
  --devices $DEVICES \
  --num_nodes $NUM_NODES \
  --strategy $STRATEGY \
  --accumulate_grad_batches $ACC_BATCH \
  --data_path $DATA_PATH \
  --data_name $DATA_NAME \
  --hidden_size $HIDDEN_SIZE \
  --num_heads $NUM_HEADS \
  --dropout $DROPOUT \
  --model_locate $MODEL_LOCATE \
  --loss $loss \
  --freeze_backbone