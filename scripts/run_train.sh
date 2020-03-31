# !/bin/bash

# Dropouts
ATTENTION_DROPOUT=0.25
ATTENTION_DROPOUT_AUDIO=0.0
ATTENTION_DROPOUT_VISION=0.0
RELU_DROPOUT=0.1
EMBED_DROPOUT=0.3
RESIDUAL_DROPOUT=0.1
OUT_DROPOUT=0.1

# Architectures
NUM_LAYERS=4
DIM_MODEL=40
NUM_HEADS=10
BATCH_SIZE=16
GRAD_CLIP=0.8
LEARNING_RATE=2e-3
EPOCHS=10

# python -m torch.distributed.launch \
#     --nproc_per_node=2 \
#     src/train.py \
#     --attn_dropout=${ATTENTION_DROPOUT} \
#     --attn_dropout_a=${ATTENTION_DROPOUT_AUDIO} \
#     --attn_dropout_v=${ATTENTION_DROPOUT_VISION} \
#     --relu_dropout=${RELU_DROPOUT} \
#     --embed_dropout=${EMBED_DROPOUT} \
#     --res_dropout=${RESIDUAL_DROPOUT} \
#     --out_dropout=${OUT_DROPOUT} \
#     --layers=${NUM_LAYERS} \
#     --d_model=${DIM_MODEL} \
#     --num_heads=${NUM_HEADS} \
#     --batch_size=${BATCH_SIZE} \
#     --clip=${GRAD_CLIP} \
#     --lr=${LEARNING_RATE} \
#     --num_epochs=${EPOCHS} \

python src/train.py \
    --attn_dropout=${ATTENTION_DROPOUT} \
    --attn_dropout_a=${ATTENTION_DROPOUT_AUDIO} \
    --attn_dropout_v=${ATTENTION_DROPOUT_VISION} \
    --relu_dropout=${RELU_DROPOUT} \
    --embed_dropout=${EMBED_DROPOUT} \
    --res_dropout=${RESIDUAL_DROPOUT} \
    --out_dropout=${OUT_DROPOUT} \
    --layers=${NUM_LAYERS} \
    --d_model=${DIM_MODEL} \
    --num_heads=${NUM_HEADS} \
    --batch_size=${BATCH_SIZE} \
    --clip=${GRAD_CLIP} \
    --lr=${LEARNING_RATE} \
    --num_epochs=${EPOCHS} \
