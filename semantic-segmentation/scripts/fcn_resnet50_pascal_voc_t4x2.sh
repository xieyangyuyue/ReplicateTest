#!/usr/bin/env bash
# ============================================================================
# Kaggle T4 x2 推荐配置：FCN(v2) + ResNet50 + PASCAL VOC，DDP + AMP
#
# 为什么是 --model fcn？
#   core/models/fcn.py 中的 fcn32s/fcn16s/fcn8s 是按 VGG16 的 pool3/4/5 层级
#   硬编码的；ResNet 骨干的 stride-8 FCN 实现在 core/models/fcnv2.py，注册名为 fcn。
#   为了兼容旧命令，--model fcn8s --backbone resnet50 也会自动路由到 FCNv2，
#   但推荐直接写 --model fcn，语义更清晰。
#
# 说明：
#   --batch-size 是每张 GPU 的批大小；--lr 会按 GPU 数线性放大。
#   若显存不足，优先减小 --batch-size（如 2），或增大 --workers 前先确认 CPU 核数。
# ============================================================================
set -e  # 任一命令失败立即退出
cd "$(dirname "$0")"  # 切到 scripts/ 目录

NGPUS="${NGPUS:-2}"  # 默认 2 卡
SAVE_DIR="${SAVE_DIR:-../runs/checkpoints}"  # 权重保存目录（相对 scripts/）
LOG_DIR="${LOG_DIR:-../runs/logs}"  # 日志目录（相对 scripts/）

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
torchrun --nproc_per_node="${NGPUS}" --master_port=29500 train.py \
    --model fcn --backbone resnet50 --dataset pascal_voc \
    --lr 0.0001 --epochs 80 --batch-size 4 --workers 2 --amp \
    --best-metric miou \
    --save-dir "${SAVE_DIR}" --log-dir "${LOG_DIR}"
