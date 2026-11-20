#!/usr/bin/env bash
# ============================================================================
# 多卡（DDP）训练示例：2 张 GPU
# 说明：
#   1) torchrun 会为每个进程设置 LOCAL_RANK / RANK / WORLD_SIZE，
#      train.py 会自动读取并初始化 NCCL + DDP；
#   2) --batch-size / --val-batch-size 是“每张 GPU”的批大小，--lr 会按 GPU 数线性放大；
#   3) --amp 在 T4 上可显著降低显存占用并提速；
#   4) 验证阶段无梯度，--val-batch-size 可比训练批大小大，提升双卡利用率。
# 旧版 PyTorch 若没有 torchrun，可改用：
#   python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 \
#       train.py --model fcn32s --backbone vgg16 --dataset pascal_voc \
#       --lr 0.0001 --epochs 100 --batch-size 4 --val-batch-size 8 --workers 2 --amp
# ============================================================================
set -e  # 任一命令失败立即退出
cd "$(dirname "$0")"  # 切到 scripts/ 目录，保证 train.py 与相对路径正确

NGPUS="${NGPUS:-2}"  # 默认使用 2 张 GPU，可执行 NGPUS=2 bash xxx.sh 覆盖
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
torchrun --nproc_per_node="${NGPUS}" --master_port=29500 train.py \
    --model fcn32s --backbone vgg16 --dataset pascal_voc \
    --lr 1e-3 --epochs 80 \
    --batch-size 50 --val-batch-size 8 --workers 2 --amp \
    --best-metric miou
