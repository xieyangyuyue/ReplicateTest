#!/usr/bin/env bash  # 脚本解释器：用 bash 执行本脚本

# train  # 单卡训练 FCN32s 示例
CUDA_VISIBLE_DEVICES=0 python train.py --model fcn32s \  # 只使用 0 号 GPU，运行训练脚本，模型选 FCN-32s
    --backbone vgg16 --dataset pascal_voc \              # 骨干网络 VGG16，数据集 PASCAL VOC 2012
    --lr 0.0001 --epochs 80                              # 学习率 1e-4，共训练 80 轮
CUDA_VISIBLE_DEVICES=0 python train.py --model fcn32s \
--backbone vgg16 --dataset pascal_voc \
--lr 0.0001 --epochs 80
