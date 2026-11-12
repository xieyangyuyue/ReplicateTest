#!/usr/bin/env bash  # 脚本解释器：用 bash 执行本脚本

# train  # 4 卡分布式训练 FCN32s 示例
export NGPUS=4  # 导出环境变量 NGPUS=4：告诉训练脚本使用 4 个 GPU 进程（分布式训练在脚本内部自动拉起）
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --model fcn32s \  # 使用 0~3 号共 4 块 GPU，模型选 FCN-32s
    --backbone vgg16 --dataset pascal_voc \                    # 骨干网络 VGG16，数据集 PASCAL VOC 2012
    --lr 0.01 --epochs 80 --batch_size 16                      # 学习率 0.01（多卡线性放大：单卡 0.0025×4），80 轮，全局批大小 16
