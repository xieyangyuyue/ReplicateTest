#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python train.py --model fcn32s \
--backbone vgg16 --dataset pascal_voc \
--lr 0.0001 --epochs 100 --batch-size 4
