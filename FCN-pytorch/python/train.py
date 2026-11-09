# -*- coding: utf-8 -*-
# ============================================================================
# 【文件说明】train.py —— FCN 语义分割网络的训练与验证脚本
# 【运行方式】python train.py CamVid        （使用 CamVid 数据集训练）
#             python train.py Cityscapes    （使用 Cityscapes 数据集训练）
# 【整体流程】命令行选数据集 → 读 csv 列表构建 DataLoader → 搭建 VGG+FCNs 模型
#             → 每轮训练后在验证集上计算 IoU / 像素精度并保存模型与指标
# ============================================================================
from __future__ import print_function  # 兼容 Python2/3 的 print 函数写法

import torch                          # PyTorch 主库
import torch.nn as nn                 # 神经网络模块（损失函数、DataParallel 等）
import torch.optim as optim           # 优化器模块
from torch.optim import lr_scheduler  # 学习率调度器（按步长衰减）
from torch.autograd import Variable   # 旧版张量包装类（PyTorch 0.4 起 Tensor 已自带自动求导）
from torch.utils.data import DataLoader  # 数据加载器：批量读取 + 多进程加速

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs  # 从 fcn.py 导入模型结构（本仓库实现）
from Cityscapes_loader import CityScapesDataset      # Cityscapes 数据集类
from CamVid_loader import CamVidDataset              # CamVid 数据集类

from matplotlib import pyplot as plt  # matplotlib 绘图库（本文件仅导入、未直接使用）
import numpy as np                    # 数值计算库（保存指标数组）
import time                           # 时间库：统计训练/加载耗时
import sys                            # 系统库：读取命令行参数
import os                             # 操作系统库：路径拼接、目录创建


n_class    = 32   # 分割类别数（默认 20：Cityscapes；下方会按命令行数据集参数自动覆盖：CamVid → 32）

batch_size = 6    # 每批训练图片数量
epochs     = 500  # 总训练轮数
lr         = 1e-4 # 初始学习率
momentum   = 0    # 优化器动量系数（此处设 0，即不带动量）
w_decay    = 1e-5 # 权重衰减系数（L2 正则，防过拟合）
step_size  = 50   # 学习率衰减间隔：每 50 个 epoch 衰减一次
gamma      = 0.5  # 学习率衰减倍率：每次衰减为原来的 0.5 倍
# 把全部超参拼成一个字符串，用作模型文件名和评分目录名，便于区分不同实验配置
configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)  # 打印当前实验配置

# 根据命令行第 1 个参数选择数据集根目录
if len(sys.argv) < 2:  # 【新环境适配】未传数据集参数时打印用法并退出（原代码会直接 IndexError）
    sys.exit("用法: python train.py [CamVid | Cityscapes]")  # 提示需要指定数据集名称
if sys.argv[1] == 'CamVid':   # 参数为 CamVid
    root_dir   = "CamVid/"   # 使用 CamVid 数据目录
    n_class    = 32          # 【新环境适配】按数据集自动设置类别数：CamVid 为 32 类。
                             # 原代码固定 n_class=20，跑 CamVid 时模型输出 20 通道与 32 通道
                             # 的 one-hot 标签形状不匹配，BCEWithLogitsLoss 会直接报错
else:                         # 【注意】原代码此处漏写冒号（写成裸的 else）导致 SyntaxError，现已修复为 else:
    root_dir   = "CityScapes/"  # 其他参数默认使用 Cityscapes 数据目录
    n_class    = 20             # 【新环境适配】Cityscapes 处理后为 20 类（19 个有效类 + 兜底类）
train_file = os.path.join(root_dir, "train.csv")  # 训练集列表文件（每行：图片路径,标签路径）
val_file   = os.path.join(root_dir, "val.csv")    # 验证集列表文件

# create dir for model
model_dir = "models"               # 模型保存目录
if not os.path.exists(model_dir):  # 若目录不存在
    os.makedirs(model_dir)         # 递归创建（含多级父目录）
model_path = os.path.join(model_dir, configs)  # 模型完整保存路径（以配置串命名，无后缀）

use_gpu = torch.cuda.is_available()         # 检测 GPU 是否可用
num_gpu = list(range(torch.cuda.device_count()))  # 所有可用 GPU 的编号列表，如 [0, 1]

# ---------- 根据数据集构建训练集 Dataset 与 DataLoader ----------
if sys.argv[1] == 'CamVid':  # CamVid：训练阶段在 Dataset 内部开启随机裁剪 + 随机翻转增广
    train_data = CamVidDataset(csv_file=train_file, phase='train')  # 构建 CamVid 训练集对象
else:                        # Cityscapes：train 阶段同样在类内部强制开启裁剪与翻转
    train_data = CityscapesDataset(csv_file=train_file, phase='train')  # 构建 Cityscapes 训练集对象
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)  # 训练加载器：打乱顺序、8 进程并行读取

# ---------- 构建验证集 Dataset 与 DataLoader ----------
if sys.argv[1] == 'CamVid':  # CamVid 验证集：不翻转（flip_rate=0）
    val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)  # 构建 CamVid 验证集对象
else:                        # Cityscapes 验证集
    val_data = CityscapesDataset(csv_file=val_file, phase='val', flip_rate=0)  # 构建 Cityscapes 验证集对象
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)  # 验证加载器：批大小 1、不打乱、8 进程读取

vgg_model = VGGNet(requires_grad=True, remove_fc=True)  # VGG 编码器：参数可训练（端到端微调）、删除冗余全连接层
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)  # FCNs 分割网络：以该 VGG 为编码器

if use_gpu:  # 若 GPU 可用：模型搬上显卡并包裹为多卡并行
    ts = time.time()                       # 记录搬运开始时间
    vgg_model = vgg_model.cuda()           # 编码器移入 GPU（vgg_model 已嵌入 fcn_model 内）
    fcn_model = fcn_model.cuda()           # 分割网络整体移入 GPU
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)  # DataParallel 包裹：多卡数据并行
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))  # 打印搬运耗时

criterion = nn.BCEWithLogitsLoss()  # 带 logits 的二值交叉熵损失：内部自带 sigmoid，无需手动激活；
                                    # 将每个像素的每一类都视为独立二分类（多标签分割方式）
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)  # RMSprop 优化器
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs
                                    # 学习率调度器：每 step_size 个 epoch 把学习率乘以 gamma
                                    # （原注释写 every 30 epochs，实际间隔由 step_size=50 决定）

# create dir for score
score_dir = os.path.join("scores", configs)  # 验证指标保存目录（以配置串命名）
if not os.path.exists(score_dir):            # 若目录不存在
    os.makedirs(score_dir)                   # 创建目录
IU_scores    = np.zeros((epochs, n_class))   # 记录表：每 epoch 各类别的 IoU 得分
pixel_scores = np.zeros(epochs)              # 记录表：每 epoch 的像素精度


def train():  # 训练主函数
    for epoch in range(epochs):  # 外层循环：逐个 epoch 训练
        scheduler.step()  # 更新学习率（StepLR 按轮衰减）。保持原仓库"每轮开头调用"的顺序以复现原结果；
                         # 新版 PyTorch 会警告"应在 optimizer.step() 之后调用"，仅警告不影响训练，可忽略

        ts = time.time()  # 记录本 epoch 开始时间
        for iter, batch in enumerate(train_loader):  # 内层循环：逐批读取训练数据
            optimizer.zero_grad()  # 将所有参数梯度清零（PyTorch 的梯度默认会累加，必须手动清零）

            if use_gpu:  # GPU 可用：把数据也搬到显卡
                inputs = Variable(batch['X'].cuda())  # 输入图像张量（已减均值、CHW 格式）
                labels = Variable(batch['Y'].cuda())  # one-hot 编码的分割标签
            else:        # CPU 模式：直接使用
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])  # 直接包装输入与标签

            outputs = fcn_model(inputs)        # 前向传播：得到 (N, n_class, H, W) 的得分图
            loss = criterion(outputs, labels)  # 计算损失（BCEWithLogitsLoss）
            loss.backward()                    # 反向传播：计算各参数梯度
            optimizer.step()                   # 优化器：按梯度更新参数

            if iter % 10 == 0:  # 每 10 个批次打印一次损失
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))  # 【新环境适配】原代码 loss.data[0]：新版 PyTorch 中标量张量不允许 [0] 索引（IndexError），改用 .item() 取 Python 数值

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))  # 打印本 epoch 总耗时
        torch.save(fcn_model, model_path)  # 保存整个模型（结构+权重）到 model_path
                                          # 【注意】PyTorch>=2.6 加载此类"整个模型"文件需显式
                                          # torch.load(model_path, weights_only=False)，
                                          # 且需在与保存时相同的代码环境下加载（依赖 pickle）

        val(epoch)  # 在验证集上评估本 epoch 训练效果


def val(epoch):  # 验证函数：在验证集上计算各类 IoU 与像素精度
    fcn_model.eval()  # 切换到评估模式（冻结 BN 统计、关闭 Dropout 等训练行为）
    total_ious = []   # 收集每张验证图的各类 IoU 列表
    pixel_accs = []   # 收集每张验证图的像素精度
    for iter, batch in enumerate(val_loader):  # 逐张遍历验证集（批大小为 1）
        if use_gpu:  # GPU：输入搬上显卡
            inputs = Variable(batch['X'].cuda())  # 图像移入 GPU
        else:        # CPU：直接使用
            inputs = Variable(batch['X'])  # 直接使用 CPU 张量

        output = fcn_model(inputs)          # 前向传播得到得分图
        output = output.data.cpu().numpy()  # 脱离计算图、转回 CPU 的 numpy 数组

        N, _, h, w = output.shape  # 解包批大小 N 与空间尺寸 h×w
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)  # 通道维换到最后并展平，逐像素取得分最大的类别作为预测，再还原空间形状

        target = batch['l'].cpu().numpy().reshape(N, h, w)  # 取类别索引标签（'l' = label，非 one-hot 版本）
        for p, t in zip(pred, target):  # 逐张图计算指标
            total_ious.append(iou(p, t))        # 各类 IoU
            pixel_accs.append(pixel_acc(p, t))  # 像素精度

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len  # 列表转为数组并转置成 (类别数 × 图片数)
    ious = np.nanmean(total_ious, axis=1)  # 沿图片维求均值（自动忽略 NaN，即忽略验证集中未出现的类）
    pixel_accs = np.array(pixel_accs).mean()  # 所有验证图的平均像素精度
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))  # 打印本 epoch 指标
    IU_scores[epoch] = ious  # 记录本 epoch 各类 IoU
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)      # 把 IoU 记录数组存盘（.npy）
    pixel_scores[epoch] = pixel_accs  # 记录本 epoch 像素精度
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)  # 把像素精度记录存盘


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
# 借用并修改自 Kaixhin/FCN-semantic-segmentation：计算各类别的交并比（IoU）
def iou(pred, target):  # pred：预测类别图 (H,W)；target：真值类别图 (H,W)
    ious = []  # 保存每个类别的 IoU
    for cls in range(n_class):  # 遍历全部类别
        pred_inds = pred == cls      # 布尔掩码：预测为该类的像素
        target_inds = target == cls  # 布尔掩码：真值为该类的像素
        intersection = pred_inds[target_inds].sum()  # 交集：两者都为该类的像素数
        union = pred_inds.sum() + target_inds.sum() - intersection  # 并集：两者像素数之和减去交集
        if union == 0:  # 并集为 0：该类在预测和真值中均未出现
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
                                     # 记为 NaN，评估时会被 nanmean 忽略
        else:
            ious.append(float(intersection) / max(union, 1))  # IoU = 交集 / 并集（max 防除零）
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious  # 返回长度为 n_class 的各类 IoU 列表


def pixel_acc(pred, target):  # 像素精度：预测正确的像素数占总像素数的比例
    correct = (pred == target).sum()  # 预测与真值类别相同的像素数
    total   = (target == target).sum()  # 总像素数（利用 target==target 恒真来计数）
    return correct / total  # 返回精度比值


if __name__ == "__main__":  # 作为主程序运行：先验证初始模型、再开始训练
    val(0)  # show the accuracy before training  # 训练前先验证一次：展示未训练模型的"底线"精度
    train()  # 正式开始训练
