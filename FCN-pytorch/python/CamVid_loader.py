# -*- coding: utf-8 -*-
# ============================================================================
# 【文件说明】CamVid_loader.py —— CamVid 数据集的 Dataset（数据集读取与预处理）
# 【数据集】CamVid（剑桥大学驾驶场景分割数据集）：原图 720×960，32 个语义类别
# 【功能】读取 csv 列表 → 读图与 .npy 索引标签 → 随机裁剪/翻转（数据增广）
#         → BGR 转换 + 归一化 + 减均值 → 转 torch 张量 → 生成 one-hot 标签
# 【说明】标签由 CamVid_utils.py 预先生成：彩色标签图 → 每像素类别索引的 .npy
# ============================================================================
from __future__ import print_function  # 兼容 Python2/3 的 print 函数写法

from matplotlib import pyplot as plt  # matplotlib 绘图库（show_batch 可视化用）
import pandas as pd   # pandas：读取 csv 形式的数据列表文件
import numpy as np    # numpy：数组读写与变换
from PIL import Image  # 【新环境适配】PIL（Pillow）图像读取：替代已彻底移除的 scipy.misc.imread（scipy>=1.3 弃用、>=1.10 删除；注意 imageio 是独立的第三方包，不是 scipy 的子模块）
import random         # 随机数：随机裁剪位置、随机翻转
import os             # os：路径拼接

import torch                          # PyTorch 主库
from torch.utils.data import Dataset, DataLoader  # 数据集抽象基类与批量加载器
from torchvision import utils         # torchvision 工具（此处用 make_grid 拼网格图）


root_dir   = "CamVid/"  # CamVid 数据集根目录
train_file = os.path.join(root_dir, "train.csv")  # 训练集列表：每行 "图片路径,标签路径"
val_file   = os.path.join(root_dir, "val.csv")    # 验证集列表：每行 "图片路径,标签路径"

num_class = 32  # CamVid 语义类别数
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
                   # 三个通道的均值（BGR 顺序，VGG/caffe 风格预处理常数），除以 255 先做归一化
h, w      = 720, 960  # CamVid 原始图像尺寸：高 720、宽 960
train_h   = int(h * 2 / 3)  # 480  # 训练时随机裁剪的高：原图的 2/3 → 480
train_w   = int(w * 2 / 3)  # 640  # 训练时随机裁剪的宽：原图的 2/3 → 640
val_h     = int(h/32) * 32  # 704  # 验证图的高：向下取 32 的倍数（720→704），保证能被网络 5 次下/上采样整除
val_w     = w               # 960  # 验证图的宽：960 本身是 32 的倍数，保持不变


class CamVidDataset(Dataset):  # CamVid 数据集类：继承 PyTorch 的 Dataset 基类

    # 构造参数：csv_file=列表文件路径；phase='train'或'val'；n_class=类别数；
    #           crop=是否随机裁剪增广；flip_rate=随机水平翻转的概率
    def __init__(self, csv_file, phase, n_class=num_class, crop=True, flip_rate=0.5):
        self.data      = pd.read_csv(csv_file)  # 读 csv 为 DataFrame（两列：img 图片路径、label 标签路径）
        self.means     = means                  # 保存 BGR 通道均值
        self.n_class   = n_class                # 保存类别数（用于 one-hot 通道数）

        self.flip_rate = flip_rate  # 保存翻转概率
        self.crop      = crop      # 保存是否裁剪
        if phase == 'train':       # 训练阶段：使用训练裁剪尺寸
            self.new_h = train_h   # 目标高 480
            self.new_w = train_w   # 目标宽 640
        elif phase == 'val':       # 验证阶段：关闭数据增广
            self.flip_rate = 0.    # 翻转概率置 0（验证不做随机增广）
            self.crop = False      # 关闭裁剪（用 resize 到 704×960 的整图）
            self.new_h = val_h     # 目标高 704
            self.new_w = val_w     # 目标宽 960


    def __len__(self):  # 返回数据集样本总数（DataLoader 划分批次时需要）
        return len(self.data)  # 即 csv 文件的行数

    def __getitem__(self, idx):  # 按索引取一个样本，完成全部预处理（DataLoader 每次调用）
        img_name   = self.data.iloc[idx, 0]  # 取第 idx 行第 0 列：图片路径（【新环境适配】原代码 .ix 在 pandas>=1.0 已删除，改为等价的 .iloc）
        img        = np.array(Image.open(img_name).convert('RGB'))  # 读入并统一转为 RGB 三通道（【新环境适配】替代已移除的 scipy.misc.imread(mode='RGB')；Pillow 是 torchvision 自带依赖）
        label_name = self.data.iloc[idx, 1]  # 取第 1 列：标签文件路径（.npy）（.ix → .iloc 同上）
        label      = np.load(label_name)   # 读入标签数组：每像素存类别索引 (H, W)

        if self.crop:  # 若启用随机裁剪增广
            h, w, _ = img.shape  # 获取图像实际高、宽
            top   = random.randint(0, h - self.new_h)   # 随机采样裁剪窗口左上角的行坐标
            left  = random.randint(0, w - self.new_w)   # 随机采样裁剪窗口左上角的列坐标
            img   = img[top:top + self.new_h, left:left + self.new_w]    # 按窗口裁剪图像
            label = label[top:top + self.new_h, left:left + self.new_w]  # 标签用同一窗口裁剪（保证像素级对齐）
        else:  # 【新环境适配·补上原仓库缺失的验证阶段裁剪】
            # 原因：CamVid 验证图 720×960，高 720 不是 32 的倍数；编码器 5 次 2× 下采样后 720→22，
            # 解码器 5 次精确 2× 上采样只能回到 704，与 720 高的标签算损失/算 IoU 会因形状不匹配报错。
            # 作者定义了 val_h=704/val_w=960 却没写使用它们的代码，此处按其本意补上中心裁剪
            # （裁掉上下各 8 行；中心裁剪不引入插值，不会破坏标签的类别边界）。
            h, w, _ = img.shape  # 获取图像实际高、宽
            top   = (h - self.new_h) // 2   # 垂直居中起点：720-704=16，上下各裁 8 行
            left  = (w - self.new_w) // 2   # 水平居中起点：960-960=0，宽度不裁
            img   = img[top:top + self.new_h, left:left + self.new_w]    # 中心裁剪图像到 704×960
            label = label[top:top + self.new_h, left:left + self.new_w]  # 标签用同一窗口裁剪（保证对齐）

        if random.random() < self.flip_rate:  # 以 flip_rate 的概率随机水平翻转（增广）
            img   = np.fliplr(img)    # 图像左右翻转
            label = np.fliplr(label)  # 标签同步左右翻转

        # reduce mean
        # 减均值标准化（VGG/caffe 风格的预处理）
        img = img[:, :, ::-1]  # switch to BGR  # 通道维反转：RGB → BGR（与预训练模型一致）
        img = np.transpose(img, (2, 0, 1)) / 255.  # HWC → CHW（PyTorch 要求通道在前），像素值除以 255 归一化到 [0,1]
        img[0] -= self.means[0]  # B 通道减去均值
        img[1] -= self.means[1]  # G 通道减去均值
        img[2] -= self.means[2]  # R 通道减去均值

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()    # numpy → float32 张量（copy 确保内存连续）
        label = torch.from_numpy(label.copy()).long()  # 标签 → long 整型张量（类别索引必须为整数）

        # create one-hot encoding
        h, w = label.size()  # 取标签的高、宽（可能已被裁剪）
        target = torch.zeros(self.n_class, h, w)  # 新建 (n_class, H, W) 的全零目标张量
        for c in range(self.n_class):  # 逐类别填充 one-hot
            target[c][label == c] = 1  # 标签值为 c 的像素，在第 c 个通道上置 1

        sample = {'X': img, 'Y': target, 'l': label}  # 打包样本：X=输入图像；Y=one-hot 标签（算损失用）；l=索引标签（评估用）

        return sample  # 返回样本字典


def show_batch(batch):  # 可视化一个批次的图像（调试用）
    img_batch = batch['X']  # 取出批次图像张量 (N, 3, H, W)
    img_batch[:,0,...].add_(means[0])  # 把预处理时减掉的均值加回 B 通道（还原可视化）
    img_batch[:,1,...].add_(means[1])  # 加回 G 通道均值
    img_batch[:,2,...].add_(means[2])  # 加回 R 通道均值
    batch_size = len(img_batch)  # 批大小（此处仅赋值、未再使用）

    grid = utils.make_grid(img_batch)  # 把批次中的多张图拼成一张网格大图
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))  # 转 numpy；[::-1] 把 BGR 反转回 RGB；CHW→HWC 后显示

    plt.title('Batch from dataloader')  # 设置图像标题


if __name__ == "__main__":  # 直接运行本文件时的自测代码
    train_data = CamVidDataset(csv_file=train_file, phase='train')  # 构建训练集对象

    # show a batch
    batch_size = 4  # 测试用批大小
    for i in range(batch_size):  # 逐个取出样本
        sample = train_data[i]  # 索引访问（触发 __getitem__）
        print(i, sample['X'].size(), sample['Y'].size())  # 打印输入与 one-hot 标签的形状

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)  # 构建加载器：打乱、4 进程并行

    for i, batch in enumerate(dataloader):  # 逐批迭代
        print(i, batch['X'].size(), batch['Y'].size())  # 打印该批数据的形状
    
        # observe 4th batch
        if i == 3:  # 观察第 4 个批次（索引 3）
            plt.figure()     # 新建画布
            show_batch(batch)  # 可视化该批次图像
            plt.axis('off')  # 关闭坐标轴
            plt.ioff()       # 关闭交互模式
            plt.show()       # 弹窗显示
            break            # 看完这一批即退出循环
