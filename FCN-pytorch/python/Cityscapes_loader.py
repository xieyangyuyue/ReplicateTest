# -*- coding: utf-8 -*-
# ============================================================================
# 【文件说明】Cityscapes_loader.py —— Cityscapes 数据集的 Dataset（读取与预处理）
# 【数据集】Cityscapes（城市街景分割数据集）：原图 1024×2048，处理后 20 个类别
# 【功能】与 CamVid_loader.py 相同：读 csv → 读图与 .npy 标签 → 裁剪/翻转增广
#         → BGR + 归一化 + 减均值 → 转张量 → one-hot 标签
# 【差异】相比 CamVid 版：类别数 20；训练裁剪为原图 1/2（512×1024）；
#         验证用整图（1024×2048，本身是 32 的倍数）
# ============================================================================
from __future__ import print_function  # 兼容 Python2/3 的 print 函数写法

from matplotlib import pyplot as plt  # matplotlib 绘图库（show_batch 可视化用）
import pandas as pd   # pandas：读取 csv 数据列表
import numpy as np    # numpy：数组读写与变换
from PIL import Image  # 【新环境适配】PIL（Pillow）图像读取：替代已彻底移除的 scipy.misc.imread（scipy>=1.10 删除；imageio 是独立包，不是 scipy 子模块）
import random         # 随机数：随机裁剪、随机翻转
import os             # os：路径拼接

import torch                          # PyTorch 主库
from torch.utils.data import Dataset, DataLoader  # 数据集基类与批量加载器
from torchvision import utils         # torchvision 工具（make_grid 拼图）


root_dir   = "CityScapes/"  # Cityscapes 数据集根目录
train_file = os.path.join(root_dir, "train.csv")  # 训练集列表：每行 "图片路径,标签路径"
val_file   = os.path.join(root_dir, "val.csv")    # 验证集列表：每行 "图片路径,标签路径"

num_class = 20  # 类别数：Cityscapes 官方 19 个训练类 + 1 个兜底类（见 Cityscapes_utils.py）
means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
                   # 三个通道的均值（BGR 顺序，VGG/caffe 风格预处理常数），除以 255 先归一化
h, w      = 1024, 2048  # Cityscapes 原始图像尺寸：高 1024、宽 2048
train_h   = int(h/2)  # 512   # 训练时随机裁剪的高：原图的 1/2 → 512
train_w   = int(w/2)  # 1024  # 训练时随机裁剪的宽：原图的 1/2 → 1024
val_h     = h  # 1024  # 验证用整图：高 1024（已是 32 的倍数）
val_w     = w  # 2048  # 验证用整图：宽 2048（已是 32 的倍数）


class CityScapesDataset(Dataset):  # Cityscapes 数据集类：继承 PyTorch 的 Dataset 基类

    # 构造参数：csv_file=列表文件；phase='train'或'val'；n_class=类别数；
    #           crop=是否随机裁剪（默认 False，训练阶段会在下方强制打开）；flip_rate=翻转概率（默认 0）
    def __init__(self, csv_file, phase, n_class=num_class, crop=False, flip_rate=0.):
        self.data      = pd.read_csv(csv_file)  # 读 csv 为 DataFrame（img、label 两列）
        self.means     = means                  # 保存 BGR 通道均值
        self.n_class   = n_class                # 保存类别数（one-hot 通道数）

        self.flip_rate = flip_rate  # 保存翻转概率
        self.crop      = crop      # 保存是否裁剪
        if phase == 'train':       # 训练阶段：强制开启增广
            self.crop = True       # 打开随机裁剪
            self.flip_rate = 0.5   # 翻转概率设为 0.5
            self.new_h = train_h   # 裁剪高 512
            self.new_w = train_w   # 裁剪宽 1024
        else:                      # 【新环境适配补充】非训练阶段：补齐验证尺寸初始化（原代码漏写，
            self.new_h = val_h     # 引用 self.new_h 时会 AttributeError）。Cityscapes 验证图 1024×2048
            self.new_w = val_w     # 本身就是 32 的倍数，下方中心裁剪 top=left=0 实际不裁任何像素

    def __len__(self):  # 返回数据集样本总数
        return len(self.data)  # csv 的行数

    def __getitem__(self, idx):  # 按索引取一个样本并完成预处理
        img_name   = self.data.iloc[idx, 0]  # 第 idx 行第 0 列：图片路径（【新环境适配】.ix 在 pandas>=1.0 已删除，改为等价的 .iloc）
        img        = np.array(Image.open(img_name).convert('RGB'))  # 读入并统一转为 RGB（【新环境适配】替代已移除的 scipy.misc.imread(mode='RGB')）
        label_name = self.data.iloc[idx, 1]  # 第 1 列：标签文件路径（.npy）（.ix → .iloc 同上）
        label      = np.load(label_name)   # 读入标签数组：每像素类别索引 (H, W)

        if self.crop:  # 若启用随机裁剪增广（训练阶段）
            h, w, _ = img.shape  # 获取实际高、宽
            top   = random.randint(0, h - self.new_h)   # 随机采样裁剪窗口左上角行坐标
            left  = random.randint(0, w - self.new_w)   # 随机采样裁剪窗口左上角列坐标
            img   = img[top:top + self.new_h, left:left + self.new_w]    # 裁剪图像
            label = label[top:top + self.new_h, left:left + self.new_w]  # 标签按同一窗口裁剪（保持对齐）
        else:  # 【新环境适配补充】验证/测试阶段的中心裁剪（与 CamVid_loader 同一修复，防御性代码）
            h, w, _ = img.shape  # 获取实际高、宽
            top   = (h - self.new_h) // 2   # 垂直居中裁剪起点（1024 高时为 0，不裁）
            left  = (w - self.new_w) // 2   # 水平居中裁剪起点（2048 宽时为 0，不裁）
            img   = img[top:top + self.new_h, left:left + self.new_w]    # 中心裁剪（非 32 倍数时才生效）
            label = label[top:top + self.new_h, left:left + self.new_w]  # 标签同一窗口裁剪（保持对齐）

        if random.random() < self.flip_rate:  # 以 flip_rate 概率随机水平翻转
            img   = np.fliplr(img)    # 图像左右翻转
            label = np.fliplr(label)  # 标签同步翻转

        # reduce mean
        # 减均值标准化（VGG/caffe 风格的预处理）
        img = img[:, :, ::-1]  # switch to BGR  # 通道反转：RGB → BGR
        img = np.transpose(img, (2, 0, 1)) / 255.  # HWC → CHW，除以 255 归一化到 [0,1]
        img[0] -= self.means[0]  # B 通道减均值
        img[1] -= self.means[1]  # G 通道减均值
        img[2] -= self.means[2]  # R 通道减均值

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()    # numpy → float32 张量（copy 保证内存连续）
        label = torch.from_numpy(label.copy()).long()  # 标签 → long 整型张量

        # create one-hot encoding
        h, w = label.size()  # 取标签高、宽（可能已被裁剪）
        target = torch.zeros(self.n_class, h, w)  # 新建 (n_class, H, W) 全零目标张量
        for c in range(self.n_class):  # 逐类别填充
            target[c][label == c] = 1  # 标签为 c 的像素在第 c 通道置 1（one-hot）

        sample = {'X': img, 'Y': target, 'l': label}  # 打包：X=输入图像；Y=one-hot 标签（算损失）；l=索引标签（评估）

        return sample  # 返回样本字典


def show_batch(batch):  # 可视化一个批次的图像（调试用）
    img_batch = batch['X']  # 取批次图像张量
    img_batch[:,0,...].add_(means[0])  # 加回 B 通道均值（还原可视化）
    img_batch[:,1,...].add_(means[1])  # 加回 G 通道均值
    img_batch[:,2,...].add_(means[2])  # 加回 R 通道均值
    batch_size = len(img_batch)  # 批大小（此处仅赋值、未再使用）

    grid = utils.make_grid(img_batch)  # 批次内多张图拼成网格大图
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))  # 转 numpy；BGR→RGB（[::-1]）；CHW→HWC 后显示

    plt.title('Batch from dataloader')  # 设置标题


if __name__ == "__main__":  # 直接运行本文件时的自测代码
    train_data = CityScapesDataset(csv_file=train_file, phase='train')  # 构建训练集对象

    # show a batch
    batch_size = 4  # 测试批大小
    for i in range(batch_size):  # 逐个取样本
        sample = train_data[i]  # 索引访问（触发 __getitem__）
        print(i, sample['X'].size(), sample['Y'].size())  # 打印输入与标签形状

    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)  # 加载器：不打乱（与 CamVid 版的差异）、4 进程

    for i, batch in enumerate(dataloader):  # 逐批迭代
        print(i, batch['X'].size(), batch['Y'].size())  # 打印批次形状
    
        # observe 4th batch
        if i == 3:  # 观察第 4 个批次（索引 3）
            plt.figure()       # 新建画布
            show_batch(batch)  # 可视化该批次
            plt.axis('off')    # 关闭坐标轴
            plt.ioff()         # 关闭交互模式
            plt.show()         # 显示图像
            break              # 看完即退出
