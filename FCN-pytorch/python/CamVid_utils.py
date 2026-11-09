# -*- coding: utf-8 -*-
# ============================================================================
# 【文件说明】CamVid_utils.py —— CamVid 数据集的预处理脚本
# 【功能】1) divide_train_val：把全部图片随机划分为训练/验证集，生成 train.csv、val.csv；
#         2) parse_label：读取"颜色-类别"对照表，把彩色标签图逐像素转成类别索引，
#            保存为 .npy 数组（训练时 CamVid_loader 直接读取，大幅提速）。
# 【用法】先下载 CamVid 数据放到 CamVid/ 目录下，再直接运行本文件即可。
# ============================================================================
from __future__ import print_function  # 兼容 Python2/3 的 print 函数写法

from matplotlib import pyplot as plt  # matplotlib 绘图库（imshow 调试用）
import matplotlib.image as mpimg      # matplotlib 的图像读取模块
import numpy as np    # numpy：数组操作与 .npy 保存
from PIL import Image  # 【新环境适配】PIL（Pillow）图像读取：替代已彻底移除的 scipy.misc.imread（scipy>=1.10 删除；imageio 是独立包，不是 scipy 子模块）
import random         # 随机数：打乱划分、随机抽样
import os             # os：目录遍历、路径拼接、目录创建


#############################
    # global variables #
#############################
# 全局变量：CamVid 数据集的目录与文件路径
root_dir          = "CamVid/"                                    # 数据集根目录
data_dir          = os.path.join(root_dir, "701_StillsRaw_full")    # train data  # 原始图像目录（701 张视频帧）
label_dir         = os.path.join(root_dir, "LabeledApproved_full")  # train label  # 人工审核过的彩色标签图目录
label_colors_file = os.path.join(root_dir, "label_colors.txt")      # color to label  # 颜色-类别对照表（每行 "R G B 类别名"）
val_label_file    = os.path.join(root_dir, "val.csv")               # validation file  # 待生成的验证集列表
train_label_file  = os.path.join(root_dir, "train.csv")             # train file  # 待生成的训练集列表

# create dir for label index
label_idx_dir = os.path.join(root_dir, "Labeled_idx")  # 类别索引标签（.npy）的输出目录
if not os.path.exists(label_idx_dir):  # 若目录不存在
    os.makedirs(label_idx_dir)         # 创建目录

label2color = {}  # 映射：类别名 → RGB 颜色
color2label = {}  # 映射：RGB 颜色 → 类别名
label2index = {}  # 映射：类别名 → 类别索引（0~31，即对照表中的行号）
index2label = {}  # 映射：类别索引 → 类别名


# 划分训练/验证集：val_rate=验证集比例；shuffle=是否随机打乱；random_seed=随机种子（固定则可复现）
def divide_train_val(val_rate=0.1, shuffle=True, random_seed=None):
    data_list   = os.listdir(data_dir)  # 列出原图目录下的所有文件名
    data_len    = len(data_list)        # 文件总数
    val_len     = int(data_len * val_rate)  # 验证集数量（10%）

    if random_seed:  # 若指定了随机种子
        random.seed(random_seed)  # 固定随机序列，使每次划分结果一致（可复现）

    if shuffle:  # 若需要打乱
        data_idx = random.sample(range(data_len), data_len)  # 生成 0..N-1 的一个随机全排列
    else:        # 否则保持原顺序
        data_idx = list(range(data_len))  # 顺序索引列表

    val_idx     = [data_list[i] for i in data_idx[:val_len]]  # 打乱后的前 val_len 个文件名 → 验证集
    train_idx   = [data_list[i] for i in data_idx[val_len:]]  # 其余文件名 → 训练集

    # create val.csv
    v = open(val_label_file, "w")  # 新建（覆写）val.csv
    v.write("img,label\n")  # 写入表头：两列名
    for idx, name in enumerate(val_idx):  # 遍历验证集文件名
        if 'png' not in name:  # 只处理 png 图像文件
            continue  # 其余文件跳过
        img_name = os.path.join(data_dir, name)  # 原图完整路径（701_StillsRaw_full/ 下）
        lab_name = os.path.join(label_idx_dir, name)  # 对应索引标签路径（Labeled_idx/ 下）
        lab_name = lab_name.split(".")[0] + "_L.png.npy"  # 改写文件名：xxx.png → xxx_L.png.npy（与彩色标签 _L 后缀对应）
        v.write("{},{}\n".format(img_name, lab_name))  # 写入一行 "原图路径,标签路径"

    # create train.csv
    t = open(train_label_file, "w")  # 新建（覆写）train.csv
    t.write("img,label\n")  # 写入表头
    for idx, name in enumerate(train_idx):  # 遍历训练集文件名
        if 'png' not in name:  # 只处理 png
            continue  # 跳过其他文件
        img_name = os.path.join(data_dir, name)  # 原图完整路径
        lab_name = os.path.join(label_idx_dir, name)  # 对应标签路径
        lab_name = lab_name.split(".")[0] + "_L.png.npy"  # 改写为 .npy 命名
        t.write("{},{}\n".format(img_name, lab_name))  # 写入一行


def parse_label():  # 解析彩色标签：建颜色-类别映射表，并把每张标签图转成类别索引矩阵
    # change label to class index
    f = open(label_colors_file, "r").read().split("\n")[:-1]  # ignore the last empty line
      # 读取对照表全文并按换行拆分；去掉末尾多出的空行
    for idx, line in enumerate(f):  # 逐行解析（行号 idx 即类别索引）
        label = line.split()[-1]  # 行内最后一个词：类别名（如 Sky、Car）
        color = tuple([int(x) for x in line.split()[:-1]])  # 前三个数字 → (R, G, B) 元组
        print(label, color)  # 打印解析出的 类别名 与 颜色
        label2color[label] = color  # 填充映射：类别名 → 颜色
        color2label[color] = label  # 填充映射：颜色 → 类别名
        label2index[label] = idx    # 填充映射：类别名 → 索引（行号）
        index2label[idx]   = label  # 填充映射：索引 → 类别名
        # rgb = np.zeros((255, 255, 3), dtype=np.uint8)  # （被注释的调试代码：生成纯色块图检查颜色）
        # rgb[..., 0] = color[0]                          # 填充 R 通道
        # rgb[..., 1] = color[1]                          # 填充 G 通道
        # rgb[..., 2] = color[2]                          # 填充 B 通道
        # imshow(rgb, title=label)                        # 显示该颜色色块

    # 【新环境适配·提速】预建颜色查找表（LUT）：把上面两级字典 color2label/label2index
    # 压平为 256×256×256 的数组（以 R、G、B 三通道值作下标、类别索引作值），
    # 供下方用数组花式索引整图查表，替代逐像素的双重 for 循环（结果完全等价）
    lut = np.zeros((256, 256, 256), dtype=np.uint8)  # 查找表默认填 0：查不到的颜色保持索引 0，与原代码 except 分支行为一致
    for color, label in color2label.items():  # 遍历 颜色 → 类别名 映射
        lut[color[0], color[1], color[2]] = label2index[label]  # 该颜色对应的类别索引写入 LUT 对应位置
    
    for idx, name in enumerate(os.listdir(label_dir)):  # 遍历每张彩色标签图
        filename = os.path.join(label_idx_dir, name)  # 对应索引标签的保存路径
        if os.path.exists(filename + '.npy'):  # 若已生成过（支持断点续跑）
            print("Skip %s" % (name))  # 打印跳过
            continue  # 处理下一张
        print("Parse %s" % (name))  # 打印正在解析的文件名
        img = os.path.join(label_dir, name)  # 彩色标签图完整路径
        img = np.array(Image.open(img).convert('RGB'))  # 读入并转为 RGB（【新环境适配】替代已移除的 scipy.misc.imread；此处为彩色标签图）
        height, weight, _ = img.shape  # 取高与宽（变量名 weight 实际是 width 宽）

        # 【新环境适配·提速】整图向量化查表：用上方预建的颜色查找表 lut，通过数组花式索引
        # 一次性完成全部像素的 颜色→类别索引 转换（毫秒级）。原实现为下方注释中的双重 for 循环
        # 逐像素查字典（720×960≈69 万像素/张，纯 Python 每张需数十秒、全数据集 701 张要数小时），
        # 两者结果完全一致：查不到的颜色保持索引 0（与原 except 分支的兜底行为相同）。
        idx_mat = lut[img[..., 0], img[..., 1], img[..., 2]]  # (H,W,3) 彩色标签 → (H,W) 类别索引矩阵
        # ---- 原始实现（保留备查，与上行完全等价但极慢）----
        # idx_mat = np.zeros((height, weight))  # 新建同尺寸的索引矩阵，默认填充 0
        # for h in range(height):  # 逐行遍历像素
        #     for w in range(weight):  # 逐列遍历像素（双重循环查表，速度较慢）
        #         color = tuple(img[h, w])  # 该像素的 (R, G, B) 颜色
        #         try:  # 尝试把颜色查表转换为类别索引
        #             label = color2label[color]  # 查表：颜色 → 类别名
        #             index = label2index[label]  # 查表：类别名 → 索引
        #             idx_mat[h, w] = index  # 写入索引矩阵
        #         except:  # 查表失败（颜色不在对照表中）时进入此分支
        #             print("error: img:%s, h:%d, w:%d" % (name, h, w))  # 颜色查不到时打印出错位置（该像素保持 0）
        idx_mat = idx_mat.astype(np.uint8)  # 转为 8 位无符号整数（32 类足够，节省空间）
        np.save(filename, idx_mat)  # 保存为 .npy 文件
        print("Finish %s" % (name))  # 打印完成

    # test some pixels' label    
    img = os.path.join(label_dir, os.listdir(label_dir)[0])  # 取第一张标签图做抽检验证
    img = np.array(Image.open(img).convert('RGB'))  # 读入该标签图（【新环境适配】替代已移除的 scipy.misc.imread）
    test_cases = [(555, 405), (0, 0), (380, 645), (577, 943)]  # 抽检的像素坐标 (h, w)
    test_ans   = ['Car', 'Building', 'Truck_Bus', 'Car']  # 期望查到的类别名
    for idx, t in enumerate(test_cases):  # 逐个校验
        color = img[t]  # 取该坐标像素的颜色
        assert color2label[tuple(color)] == test_ans[idx]  # 断言 颜色→类别 的查询结果与期望一致


'''debug function'''
def imshow(img, title=None):  # 调试用图像显示函数：参数可以是文件路径或数组
    try:  # 先假设传入的是文件路径
        img = mpimg.imread(img)  # 尝试按路径读取图像
        imgplot = plt.imshow(img)  # 并显示
    except:  # 读取失败说明传入的是数组
        plt.imshow(img, interpolation='nearest')  # 读取失败则当作数组直接显示（最近邻插值）

    if title is not None:  # 若提供了标题
        plt.title(title)  # 设置图像标题
    
    plt.show()  # 弹窗显示


if __name__ == '__main__':  # 直接运行本文件时执行完整预处理流程
    divide_train_val(random_seed=1)  # 第一步：划分训练/验证集（种子 1，结果可复现）
    parse_label()  # 第二步：解析全部彩色标签为索引 .npy 文件
