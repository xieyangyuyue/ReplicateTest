# -*- coding: utf-8 -*-
# ============================================================================
# 【文件说明】Cityscapes_utils.py —— Cityscapes 数据集的预处理脚本
# 【功能】把 Cityscapes 的彩色标签图（*_gtFine_color.png）逐像素解析为
#         "trainId 类别索引"的 .npy 数组，并生成 train/val/test 三个 csv 列表。
# 【说明】Cityscapes 官方定义 34 个类别（含 void 类），其中参与训练与评估的只有
#         19 类（trainId 1~19），trainId=255 表示 void（不参与训练）。
#         本代码额外规定：索引 0 = void 兜底类、索引 19 = 查表失败兜底，
#         因此训练时的类别数 n_class = 20（见 Cityscapes_loader.py）。
# 【用法】先下载 Cityscapes（gtFine + leftImg8bit）放到 CityScapes/ 目录，运行本文件。
# ============================================================================
from __future__ import print_function  # 兼容 Python2/3 的 print 函数写法

from collections import namedtuple  # 具名元组：定义标签的数据结构（不可变、可用字段名访问）
from matplotlib import pyplot as plt  # matplotlib 绘图库（imshow 调试用）
import matplotlib.image as mpimg      # matplotlib 的图像读取模块
import numpy as np    # numpy：数组操作与 .npy 保存
from PIL import Image  # 【新环境适配】PIL（Pillow）图像读取：替代已彻底移除的 scipy.misc.imread（scipy>=1.10 删除；imageio 是独立包，不是 scipy 子模块）
import random         # 随机数库（本文件实际未使用）
import os             # os：目录遍历、路径拼接


#############################
    # global variables #
#############################
# 全局变量：Cityscapes 数据集的目录与文件路径
root_dir  = "CityScapes/"  # 数据集根目录

label_dir = os.path.join(root_dir, "gtFine")  # 官方精标注目录（内含 train/val/test 三个子目录）
train_dir = os.path.join(label_dir, "train")  # 训练集标注子目录（按城市分文件夹）
val_dir   = os.path.join(label_dir, "val")    # 验证集标注子目录
test_dir  = os.path.join(label_dir, "test")   # 测试集标注子目录

# create dir for label index
label_idx_dir = os.path.join(root_dir, "Labeled_idx")  # 索引标签（.npy）输出根目录
train_idx_dir = os.path.join(label_idx_dir, "train")   # 训练集索引标签输出目录
val_idx_dir   = os.path.join(label_idx_dir, "val")     # 验证集索引标签输出目录
test_idx_dir  = os.path.join(label_idx_dir, "test")    # 测试集索引标签输出目录
for dir in [train_idx_dir, val_idx_dir, test_idx_dir]:  # 三个输出目录逐个检查
    if not os.path.exists(dir):  # 若不存在
        os.makedirs(dir)         # 则创建

train_file = os.path.join(root_dir, "train.csv")  # 待生成的训练集列表
val_file   = os.path.join(root_dir, "val.csv")    # 待生成的验证集列表
test_file  = os.path.join(root_dir, "test.csv")   # 待生成的测试集列表

color2index = {}  # 映射：RGB 颜色 → trainId 类别索引

# 具名元组：描述 Cityscapes 每个类别的属性结构
Label = namedtuple('Label', [
                   'name',          # 类别名（如 road、car）
                   'id',            # 官方类别 id（0~33，含 void）
                   'trainId',       # 训练用 id（有效类为 1~19；255 表示 void 不参与训练）
                   'category',      # 大类别（void/flat/construction/object/nature/sky/human/vehicle）
                   'categoryId',    # 大类别 id（0~7）
                   'hasInstances',  # 是否为可数个体目标（如车、行人，可用于实例分割）
                   'ignoreInEval',  # 评估时是否忽略该类
                   'color'])        # 彩色标注图中该类的 RGB 颜色

# Cityscapes 官方 34 个类别定义（转录自官方 labels.py，各列含义见上方 namedtuple）
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),  # 未标注（void，忽略）
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),  # 自车（安装相机的车辆）
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),  # 图像矫正边框区域
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),  # 感兴趣区域（ROI）之外
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),  # 静态物体
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),  # 动态物体
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),  # 地面
    Label(  'road'                 ,  7 ,        1 , 'flat'            , 1       , False        , False        , (128, 64,128) ),  # 道路（trainId=1）
    Label(  'sidewalk'             ,  8 ,        2 , 'flat'            , 1       , False        , False        , (244, 35,232) ),  # 人行道（trainId=2）
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),  # 停车场
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),  # 铁轨
    Label(  'building'             , 11 ,        3 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),  # 建筑物（trainId=3）
    Label(  'wall'                 , 12 ,        4 , 'construction'    , 2       , False        , False        , (102,102,156) ),  # 墙体（trainId=4）
    Label(  'fence'                , 13 ,        5 , 'construction'    , 2       , False        , False        , (190,153,153) ),  # 栅栏（trainId=5）
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),  # 护栏
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),  # 桥
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),  # 隧道
    Label(  'pole'                 , 17 ,        6 , 'object'          , 3       , False        , False        , (153,153,153) ),  # 电线杆（trainId=6）
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),  # 杆群
    Label(  'traffic light'        , 19 ,        7 , 'object'          , 3       , False        , False        , (250,170, 30) ),  # 红绿灯（trainId=7）
    Label(  'traffic sign'         , 20 ,        8 , 'object'          , 3       , False        , False        , (220,220,  0) ),  # 交通标志（trainId=8）
    Label(  'vegetation'           , 21 ,        9 , 'nature'          , 4       , False        , False        , (107,142, 35) ),  # 植被（trainId=9）
    Label(  'terrain'              , 22 ,       10 , 'nature'          , 4       , False        , False        , (152,251,152) ),  # 地形/草丛（trainId=10）
    Label(  'sky'                  , 23 ,       11 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),  # 天空（trainId=11）
    Label(  'person'               , 24 ,       12 , 'human'           , 6       , True         , False        , (220, 20, 60) ),  # 行人（trainId=12）
    Label(  'rider'                , 25 ,       13 , 'human'           , 6       , True         , False        , (255,  0,  0) ),  # 骑手（trainId=13）
    Label(  'car'                  , 26 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),  # 轿车（trainId=14）
    Label(  'truck'                , 27 ,       15 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),  # 卡车（trainId=15）
    Label(  'bus'                  , 28 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),  # 公交车（trainId=16）
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),  # 房车（忽略）
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),  # 拖车（忽略）
    Label(  'train'                , 31 ,       17 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),  # 列车（trainId=17）
    Label(  'motorcycle'           , 32 ,       18 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),  # 摩托车（trainId=18）
    Label(  'bicycle'              , 33 ,       19 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),  # 自行车（trainId=19）
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),  # 车牌（特殊类，忽略）
]  # labels 列表定义结束（Cityscapes 官方 34 个类别）


def parse_label():  # 解析标签：建颜色→索引映射，并把三个子集的彩色标签图转成 .npy 索引矩阵
    # change label to class index
    color2index[(0,0,0)] = 0  # add an void class  # 先注册 void 类：黑色 (0,0,0) → 索引 0（多个 void 类共用黑色）
    for obj in labels:  # 遍历 34 个官方类别
        if obj.ignoreInEval:  # 评估时忽略的类（void 类等）
            continue  # 跳过，不加入映射表（它们的颜色将查不到索引）
        idx   = obj.trainId  # 取该类的训练 id（1~19）
        label = obj.name     # 取类别名（此变量此处未再使用）
        color = obj.color    # 取该类的 RGB 颜色
        color2index[color] = idx  # 填充映射：颜色 → trainId 索引

    # 【新环境适配·提速】预建颜色查找表（LUT）：把上面的 color2index 字典压平为
    # 256×256×256 数组（R/G/B 作下标、trainId 作值），供下方整图花式索引查表，
    # 替代原双重 for 循环逐像素查表（Cityscapes 每张 1024×2048≈210 万像素，
    # 纯 Python 每张需 1~2 分钟、全数据集约 5000 张需数天；向量化后每张毫秒级，结果完全等价）
    lut = np.full((256, 256, 256), 19, dtype=np.uint8)  # 查找表默认填 19：查不到的颜色归入兜底类，与原代码 except 分支一致
    for color, index in color2index.items():  # 遍历已注册的 颜色 → trainId 映射
        lut[color[0], color[1], color[2]] = index  # 把索引写入 LUT 对应位置（黑色 (0,0,0)→0 会覆盖默认值，与原逻辑一致）

    # parse train, val, test data    
    # 依次处理 训练/验证/测试 三个子集：zip 把 标注目录、索引输出目录、csv 文件 一一配对
    for label_dir, index_dir, csv_file in zip([train_dir, val_dir, test_dir], [train_idx_dir, val_idx_dir, test_idx_dir], [train_file, val_file, test_file]):
        f = open(csv_file, "w")  # 新建（覆写）对应的 csv 列表文件
        f.write("img,label\n")  # 写表头
        for city in os.listdir(label_dir):  # 遍历该子集下的每个城市文件夹（Cityscapes 按城市组织）
            city_dir = os.path.join(label_dir, city)  # 当前城市的标注目录
            city_idx_dir = os.path.join(index_dir, city)  # 当前城市的索引标签输出目录
            data_dir = city_dir.replace("gtFine", "leftImg8bit")  # 把标注路径换成对应的原图目录（官方数据两棵目录树同构）
            if not os.path.exists(city_idx_dir):  # 若该城市的输出目录不存在
                os.makedirs(city_idx_dir)         # 创建
            for filename in os.listdir(city_dir):  # 遍历该城市目录下的所有标注文件
                if 'color' not in filename:  # 只处理彩色标签（*_gtFine_color.png），跳过 labelIds/instanceIds 等其他文件
                    continue  # 跳过
                lab_name = os.path.join(city_idx_dir, filename)  # 索引标签的保存路径（与标注同名，稍后存 .npy）
                img_name = filename.split("gtFine")[0] + "leftImg8bit.png"  # 把文件名中的 gtFine 换成 leftImg8bit → 原图文件名
                img_name = os.path.join(data_dir, img_name)  # 拼出原图完整路径
                f.write("{},{}.npy\n".format(img_name, lab_name))  # 向 csv 写入一行 "原图路径,标签.npy 路径"

                if os.path.exists(lab_name + '.npy'):  # 若该标签已解析过（断点续跑）
                    print("Skip %s" % (filename))  # 打印跳过
                    continue  # 处理下一个
                print("Parse %s" % (filename))  # 打印正在解析
                img = os.path.join(city_dir, filename)  # 彩色标签图完整路径
                img = np.array(Image.open(img).convert('RGB'))  # 读入并转为 RGB（【新环境适配】替代已移除的 scipy.misc.imread；此处为彩色标签图）
                height, weight, _ = img.shape  # 取高与宽（变量名 weight 实为 width）
        
                # 【新环境适配·提速】整图向量化查表：用上方预建的 lut 一次完成全部像素的
                # 颜色→trainId 转换（毫秒级），与原双重 for 循环结果完全一致
                # （查不到的颜色归 19 兜底；原始实现见下方保留的注释）。
                idx_mat = lut[img[..., 0], img[..., 1], img[..., 2]]  # (H,W,3) 彩色标签 → (H,W) trainId 索引矩阵
                # ---- 原始实现（保留备查，与上行完全等价但极慢）----
                # idx_mat = np.zeros((height, weight))  # 新建同尺寸索引矩阵，默认 0
                # for h in range(height):  # 逐行遍历
                #     for w in range(weight):  # 逐列遍历（逐像素查表，非常耗时）
                #         color = tuple(img[h, w])  # 该像素的 (R, G, B)
                #         try:  # 尝试查表：颜色 → trainId
                #             index = color2index[color]  # 查表：颜色 → trainId
                #             idx_mat[h, w] = index  # 写入索引
                #         except:  # 查表失败（该颜色未注册，例如被跳过的 void 类颜色）
                #             # no index, assign to void
                #             idx_mat[h, w] = 19  # 查不到的颜色（如被跳过的 void 类颜色）统一赋 19 兜底
                #                                 # 【注意】19 同时也是 bicycle 的 trainId，属于原代码的潜在混淆点
                idx_mat = idx_mat.astype(np.uint8)  # 转为 8 位无符号整数
                np.save(lab_name, idx_mat)  # 保存 .npy 到输出目录
                print("Finish %s" % (filename))  # 打印完成


'''debug function'''
def imshow(img, title=None):  # 调试用图像显示函数：参数可以是文件路径或数组
    try:  # 先假设传入的是文件路径
        img = mpimg.imread(img)  # 尝试按路径读取图像
        imgplot = plt.imshow(img)  # 并显示
    except:  # 读取失败说明传入的是数组
        plt.imshow(img, interpolation='nearest')  # 读取失败则当作数组显示（最近邻插值）

    if title is not None:  # 若提供了标题
        plt.title(title)  # 设置标题
    
    plt.show()  # 弹窗显示


if __name__ == '__main__':  # 直接运行本文件时执行标签解析
    parse_label()  # 开始解析 Cityscapes 全部标签
