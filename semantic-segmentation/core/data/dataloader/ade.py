# ============================================================================
# 【文件说明】ade.py —— ADE20K 场景解析（Scene Parsing）语义分割数据集
#             加载器：在 ADEChallengeData2016 目录下配对 images/*.jpg 与
#             annotations/*.png，并在 _mask_transform 中把标签 1~150 统一
#             减 1 映射为 0~149、未标注 0 映射为忽略类 -1。
# 【数据集】ADE20K（MIT 场景解析基准）：150 类（NUM_CLASS=150，另有忽略类）；
#           20210 张训练图 + 2000 张验证图；
#           目录结构（root 默认 ../datasets/ade）：
#             ADEChallengeData2016/images/training|validation/*.jpg      原图
#             ADEChallengeData2016/annotations/training|validation/*.png 标注掩膜
#           标签格式：掩膜像素值 0 = 未标注（忽略），1~150 = 类别（内部转为
#           0~149）；推理结果需 +pred_offset(=1) 才是官方标签 id。
#           【注意】本类默认 split='test'（与其他数据集默认 'train' 不同），
#           且 'val'/'test' 等所有非 train 划分都读取 validation 目录。
# ============================================================================
"""Pascal ADE20K Semantic Segmentation Dataset."""
import os  # 标准库：路径拼接/存在性检查/列目录/拆分扩展名
import torch  # 深度学习框架：把掩膜转为 LongTensor
import numpy as np  # 数值库：掩膜转 int32 数组

from PIL import Image  # PIL 图像读取与格式转换
from .segbase import SegmentationDataset  # 相对导入分割数据集基类（同步增广逻辑所在）


# ADE20K 场景解析数据集类：继承基类增广，自实现目录配对与标签 -1 偏移
class ADE20KSegmentation(SegmentationDataset):
    # 类 docstring：ADE20K 语义分割数据集说明（数据路径/划分/变换参数与用法示例）
    """ADE20K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/ade'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = ADE20KSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'ADEChallengeData2016'  # 数据集子目录名（root 之下）
    NUM_CLASS = 150  # 类别数：150 个场景/物体类别（忽略类不计入）

    # root: 数据集根目录，默认 '../datasets/ade'
    # split: 'train' 或 'val'（【注意】默认 'test'，且非 train 一律按验证目录处理，见 _get_ade20k_pairs）
    # mode: 'train'/'val'/'test'/'testval'，None 时等于 split；transform: 图像变换；**kwargs: 透传基类参数
    def __init__(self, root='../datasets/ade', split='test', mode=None, transform=None, **kwargs):
        super(ADE20KSegmentation, self).__init__(root, split, mode, transform, **kwargs)  # 初始化基类（保存 root/split/mode/transform/增广尺寸）
        root = os.path.join(root, self.BASE_DIR)  # 拼接实际数据目录并重绑定局部变量 root（self.root 仍保持原始根路径）
        assert os.path.exists(root), "Please setup the dataset using ../datasets/ade20k.py"  # 断言数据目录存在，否则提示先运行下载脚本
        self.images, self.masks = _get_ade20k_pairs(root, split)  # 扫描目录得到配对的图像/掩膜路径列表
        assert (len(self.images) == len(self.masks))  # 断言图像与掩膜数量一致
        if len(self.images) == 0:  # 未找到任何图像
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")  # 报错提示数据路径有误
        print('Found {} images in the folder {}'.format(len(self.images), root))  # 打印找到的样本数

    # 按索引取一条样本：读图 → 按 mode 增广 → 转 Tensor
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')  # 打开图像并统一转 RGB 三通道
        if self.mode == 'test':  # 纯测试模式：无标注，只返回变换后的图像与文件名
            img = self._img_transform(img)  # 先转 ndarray 再送 transform（【注意】与 cityscapes/sbu 直接传 PIL 的写法不一致，若 transform 含仅支持 PIL 的操作会出错）
            if self.transform is not None:  # 若提供了外部变换
                img = self.transform(img)  # 应用变换（ToTensor/Normalize）
            return img, os.path.basename(self.images[index])  # 返回 (图像张量, 文件名)
        mask = Image.open(self.masks[index])  # 打开对应标注掩膜 PNG
        # synchrosized transform
        if self.mode == 'train':  # 训练模式
            img, mask = self._sync_transform(img, mask)  # 随机增广（翻转/缩放/裁剪/模糊），img 与 mask 同步
        elif self.mode == 'val':  # 验证模式
            img, mask = self._val_sync_transform(img, mask)  # 确定性变换（等比缩放 + 中心裁剪）
        else:  # 其余模式只允许 testval
            assert self.mode == 'testval'  # testval：不做几何增广，原尺寸直接转数组（用于全图评估）
            img, mask = self._img_transform(img), self._mask_transform(mask)  # 图像转 ndarray，掩膜转 Tensor 并完成 -1 偏移
        # general resize, normalize and to Tensor
        if self.transform is not None:  # 若提供了外部变换
            img = self.transform(img)  # 仅对图像做 ToTensor + ImageNet Normalize（掩膜不归一化）
        return img, mask, os.path.basename(self.images[index])  # 返回 (图像张量, 标签, 文件名)

    # 掩膜后处理：PIL → int32 数组，标签统一减 1（1~150 → 0~149；0 未标注 → -1 忽略）
    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32') - 1)  # 转 int32 后减 1，再转 LongTensor（交叉熵要求整型标签）

    def __len__(self):  # 数据集样本总数
        return len(self.images)  # 以图像列表长度为准

    @property  # property 装饰器
    def pred_offset(self):  # 预测偏移：模型输出 0~149，+1 后才是官方标签 1~150
        return 1  # ADE20K 官方标签从 1 开始，故偏移为 1

    @property  # property 装饰器
    def classes(self):  # 150 个类别名称元组（顺序与官方标签 1~150 一一对应）
        """Category names."""  # 原英文 docstring：类别名称列表
        return ("wall", "building, edifice", "sky", "floor, flooring", "tree",  # 1 墙 / 2 建筑 / 3 天空 / 4 地板 / 5 树
                "ceiling", "road, route", "bed", "windowpane, window", "grass",  # 6 天花板 / 7 道路 / 8 床 / 9 窗 / 10 草
                "cabinet", "sidewalk, pavement",  # 11 柜子 / 12 人行道
                "person, individual, someone, somebody, mortal, soul",  # 13 人
                "earth, ground", "door, double door", "table", "mountain, mount",  # 14 地面 / 15 门 / 16 桌子 / 17 山
                "plant, flora, plant life", "curtain, drape, drapery, mantle, pall",  # 18 植物 / 19 窗帘
                "chair", "car, auto, automobile, machine, motorcar",  # 20 椅子 / 21 汽车
                "water", "painting, picture", "sofa, couch, lounge", "shelf",  # 22 水 / 23 画 / 24 沙发 / 25 架子
                "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair",  # 26 房子 / 27 海 / 28 镜子 / 29 地毯 / 30 田野 / 31 扶手椅
                "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press",  # 32 座位 / 33 栅栏 / 34 书桌 / 35 岩石 / 36 衣柜
                "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion",  # 37 灯 / 38 浴缸 / 39 栏杆 / 40 垫子
                "base, pedestal, stand", "box", "column, pillar", "signboard, sign",  # 41 底座 / 42 箱子 / 43 柱子 / 44 招牌
                "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink",  # 45 五斗柜 / 46 柜台 / 47 沙 / 48 水槽
                "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox",  # 49 摩天楼 / 50 壁炉 / 51 冰箱
                "grandstand, covered stand", "path", "stairs, steps", "runway",  # 52 看台 / 53 小径 / 54 楼梯 / 55 跑道
                "case, display case, showcase, vitrine",  # 56 陈列柜
                "pool table, billiard table, snooker table", "pillow",  # 57 台球桌 / 58 枕头
                "screen door, screen", "stairway, staircase", "river", "bridge, span",  # 59 纱门 / 60 楼梯间 / 61 河流 / 62 桥
                "bookcase", "blind, screen", "coffee table, cocktail table",  # 63 书柜 / 64 百叶窗 / 65 咖啡桌
                "toilet, can, commode, crapper, pot, potty, stool, throne",  # 66 马桶
                "flower", "book", "hill", "bench", "countertop",  # 67 花 / 68 书 / 69 小山 / 70 长凳 / 71 台面
                "stove, kitchen stove, range, kitchen range, cooking stove",  # 72 炉子
                "palm, palm tree", "kitchen island",  # 73 棕榈树 / 74 厨房岛台
                "computer, computing machine, computing device, data processor, "  # 75 计算机（与下一行字符串隐式拼接）
                "electronic computer, information processing system",  # 75 计算机（续）
                "swivel chair", "boat", "bar", "arcade machine",  # 76 转椅 / 77 船 / 78 酒吧 / 79 街机
                "hovel, hut, hutch, shack, shanty",  # 80 简陋小屋
                "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, "  # 81 公交车（与下一行字符串隐式拼接）
                "motorcoach, omnibus, passenger vehicle",  # 81 公交车（续）
                "towel", "light, light source", "truck, motortruck", "tower",  # 82 毛巾 / 83 灯 / 84 卡车 / 85 塔
                "chandelier, pendant, pendent", "awning, sunshade, sunblind",  # 86 吊灯 / 87 遮阳棚
                "streetlight, street lamp", "booth, cubicle, stall, kiosk",  # 88 路灯 / 89 亭子
                "television receiver, television, television set, tv, tv set, idiot "  # 90 电视（与下一行字符串隐式拼接）
                "box, boob tube, telly, goggle box",  # 90 电视（续）
                "airplane, aeroplane, plane", "dirt track",  # 91 飞机 / 92 土路
                "apparel, wearing apparel, dress, clothes",  # 93 服装
                "pole", "land, ground, soil",  # 94 杆 / 95 陆地
                "bannister, banister, balustrade, balusters, handrail",  # 96 扶手栏杆
                "escalator, moving staircase, moving stairway",  # 97 自动扶梯
                "ottoman, pouf, pouffe, puff, hassock",  # 98 软凳
                "bottle", "buffet, counter, sideboard",  # 99 瓶子 / 100 餐具柜
                "poster, posting, placard, notice, bill, card",  # 101 海报
                "stage", "van", "ship", "fountain",  # 102 舞台 / 103 厢式车 / 104 船 / 105 喷泉
                "conveyer belt, conveyor belt, conveyer, conveyor, transporter",  # 106 传送带
                "canopy", "washer, automatic washer, washing machine",  # 107 天篷 / 108 洗衣机
                "plaything, toy", "swimming pool, swimming bath, natatorium",  # 109 玩具 / 110 游泳池
                "stool", "barrel, cask", "basket, handbasket", "waterfall, falls",  # 111 凳子 / 112 桶 / 113 篮子 / 114 瀑布
                "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle",  # 115 帐篷 / 116 包 / 117 小型摩托 / 118 摇篮
                "oven", "ball", "food, solid food", "step, stair", "tank, storage tank",  # 119 烤箱 / 120 球 / 121 食物 / 122 台阶 / 123 储罐
                "trade name, brand name, brand, marque", "microwave, microwave oven",  # 124 商标 / 125 微波炉
                "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna",  # 126 花盆 / 127 动物
                "bicycle, bike, wheel, cycle", "lake",  # 128 自行车 / 129 湖
                "dishwasher, dish washer, dishwashing machine",  # 130 洗碗机
                "screen, silver screen, projection screen",  # 131 银幕
                "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase",  # 132 毯子 / 133 雕塑 / 134 排气罩 / 135 壁灯 / 136 花瓶
                "traffic light, traffic signal, stoplight", "tray",  # 137 红绿灯 / 138 托盘
                "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, "  # 139 垃圾桶（与下一行字符串隐式拼接）
                "dustbin, trash barrel, trash bin",  # 139 垃圾桶（续）
                "fan", "pier, wharf, wharfage, dock", "crt screen",  # 140 风扇 / 141 码头 / 142 CRT 屏幕
                "plate", "monitor, monitoring device", "bulletin board, notice board",  # 143 盘子 / 144 显示器 / 145 公告板
                "shower", "radiator", "glass, drinking glass", "clock", "flag")  # 146 淋浴 / 147 散热器 / 148 杯子 / 149 钟 / 150 旗


# 扫描目录构造 ADE20K 的 (图像路径, 掩膜路径) 配对列表（模块级辅助函数）
# folder: ADEChallengeData2016 目录；mode: 形参名虽叫 mode，实际传入的是 split（'train' 走训练目录，其余一律走验证目录）
def _get_ade20k_pairs(folder, mode='train'):
    img_paths = []  # 图像路径累积列表
    mask_paths = []  # 掩膜路径累积列表（与 img_paths 按下标一一对应）
    if mode == 'train':  # 训练划分
        img_folder = os.path.join(folder, 'images/training')  # 训练图像目录
        mask_folder = os.path.join(folder, 'annotations/training')  # 训练标注目录
    else:  # 【注意】'val'/'test' 等所有非 train 划分共用 validation 目录
        img_folder = os.path.join(folder, 'images/validation')  # 验证图像目录
        mask_folder = os.path.join(folder, 'annotations/validation')  # 验证标注目录
    for filename in os.listdir(img_folder):  # 遍历图像目录下所有文件（未排序，顺序取决于文件系统）
        basename, _ = os.path.splitext(filename)  # 拆出不含扩展名的主文件名（【注意】在 jpg 过滤之前计算，对非 jpg 文件属于多余开销）
        if filename.endswith(".jpg"):  # 只处理 jpg 图像
            imgpath = os.path.join(img_folder, filename)  # 拼出图像完整路径
            maskname = basename + '.png'  # 掩膜文件名 = 同名 .png
            maskpath = os.path.join(mask_folder, maskname)  # 拼出掩膜完整路径
            if os.path.isfile(maskpath):  # 掩膜存在才收录
                img_paths.append(imgpath)  # 收录图像路径
                mask_paths.append(maskpath)  # 收录掩膜路径
            else:  # 缺少标注
                print('cannot find the mask:', maskpath)  # 打印缺失信息（仅警告，不中断）

    return img_paths, mask_paths  # 返回配对结果


if __name__ == '__main__':  # 直接运行本文件时的自测入口
    train_dataset = ADE20KSegmentation()  # 实例化数据集（【注意】变量名叫 train_dataset，但默认 split='test'，实际加载的是验证集）
