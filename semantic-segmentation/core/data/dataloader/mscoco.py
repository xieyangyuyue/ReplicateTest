# ============================================================================
# 【文件说明】mscoco.py —— MS COCO 语义分割数据集加载器（面向 VOC 预训练）：
#             用 pycocotools 读取 COCO 2017 实例标注，把与 VOC 20 物体类
#             对应的实例掩膜合成为单通道语义标签图（uint8，0=背景，1~20=物体）。
# 【数据集】MS COCO 2017（root 默认 ../datasets/coco）：21 类（含背景，
#           NUM_CLASS=21，类别名与 VOC 完全一致）；train2017 约 11.8 万图、
#           val2017 5000 图，仅保留前景像素数 >1000 的“合格”图像（首次运行
#           _preprocess 全量过滤，并把合格 id 用 pickle 缓存到
#           train_ids.mx / val_ids.mx）；目录结构：
#             train2017/、val2017/                        原图
#             annotations/instances_{train,val}2017.json  COCO 实例标注
#             annotations/{train,val}_ids.mx              合格图像 id 的 pickle 缓存
#           CAT_LIST：VOC 类别序号(0~20) → COCO category_id 映射表。
#           【注意】类 docstring 中 “Path to ADE20K folder” 系从 ade.py
#           复制粘贴的笔误，实际应为 COCO 目录。
# ============================================================================
"""MSCOCO Semantic Segmentation pretraining for VOC."""
import os  # 标准库：路径拼接/存在性检查/取文件名
import pickle  # 标准库：合格图像 id 列表的序列化缓存（.mx 文件实为 pickle 格式）
import torch  # 深度学习框架：把掩膜转成 LongTensor
import numpy as np  # 数值库：合成语义掩膜（uint8 矩阵）

from tqdm import trange  # tqdm 进度条：预处理全量 COCO 时显示进度
from PIL import Image  # PIL 图像读取、由 ndarray 构建掩膜图像
from .segbase import SegmentationDataset  # 相对导入分割数据集基类（同步增广逻辑所在）


# MS COCO → VOC 风格分割数据集类：把 COCO 实例标注在线合成为 VOC 21 类语义掩膜
class COCOSegmentation(SegmentationDataset):
    # 类 docstring：COCO 转 VOC 预训练分割数据集说明（数据路径/划分/变换与用法示例）
    """COCO Semantic Segmentation Dataset for VOC Pre-training.

    Parameters
    ----------
    root : string
        Path to ADE20K folder. Default is './datasets/coco'
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
    >>> trainset = COCOSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,  # VOC 类 0~14 对应的 COCO category_id（0 为背景占位，实际无对应类）
                1, 64, 20, 63, 7, 72]  # VOC 类 15~20：人/盆栽/羊/沙发/火车/电视
    NUM_CLASS = 21  # 类别数：背景 + 20 个 VOC 物体类

    # root: 数据集根目录，默认 '../datasets/coco'
    # split: 'train' 或其他（其他一律按验证集 val2017 处理）
    # mode: 'train'/'val'/'test'/'testval'，None 时等于 split；transform: 图像变换；**kwargs: 透传基类参数
    def __init__(self, root='../datasets/coco', split='train', mode=None, transform=None, **kwargs):
        super(COCOSegmentation, self).__init__(root, split, mode, transform, **kwargs)  # 初始化基类（保存 root/split/mode/transform/增广尺寸）
        # lazy import pycocotools
        from pycocotools.coco import COCO  # 惰性导入 COCO 标注读取器（仅在使用本数据集时才要求安装 pycocotools）
        from pycocotools import mask  # 惰性导入 RLE 掩膜编解码工具
        if split == 'train':  # 训练划分
            print('train set')  # 提示当前为训练集
            ann_file = os.path.join(root, 'annotations/instances_train2017.json')  # 训练标注文件
            ids_file = os.path.join(root, 'annotations/train_ids.mx')  # 训练合格图像 id 缓存文件
            self.root = os.path.join(root, 'train2017')  # 训练图像目录（覆盖基类保存的 self.root）
        else:  # 其余一律按验证划分处理
            print('val set')  # 提示当前为验证集
            ann_file = os.path.join(root, 'annotations/instances_val2017.json')  # 验证标注文件
            ids_file = os.path.join(root, 'annotations/val_ids.mx')  # 验证合格图像 id 缓存文件
            self.root = os.path.join(root, 'val2017')  # 验证图像目录（覆盖基类保存的 self.root）
        self.coco = COCO(ann_file)  # 解析 COCO json 标注并建立索引（大文件，耗时操作）
        self.coco_mask = mask  # 保存 RLE 工具引用，供 _gen_seg_mask 使用
        if os.path.exists(ids_file):  # 已有 id 缓存文件
            with open(ids_file, 'rb') as f:  # 二进制只读打开缓存文件
                self.ids = pickle.load(f)  # 反序列化合格图像 id 列表
        else:  # 无缓存则现场预处理生成
            ids = list(self.coco.imgs.keys())  # 取标注中全部图像 id
            self.ids = self._preprocess(ids, ids_file)  # 逐图合成掩膜并过滤，结果写入缓存
        self.transform = transform  # 再次保存 transform（【注意】基类构造里已赋值过，此处属于冗余写法）

    # 按索引取一条样本：按 id 读图 → 由实例标注合成掩膜 → 按 mode 增广 → 转 Tensor
    def __getitem__(self, index):
        coco = self.coco  # 局部引用 COCO 对象
        img_id = self.ids[index]  # 取本样本的 COCO 图像 id（整数）
        img_metadata = coco.loadImgs(img_id)[0]  # 取图像元信息（宽高、文件名等）
        path = img_metadata['file_name']  # 图像相对文件名
        img = Image.open(os.path.join(self.root, path)).convert('RGB')  # 打开图像并统一转 RGB 三通道
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))  # 取该图全部实例标注
        mask = Image.fromarray(self._gen_seg_mask(  # 由实例标注合成 VOC 21 类语义掩膜并转 PIL 图像
            cocotarget, img_metadata['height'], img_metadata['width']))  # 传入图像高宽生成同尺寸掩膜
        # synchrosized transform
        if self.mode == 'train':  # 训练模式
            img, mask = self._sync_transform(img, mask)  # 随机增广（翻转/缩放/裁剪/模糊），img 与 mask 同步
        elif self.mode == 'val':  # 验证模式
            img, mask = self._val_sync_transform(img, mask)  # 确定性变换（等比缩放 + 中心裁剪）
        else:  # 其余模式只允许 testval
            assert self.mode == 'testval'  # testval：不做几何增广，原尺寸直接转数组
            img, mask = self._img_transform(img), self._mask_transform(mask)  # 图像转 ndarray，掩膜转 Tensor
        # general resize, normalize and toTensor
        if self.transform is not None:  # 若提供了外部变换
            img = self.transform(img)  # 仅对图像做 ToTensor + ImageNet Normalize（掩膜不归一化）
        return img, mask, os.path.basename(self.ids[index])  # 【注意】疑似 bug：ids 元素是整数 id，而 os.path.basename 要求字符串参数，此处会抛 TypeError（对比其他数据集此处应传图像路径/文件名）

    # 掩膜后处理：int32 数组 → LongTensor（合成掩膜已是 0~20，无需重映射）
    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))  # 转 int32 后再转 LongTensor

    # 由 COCO 实例标注合成语义分割掩膜
    # target: 实例标注列表；h/w: 图像高宽；返回 uint8 掩膜（0=背景，1~20=VOC 类）
    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)  # 初始化全 0（背景）掩膜
        coco_mask = self.coco_mask  # 局部引用 RLE 编解码工具
        for instance in target:  # 遍历该图每个实例标注
            rle = coco_mask.frPyObjects(instance['Segmentation'], h, w)  # 把多边形/RLE 统一编码为 RLE（【注意】COCO 官方键名为小写 'segmentation'，此处大写 'Segmentation' 对标准 instances json 可能抛 KeyError）
            m = coco_mask.decode(rle)  # RLE 解码为二值掩膜（单个形状为 H×W，多个形状为 H×W×N）
            cat = instance['category_id']  # 该实例的 COCO 类别 id
            if cat in self.CAT_LIST:  # 类别在 VOC 映射表内
                c = self.CAT_LIST.index(cat)  # 查出对应的 VOC 类别序号（0~20）
            else:  # 非 VOC 关注类别
                continue  # 跳过该实例
            if len(m.shape) < 3:  # 单个二值掩膜（H×W）
                mask[:, :] += (mask == 0) * (m * c)  # 仅在仍为背景(0)的像素写入类别 c，已分配过的像素不覆盖
            else:  # 多个二值掩膜（H×W×N）
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)  # 先在通道维求并集（任一形状命中即为前景），再写入类别 c
        return mask  # 返回合成好的语义掩膜

    # 预处理：逐图合成掩膜，筛出前景像素 >1000 的“合格”图像 id 并写入缓存文件（每个 split 只需运行一次）
    def _preprocess(self, ids, ids_file):
        # 提示信息（行尾续行符 \ 后不能再加注释）
        print("Preprocessing mask, this will take a while." + \
              "But don't worry, it only run once for each split.")  # 打印提示：首次预处理较慢，但每个 split 只运行一次
        tbar = trange(len(ids))  # 创建 tqdm 整数迭代进度条
        new_ids = []  # 合格图像 id 累积列表
        for i in tbar:  # 遍历全部图像 id
            img_id = ids[i]  # 当前图像 id
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))  # 取该图全部实例标注
            img_metadata = self.coco.loadImgs(img_id)[0]  # 取图像元信息
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width'])  # 合成该图的语义掩膜
            # more than 1k pixels
            if (mask > 0).sum() > 1000:  # 前景像素超过 1000
                new_ids.append(img_id)  # 视为合格样本收录
            # 更新进度条描述（行尾续行符 \ 后不能再加注释）
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))  # 显示处理进度与已获合格图像数
        print('Found number of qualified images: ', len(new_ids))  # 打印合格图像总数
        with open(ids_file, 'wb') as f:  # 二进制写打开缓存文件
            pickle.dump(new_ids, f)  # 序列化合格 id 列表落盘
        return new_ids  # 返回合格 id 列表

    @property  # property 装饰器
    def classes(self):  # 21 个类别名称元组（与 VOC 完全一致）
        """Category names."""  # 原英文 docstring：类别名称列表
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',  # 0 背景 / 1 飞机 / 2 自行车 / 3 鸟 / 4 船 / 5 瓶子
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',  # 6 公交 / 7 汽车 / 8 猫 / 9 椅子 / 10 牛 / 11 餐桌 / 12 狗 / 13 马
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',  # 14 摩托 / 15 人 / 16 盆栽 / 17 羊 / 18 沙发 / 19 火车
                'tv')  # 20 电视
