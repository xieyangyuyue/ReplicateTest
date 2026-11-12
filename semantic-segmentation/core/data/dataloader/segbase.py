# ============================================================================
# 【文件说明】segbase.py —— 所有语义分割数据集的公共基类 SegmentationDataset，
#             定义“图像-掩膜同步”的数据增广与后处理流程：
#             · 训练阶段 _sync_transform（随机增广）：50% 概率水平翻转 →
#               随机缩放（目标短边在 [0.5×base_size, 2.0×base_size] 区间内
#               均匀采样整数）→ 宽/高不足 crop_size 时向右、向下补 0 →
#               随机裁剪 crop_size×crop_size → 50% 概率高斯模糊（PSPNet
#               同款增广，半径在 [0,1) 随机，仅作用于图像）；
#             · 验证阶段 _val_sync_transform（确定性变换）：等比缩放至短边
#               = crop_size（图像双线性、掩膜最近邻）→ 中心裁剪 crop_size×
#               crop_size，全程无随机性，保证评估结果可复现；
#             · 归一化/ToTensor 不在本类完成：由外部传入的 transform（通常为
#               torchvision 的 ToTensor + Normalize，使用 ImageNet 均值
#               (.485,.456,.406)、方差 (.229,.224,.225)）在子类 __getitem__
#               末尾仅对图像执行，掩膜不参与归一化。
# 【数据集】基类不绑定具体数据集；默认 base_size=520、crop_size=480，
#           小图数据集（LIP/SBU）会覆写为 280/256。掩膜一律用最近邻
#           (NEAREST) 插值缩放，避免标签值被插值“造”出不存在的类别。
# ============================================================================
"""Base segmentation dataset"""
import random  # 标准库随机模块：训练增广中的随机翻转/缩放/裁剪位置/模糊半径
import numpy as np  # 数值计算库：把 PIL 图像与掩膜转为 ndarray

from PIL import Image, ImageOps, ImageFilter  # PIL：Image（读写/缩放/裁剪/翻转）、ImageOps（边缘补 0）、ImageFilter（高斯模糊）

__all__ = ['SegmentationDataset']  # 模块公开接口：仅导出 SegmentationDataset 基类


# 语义分割数据集基类：实现 __len__/__getitem__ 数据集协议与同步增广接口，具体目录扫描与标签转换由各数据集子类完成
class SegmentationDataset(object):
    """Segmentation Base Dataset"""  # 原英文 docstring：分割数据集基类说明

    # ---- 构造参数说明 ----
    # root: 数据集根目录（子类在其下拼接具体子目录）
    # split: 数据划分名，如 'train'/'val'/'test'/'trainval'（由子类解析为文件列表）
    # mode: 运行模式 'train'/'val'/'test'/'testval'，决定 __getitem__ 的增广分支；None 时回退为 split
    # transform: 外部图像变换（一般 ToTensor + Normalize），只作用于图像不作用于掩膜
    # base_size: 随机缩放基准尺寸（短边），训练时实际短边在其 0.5~2.0 倍之间随机
    # crop_size: 训练随机裁剪与验证中心裁剪的正方形边长
    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__()  # 调用 object 基类构造（无实际逻辑，保持继承写法完整）
        self.root = root  # 保存数据集根目录
        self.transform = transform  # 保存外部图像变换（ToTensor/Normalize 等）
        self.split = split  # 保存数据划分名
        self.mode = mode if mode is not None else split  # mode 缺省时取 split：默认按 split 对应的模式取数
        self.base_size = base_size  # 保存随机缩放基准短边尺寸
        self.crop_size = crop_size  # 保存裁剪边长

    # ---- 验证/测试阶段同步变换（_val 系列 = 确定性预处理）：无随机性，结果可复现 ----
    # img: PIL 图像；mask: PIL 掩膜；返回同步处理后的 (img, mask)
    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size  # 目标输出（裁剪）边长 = crop_size
        short_size = outsize  # 等比缩放的目标短边也取 crop_size
        w, h = img.size  # 读取当前图像宽 w、高 h
        if w > h:  # 横图（宽 > 高）：把高度固定为 short_size
            oh = short_size  # 缩放后高 = crop_size
            ow = int(1.0 * w * oh / h)  # 按纵横比等比计算缩放后宽（向零截断取整）
        else:  # 竖图或方图：把宽度固定为 short_size
            ow = short_size  # 缩放后宽 = crop_size
            oh = int(1.0 * h * ow / w)  # 按纵横比等比计算缩放后高（向零截断取整）
        img = img.resize((ow, oh), Image.BILINEAR)  # 图像用双线性插值缩放（平滑，保画质）
        mask = mask.resize((ow, oh), Image.NEAREST)  # 掩膜用最近邻缩放（标签值不被插值污染）
        # center crop
        w, h = img.size  # 重新读取缩放后的宽高
        x1 = int(round((w - outsize) / 2.))  # 中心裁剪窗口左上角 x（四舍五入取整）
        y1 = int(round((h - outsize) / 2.))  # 中心裁剪窗口左上角 y（四舍五入取整）
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))  # 裁出中心 outsize×outsize 的图像块
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))  # 用同一窗口裁掩膜，保证像素级对齐
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)  # 图像转 ndarray、掩膜转 int32（子类可重写：转 Tensor/重映射标签）
        return img, mask  # 返回同步变换后的 (图像, 掩膜)

    # ---- 训练阶段同步变换：随机增广，img 与 mask 使用同一组随机参数 ----
    # img: PIL 图像；mask: PIL 掩膜；返回增广后的 (img, mask)
    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:  # 以 50% 概率执行水平翻转
            img = img.transpose(Image.FLIP_LEFT_RIGHT)  # 图像左右镜像
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)  # 掩膜同步左右镜像，保持对齐
        crop_size = self.crop_size  # 目标裁剪边长
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))  # 随机目标短边：在 [0.5×base_size, 2.0×base_size] 内均匀取整数（默认 520 时即 260~1040）
        w, h = img.size  # 读取当前图像宽高
        if h > w:  # 竖图（高 > 宽）：把宽度缩放到 short_size
            ow = short_size  # 缩放后宽
            oh = int(1.0 * h * ow / w)  # 按纵横比等比计算缩放后高（向零截断取整）
        else:  # 横图或方图：把高度缩放到 short_size
            oh = short_size  # 缩放后高
            ow = int(1.0 * w * oh / h)  # 按纵横比等比计算缩放后宽（向零截断取整）
        img = img.resize((ow, oh), Image.BILINEAR)  # 图像用双线性插值缩放
        mask = mask.resize((ow, oh), Image.NEAREST)  # 掩膜用最近邻缩放
        # pad crop
        if short_size < crop_size:  # 缩放后短边小于裁剪边长时需要补边（保证能裁出 crop_size×crop_size）
            padh = crop_size - oh if oh < crop_size else 0  # 高度不足时向下补 crop_size-oh 个像素，否则补 0
            padw = crop_size - ow if ow < crop_size else 0  # 宽度不足时向右补 crop_size-ow 个像素，否则补 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)  # 仅在右、下两侧补 0（黑边）
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)  # 掩膜同样补 0（0 的语义由各数据集定义，多为背景/忽略）
        # random crop crop_size
        w, h = img.size  # 读取（可能补边后）的宽高
        x1 = random.randint(0, w - crop_size)  # 裁剪框左上角 x 在 [0, w-crop_size] 内均匀随机
        y1 = random.randint(0, h - crop_size)  # 裁剪框左上角 y 在 [0, h-crop_size] 内均匀随机
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))  # 按随机窗口裁剪图像
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))  # 用同一窗口裁掩膜，保持像素级对齐
        # gaussian blur as in PSP
        if random.random() < 0.5:  # 50% 概率高斯模糊（PSPNet 论文同款增广）
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))  # 模糊半径在 [0,1) 随机；仅模糊图像，掩膜不变
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)  # 图像转 ndarray、掩膜转 int32（子类可重写：转 Tensor/重映射标签）
        return img, mask  # 返回增广后的 (图像, 掩膜)

    # 图像后处理：PIL → numpy.ndarray（H×W×C，uint8，RGB 顺序），子类一般不重写
    def _img_transform(self, img):
        return np.array(img)  # 直接转 ndarray

    # 掩膜后处理默认实现：PIL → int32 ndarray；各数据集子类通常重写（转 LongTensor、重映射/偏移标签）
    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')  # 转 int32，像素值即类别标签

    @property  # property 装饰器：以 dataset.num_class 形式访问
    def num_class(self):  # 类别数属性
        """Number of categories."""  # 原英文 docstring：类别数量
        return self.NUM_CLASS  # 返回子类定义的类别总数（类属性 NUM_CLASS）

    @property  # property 装饰器：以 dataset.pred_offset 形式访问
    def pred_offset(self):  # 预测偏移属性：模型输出的类别索引 + offset = 原数据集标签 id
        return 0  # 默认无偏移；ADE20K 覆写为 1（其原始标签从 1 开始）
