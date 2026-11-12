# ============================================================================
# 【文件说明】__init__.py —— 数据集加载子包（dataloader）的包入口文件：
#             汇总导入各数据集类，维护“名称 → 数据集类”映射表 datasets，
#             并对外提供统一工厂函数 get_segmentation_dataset(name, **kwargs)，
#             供训练/评估脚本按数据集名称创建 Dataset 实例。
# 【数据集】已注册 6 个数据集：ade20k(ADE20K,150类) / pascal_voc(VOC2012,21类) /
#           pascal_aug(VOC+SBD 增强,21类) / coco(COCO 转 VOC,21类) /
#           citys(Cityscapes,19类) / sbu(SBU 阴影,2类)。
#           【注意】lip_parsing.py 中的 LIP 人体解析数据集（20类）未注册进
#           映射表，只能直接 import core.data.dataloader.lip_parsing 使用。
# ============================================================================
"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .mscoco import COCOSegmentation  # 导入 MS COCO 分割数据集类（用于 VOC 预训练）
from .cityscapes import CitySegmentation  # 导入 Cityscapes 街景分割数据集类
from .ade import ADE20KSegmentation  # 导入 ADE20K 场景解析数据集类
from .pascal_voc import VOCSegmentation  # 导入 PASCAL VOC 2012 分割数据集类
from .pascal_aug import VOCAugSegmentation  # 导入 VOC 增强版（SBD 边界增强）数据集类
from .sbu_shadow import SBUSegmentation  # 导入 SBU 阴影分割数据集类

# 数据集名称（小写字符串）→ 数据集类的映射表，是工厂函数的查找依据
datasets = {
    'ade20k': ADE20KSegmentation,  # ADE20K 场景解析数据集（150 类）
    'pascal_voc': VOCSegmentation,  # PASCAL VOC 2012 原始分割数据集（21 类含背景）
    'pascal_aug': VOCAugSegmentation,  # VOC + SBD 边界增强数据集（21 类含背景）
    'coco': COCOSegmentation,  # MS COCO 实例标注转 VOC 21 类（用于预训练）
    'citys': CitySegmentation,  # Cityscapes 街景数据集（19 类）
    'sbu': SBUSegmentation,  # SBU Shadow 阴影数据集（2 类）
}  # 映射表定义结束


# 数据集工厂函数：按名称创建对应的数据集实例
# name: 数据集名称（见上方 datasets 字典的键，大小写不敏感）
# **kwargs: 透传给数据集构造器的参数（root/split/mode/transform/base_size/crop_size 等）
def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""  # 原英文 docstring：语义分割数据集工厂函数说明
    return datasets[name.lower()](**kwargs)  # 名称转小写后查表并实例化；未注册的名称会抛 KeyError
