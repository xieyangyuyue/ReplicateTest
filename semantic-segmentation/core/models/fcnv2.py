# ============================================================================
# 【文件说明】fcnv2.py —— FCN v2 变体：基于 SegBaseModel + ResNet 骨干的 FCN 实现。
#              与 fcn.py（VGG 骨干、独立实现）不同，本版本复用统一基类的骨干加载与前向接口，
#              提供 get_fcn 通用工厂及三个 Pascal VOC 预设（ResNet50/101/152）。
#              工程位置：core/models/fcnv2.py，经 model_zoo 的 get_segmentation_model('fcn') 调用。
# 【核心思想】FCN(stride=8)：ResNet 骨干以空洞卷积使 c3/c4 保持 1/8 分辨率，
#            主分割头接 c4（2048 通道）、辅助头接 c3（1024 通道），
#            各自经 1x1 卷积输出 nclass 通道得分后双线性上采样 8 倍回原图尺寸；
#            辅助头仅训练期使用，提供深监督以加速收敛、提升稳定性。
# ============================================================================
"""Fully Convolutional Network with Stride of 8"""  # 原 docstring 保留：步长为 8 的全卷积网络
from __future__ import division  # Python 2 兼容语句：Python 3 中真除法为默认语义，此处为无操作【注意】遗留写法

import torch  # PyTorch 主模块：torch.device / torch.load 用于权重加载
import torch.nn as nn  # 神经网络模块：nn.Module、nn.Sequential、Conv2d、BN 等
import torch.nn.functional as F  # 函数式接口：F.interpolate 双线性上采样

from .segbase import SegBaseModel  # 导入分割模型公共基类，FCN 继承它以复用骨干加载与前向

__all__ = ['FCN', 'get_fcn', 'get_fcn_resnet50_voc',  # 模块公开接口：模型类 + 通用工厂
           'get_fcn_resnet101_voc', 'get_fcn_resnet152_voc']  # 以及三个 VOC 预设工厂函数（续行）


class FCN(SegBaseModel):  # FCN(v2) 模型：继承 SegBaseModel，复用 ResNet 骨干与 base_forward 接口
    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数（含背景）
    # backbone：ResNet 骨干类型，默认 'resnet50'（另支持 101/152）
    # aux：是否启用辅助损失分支，默认 True（【注意】与 fcn.py 中默认 False 不同）
    # pretrained_base：是否加载骨干 ImageNet 预训练权重
    # **kwargs：透传给基类与 _FCNHead（如 norm_layer、norm_kwargs）
    def __init__(self, nclass, backbone='resnet50', aux=True, pretrained_base=True, **kwargs):  # 构造：先经基类建骨干，再建主/辅分割头
        super(FCN, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)  # 基类构造：加载 ResNet 骨干（默认 dilated 模式，特征保持 1/8 分辨率）
        self.head = _FCNHead(2048, nclass, **kwargs)  # 主分割头：接 c4（2048 通道）→ 输出 nclass 通道得分
        if aux:  # 启用辅助分支时
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)  # 辅助分割头：接 c3（1024 通道）→ 输出 nclass 通道得分

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])  # 登记"专属层"名单（新增可训练层），训练脚本据此单独设置学习率

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像张量 (N, 3, H, W)
    # 返回：元组 (主输出,) 或 (主输出, 辅助输出)，空间尺寸均为 (N, nclass, H, W)
    def forward(self, x):  # 前向：骨干特征 → 主/辅分割头 → 双线性上采样回原图
        size = x.size()[2:]  # 记录输入空间尺寸 (H, W)，供上采样对齐
        _, _, c3, c4 = self.base_forward(x)  # 骨干前向：取 c3 (N, 1024, H/8, W/8) 与 c4 (N, 2048, H/8, W/8)，忽略 c1/c2

        outputs = []  # 输出收集列表：主输出在前、辅助输出在后
        x = self.head(c4)  # 主头得分 → (N, nclass, H/8, W/8)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 8 倍双线性上采样 → (N, nclass, H, W)
        outputs.append(x)  # 主输出加入列表
        if self.aux:  # 若启用辅助分支
            auxout = self.auxlayer(c3)  # 辅助头得分 → (N, nclass, H/8, W/8)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)  # 上采样回原尺寸 → (N, nclass, H, W)
            outputs.append(auxout)  # 辅助输出加入列表
        return tuple(outputs)  # 返回输出元组


class _FCNHead(nn.Module):  # FCN 分割头：3x3 卷积降维 + BN + ReLU + Dropout + 1x1 投影到类别数
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：输入通道数（主头 2048 / 辅助头 1024）
    # channels：输出通道数（= nclass）
    # norm_layer：归一化层类型，默认 nn.BatchNorm2d
    # norm_kwargs：传给归一化层的额外参数字典（如 SyncBN 的参数），None 表示无
    # **kwargs：兼容上层透传的其他参数（此处不使用）
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):  # 构造分割头
        super(_FCNHead, self).__init__()  # 调用父类构造函数
        inter_channels = in_channels // 4  # 瓶颈中间通道数 = 输入/4（2048→512，1024→256）
        self.block = nn.Sequential(  # 顺序组合五层
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),  # 3x3 卷积 padding=1 保持分辨率，in_channels→inter_channels，无偏置（后接 BN）
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),  # 归一化层：norm_kwargs 为 None 时展开空字典，否则展开其参数
            nn.ReLU(True),  # 就地 ReLU 激活（inplace=True 节省内存）
            nn.Dropout(0.1),  # Dropout p=0.1，防止过拟合
            nn.Conv2d(inter_channels, channels, 1)  # 1x1 卷积投影到类别数，生成得分图
        )  # 分割头 Sequential 块结束

    # ----------------------- forward 参数说明 -----------------------
    # x：输入特征 (N, in_channels, H', W')；输出 (N, channels, H', W')
    def forward(self, x):  # 前向：直接过 block，通道 in_channels→channels，空间分辨率不变
        return self.block(x)  # 返回分割得分图 (N, channels, H', W')


# ----------------------- get_fcn 参数说明 -----------------------
# dataset：数据集名称（决定类别数），如 'pascal_voc' / 'ade20k' 等
# backbone：骨干类型，默认 'resnet50'；pretrained：是否加载本仓库训练好的分割权重
# root：权重缓存目录，默认 '~/.torch/models'；pretrained_base：是否加载骨干 ImageNet 预训练
# **kwargs：透传给 FCN（如 aux、norm_layer、local_rank 等）
def get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',  # FCN 通用工厂函数
            pretrained_base=True, **kwargs):  # 续行：默认加载骨干 ImageNet 预训练权重
    acronyms = {  # 数据集名 → 权重文件命名缩写映射表
        'pascal_voc': 'pascal_voc',  # Pascal VOC（原生标注）
        'pascal_aug': 'pascal_aug',  # Pascal VOC 增强标注（SBD）
        'ade20k': 'ade',  # ADE20K 缩写为 ade
        'coco': 'coco',  # COCO
        'citys': 'citys',  # Cityscapes
    }  # 数据集缩写映射表结束
    from ..data.dataloader import datasets  # 局部导入数据集注册表（避免循环依赖）
    model = FCN(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数实例化 FCN
    if pretrained:  # 需要加载分割预训练权重时
        from .model_store import get_model_file  # 局部导入权重文件定位工具
        device = torch.device(kwargs['local_rank'])  # 【注意】可疑写法：直接取 kwargs['local_rank']，未传该键会 KeyError；int 会被当作 CUDA 设备序号，纯 CPU 环境会出错
        model.load_state_dict(torch.load(get_model_file('fcn_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 定位权重文件（如 fcn_resnet50_pascal_voc.pth）并加载
                              map_location=device))  # 权重张量映射到指定设备后载入模型
    return model  # 返回构建（并可选加载权重）的模型


# ----------------------- 函数参数说明 -----------------------
# **kwargs：透传给 get_fcn（可含 pretrained、local_rank 等）
def get_fcn_resnet50_voc(**kwargs):  # VOC 预设工厂：FCN + ResNet50 + Pascal VOC
    return get_fcn('pascal_voc', 'resnet50', **kwargs)  # 固定数据集与骨干，调用通用工厂


# ----------------------- 函数参数说明 -----------------------
# **kwargs：透传给 get_fcn
def get_fcn_resnet101_voc(**kwargs):  # VOC 预设工厂：FCN + ResNet101 + Pascal VOC
    return get_fcn('pascal_voc', 'resnet101', **kwargs)  # 固定数据集与骨干，调用通用工厂


# ----------------------- 函数参数说明 -----------------------
# **kwargs：透传给 get_fcn
def get_fcn_resnet152_voc(**kwargs):  # VOC 预设工厂：FCN + ResNet152 + Pascal VOC
    return get_fcn('pascal_voc', 'resnet152', **kwargs)  # 固定数据集与骨干，调用通用工厂
