# ============================================================================
# 【文件说明】deeplabv3.py —— 实现 DeepLabV3 语义分割模型：包含主模型类 DeepLabV3、
#   分割头 _DeepLabHead、空洞空间金字塔池化模块 _ASPP、单个空洞卷积分支 _ASPPConv、
#   图像级池化分支 _AsppPooling，以及 get_deeplabv3 系列快捷构建函数。
#   【注意】本文件首行英文 docstring 误写为 "Pyramid Scene Parsing Network"（PSPNet
#   的描述，疑似复制粘贴遗留），按"只加注释不改代码"原则保持原样。
# 【核心思想】用"并行空洞卷积"替代 DeepLabV2 的级联空洞卷积：对空洞 ResNet 骨干
#   （output stride=8）末级特征 c4 并行执行 5 个分支——1×1 卷积、三个不同空洞率
#   （12/24/36）的 3×3 空洞卷积、全局平均池化（图像级特征），捕获多尺度上下文；
#   五分支各输出 256 通道，拼接（5×256=1280）后经 1×1 卷积融合投影回 256 通道
#   （附 Dropout 0.5），再接 3×3 卷积 + 1×1 分类器输出逐像素类别预测。
# 【论文名称】Chen, Liang-Chieh, et al. "Rethinking Atrous Convolution for Semantic
#   Image Segmentation." arXiv preprint arXiv:1706.05587 (2017).
# ============================================================================
"""Pyramid Scene Parsing Network"""  # 模块英文 docstring（原样保留）：【注意】内容为 PSPNet 描述，与 DeepLabV3 不符，疑似复制粘贴错误
import torch  # PyTorch 主库：张量运算、设备指定（torch.device）与权重加载（torch.load）
import torch.nn as nn  # 神经网络子模块：nn.Conv2d / nn.BatchNorm2d / nn.Sequential 等常用层
import torch.nn.functional as F  # 函数式接口：F.interpolate 双线性上采样等无参数操作

from .segbase import SegBaseModel  # 分割模型基类：按 backbone 构建空洞 ResNet 并提供 base_forward 特征提取
from .fcn import _FCNHead  # FCN 辅助分类头：对中间层 c3 计算辅助损失（深度监督）

# __all__：声明本模块对外导出的符号（模型类 + 各数据集/骨干组合的快捷构建函数）
__all__ = ['DeepLabV3', 'get_deeplabv3', 'get_deeplabv3_resnet50_voc', 'get_deeplabv3_resnet101_voc',
           'get_deeplabv3_resnet152_voc', 'get_deeplabv3_resnet50_ade', 'get_deeplabv3_resnet101_ade',  # __all__ 续行：VOC / ADE20K 各骨干组合的快捷构建函数名
           'get_deeplabv3_resnet152_ade']  # __all__ 列表结束


class DeepLabV3(SegBaseModel):  # DeepLabV3 模型类：继承 SegBaseModel，复用骨干网络构建与通用前向逻辑
    # 类 docstring：DeepLabV3 模型说明与参数（原英文保留）
    r"""DeepLabV3

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).
    """
    # 【__init__ 参数说明】
    #   nclass：分割类别数（不含背景，由数据集决定）
    #   backbone：骨干网络类型，默认 'resnet50'（可选 resnet50 / resnet101 / resnet152）
    #   aux：是否启用辅助损失（对 c3 层接辅助头做深度监督）
    #   pretrained_base：是否加载 ImageNet 预训练骨干权重
    #   **kwargs：透传给基类与头部的额外参数（如 norm_layer、norm_kwargs）
    def __init__(self, nclass, backbone='resnet50', aux=False, pretrained_base=True, **kwargs):  # 模型构造函数
        super(DeepLabV3, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)  # 调用基类构造：初始化骨干网络并记录 aux / nclass
        self.head = _DeepLabHead(nclass, **kwargs)  # 主分割头：ASPP + 卷积分类，将 c4 映射为 (N, nclass, H/8, W/8)
        if self.aux:  # 若启用辅助损失
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)  # 辅助头：输入 ResNet stage3 特征 c3（1024 通道；通道数硬编码，仅适配 ResNet 系骨干）

        # exclusive：登记本模型新增（非骨干自带）的子模块，供权重加载逻辑区分骨干与新增部分
        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):  # 前向传播：x 为输入图像张量 (N, 3, H, W)
        size = x.size()[2:]  # 记录输入空间尺寸 (H, W)，用于最后上采样恢复原分辨率
        _, _, c3, c4 = self.base_forward(x)  # 骨干前向：c3=(N, 1024, H/8, W/8)，c4=(N, 2048, H/8, W/8)（dilated 模式下 output stride=8）
        outputs = []  # 输出收集列表：主输出在前，可选辅助输出在后
        x = self.head(c4)  # 主分割头：(N, 2048, H/8, W/8) → (N, nclass, H/8, W/8)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 双线性上采样 8 倍：(N, nclass, H/8, W/8) → (N, nclass, H, W)
        outputs.append(x)  # 主输出加入列表

        if self.aux:  # 若启用辅助损失
            auxout = self.auxlayer(c3)  # 辅助头：(N, 1024, H/8, W/8) → (N, nclass, H/8, W/8)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)  # 辅助输出同样上采样回原图分辨率：(N, nclass, H, W)
            outputs.append(auxout)  # 辅助输出加入列表
        return tuple(outputs)  # 返回 (主输出,) 或 (主输出, 辅助输出) 元组，供损失函数按序加权计算


class _DeepLabHead(nn.Module):  # DeepLabV3 分割头：ASPP 多尺度上下文聚合 + 卷积分类块
    # 参数：nclass 类别数；norm_layer 归一化层（默认 BN2d）；norm_kwargs 归一化层额外参数
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DeepLabHead, self).__init__()  # 调用 nn.Module 基类构造
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)  # ASPP 模块：输入 2048 通道，空洞率 12/24/36（对应 output stride=8），输出 256 通道；【注意】通道数硬编码，仅适配 ResNet 系骨干
        self.block = nn.Sequential(  # ASPP 之后的分类卷积块
            nn.Conv2d(256, 256, 3, padding=1, bias=False),  # 3×3 卷积细化特征：256→256，padding=1 保持尺寸
            norm_layer(256, **({} if norm_kwargs is None else norm_kwargs)),  # 归一化层（256 通道）
            nn.ReLU(True),  # ReLU 激活（inplace）
            nn.Dropout(0.1),  # Dropout，丢弃率 0.1，防止过拟合
            nn.Conv2d(256, nclass, 1)  # 1×1 分类卷积：256→nclass，输出逐像素类别 logits
        )  # 分类卷积块 Sequential 结束

    def forward(self, x):  # 前向：x 为骨干 c4 特征 (N, 2048, H/8, W/8)
        x = self.aspp(x)  # ASPP 聚合多尺度上下文：(N, 2048, H/8, W/8) → (N, 256, H/8, W/8)
        return self.block(x)  # 分类卷积块：(N, 256, H/8, W/8) → (N, nclass, H/8, W/8)


class _ASPPConv(nn.Module):  # ASPP 的单个空洞卷积分支：3×3 空洞卷积 + BN + ReLU
    # 参数：in_channels 输入通道；out_channels 输出通道；atrous_rate 空洞率（膨胀率）；norm_layer 归一化层；norm_kwargs 归一化层额外参数
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()  # 调用基类构造
        self.block = nn.Sequential(  # 顺序组合三层
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),  # 3×3 空洞卷积：padding=空洞率以保持输出尺寸不变，等效感受野扩大为 (2×rate+1)² 的核
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),  # 归一化层；norm_kwargs 为 None 时展开为空参数
            nn.ReLU(True)  # ReLU 激活（inplace）
        )  # 空洞卷积分支 Sequential 块结束

    def forward(self, x):  # 前向：空间尺寸不变，仅通道变化
        return self.block(x)  # (N, in_channels, H, W) → (N, out_channels, H, W)


class _AsppPooling(nn.Module):  # ASPP 的图像级分支：全局平均池化 + 1×1 卷积 + 上采样回原尺寸
    # 参数：in_channels 输入通道；out_channels 输出通道；norm_layer 归一化层；norm_kwargs 归一化层额外参数
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()  # 调用基类构造
        self.gap = nn.Sequential(  # 全局上下文提取块
            nn.AdaptiveAvgPool2d(1),  # 全局自适应平均池化：(N, in, H, W) → (N, in, 1, 1)，捕获图像级全局语义
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1×1 卷积降维：2048→256
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),  # 归一化层（256 通道）
            nn.ReLU(True)  # ReLU 激活（inplace）
        )  # 全局上下文提取 Sequential 块结束

    def forward(self, x):  # 前向：x 为 (N, in_channels, H/8, W/8)
        size = x.size()[2:]  # 记录输入特征图空间尺寸 (H/8, W/8)，作为上采样目标
        pool = self.gap(x)  # 全局池化 + 1×1 卷积：(N, in_channels, H/8, W/8) → (N, out_channels, 1, 1)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)  # 双线性上采样回特征图原尺寸：(N, out_channels, 1, 1) → (N, out_channels, H/8, W/8)
        return out  # 返回恢复空间尺寸后的图像级全局特征


class _ASPP(nn.Module):  # 空洞空间金字塔池化模块（Atrous Spatial Pyramid Pooling）：并行多分支捕获多尺度上下文
    # 参数：in_channels 输入通道（2048）；atrous_rates 空洞率列表（output stride=8 时为 [12,24,36]）；norm_layer 归一化层；norm_kwargs 归一化层额外参数
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()  # 调用基类构造
        out_channels = 256  # 每个分支的输出通道数（论文设定）
        self.b0 = nn.Sequential(  # 分支 0：1×1 卷积（等价于空洞率 0 的轻量分支，捕获逐点上下文）
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1×1 卷积：2048→256
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),  # 归一化层（256 通道）
            nn.ReLU(True)  # ReLU 激活（inplace）
        )  # 1×1 卷积分支 b0 Sequential 块结束

        rate1, rate2, rate3 = tuple(atrous_rates)  # 解包三个空洞率：12、24、36
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)  # 分支 1：空洞率 12 的 3×3 空洞卷积
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)  # 分支 2：空洞率 24 的 3×3 空洞卷积
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)  # 分支 3：空洞率 36 的 3×3 空洞卷积
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)  # 分支 4：全局平均池化（图像级特征）

        self.project = nn.Sequential(  # 分支融合投影块
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),  # 1×1 卷积融合拼接特征：5×256=1280→256
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),  # 归一化层（256 通道）
            nn.ReLU(True),  # ReLU 激活（inplace）
            nn.Dropout(0.5)  # Dropout，丢弃率 0.5（论文设定，防止过拟合）
        )  # 融合投影 Sequential 块结束

    def forward(self, x):  # 前向：x 为骨干 c4 特征 (N, 2048, H/8, W/8)
        feat1 = self.b0(x)  # 1×1 卷积分支：(N, 2048, H/8, W/8) → (N, 256, H/8, W/8)
        feat2 = self.b1(x)  # 空洞率 12 分支：(N, 2048, H/8, W/8) → (N, 256, H/8, W/8)
        feat3 = self.b2(x)  # 空洞率 24 分支：(N, 2048, H/8, W/8) → (N, 256, H/8, W/8)
        feat4 = self.b3(x)  # 空洞率 36 分支：(N, 2048, H/8, W/8) → (N, 256, H/8, W/8)
        feat5 = self.b4(x)  # 全局池化分支：(N, 2048, H/8, W/8) → (N, 256, H/8, W/8)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)  # 五个分支按通道拼接：(N, 5×256=1280, H/8, W/8)
        x = self.project(x)  # 融合投影：(N, 1280, H/8, W/8) → (N, 256, H/8, W/8)
        return x  # 返回 ASPP 输出


# 【工厂函数】get_deeplabv3 —— 按数据集与骨干网络组合构建 DeepLabV3，可选加载预训练权重
#   参数：dataset 数据集名称；backbone 骨干类型；pretrained 是否加载完整模型预训练权重；
#         root 权重文件存放根目录；pretrained_base 是否加载 ImageNet 预训练骨干；**kwargs 透传给 DeepLabV3
def get_deeplabv3(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',  # 默认 VOC + ResNet50，不加载完整模型权重
                  pretrained_base=True, **kwargs):  # 默认加载 ImageNet 预训练骨干
    acronyms = {  # 数据集名 → 预训练权重文件名中使用的缩写映射
        'pascal_voc': 'pascal_voc',  # PASCAL VOC
        'pascal_aug': 'pascal_aug',  # PASCAL VOC 增广版
        'ade20k': 'ade',  # ADE20K 缩写为 ade
        'coco': 'coco',  # COCO
        'citys': 'citys',  # Cityscapes
    }  # 数据集缩写映射字典结束
    from ..data.dataloader import datasets  # 延迟导入数据集注册表（避免循环导入），用于查询各类别数 NUM_CLASS
    model = DeepLabV3(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建 DeepLabV3
    if pretrained:  # 需要加载完整模型预训练权重时
        from .model_store import get_model_file  # 延迟导入：按模型名定位（必要时下载）权重文件
        device = torch.device(kwargs['local_rank'])  # 【注意】可疑写法：直接索引 kwargs['local_rank']，未传该键会 KeyError；且 local_rank 通常为 int，torch.device(int) 会抛 TypeError，疑似应为 torch.device('cuda', kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('deeplabv3_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 拼出权重文件名 deeplabv3_{backbone}_{dataset缩写} 并加载
                              map_location=device))  # map_location：将权重张量映射到 local_rank 指定的设备
    return model  # 返回构建（并可能已加载权重）的模型


def get_deeplabv3_resnet50_voc(**kwargs):  # 快捷构建：DeepLabV3 + ResNet50 + PASCAL VOC
    return get_deeplabv3('pascal_voc', 'resnet50', **kwargs)  # 委托 get_deeplabv3，固定数据集与骨干


def get_deeplabv3_resnet101_voc(**kwargs):  # 快捷构建：DeepLabV3 + ResNet101 + PASCAL VOC
    return get_deeplabv3('pascal_voc', 'resnet101', **kwargs)  # 委托 get_deeplabv3，固定数据集与骨干


def get_deeplabv3_resnet152_voc(**kwargs):  # 快捷构建：DeepLabV3 + ResNet152 + PASCAL VOC
    return get_deeplabv3('pascal_voc', 'resnet152', **kwargs)  # 委托 get_deeplabv3，固定数据集与骨干


def get_deeplabv3_resnet50_ade(**kwargs):  # 快捷构建：DeepLabV3 + ResNet50 + ADE20K
    return get_deeplabv3('ade20k', 'resnet50', **kwargs)  # 委托 get_deeplabv3，固定数据集与骨干


def get_deeplabv3_resnet101_ade(**kwargs):  # 快捷构建：DeepLabV3 + ResNet101 + ADE20K
    return get_deeplabv3('ade20k', 'resnet101', **kwargs)  # 委托 get_deeplabv3，固定数据集与骨干


def get_deeplabv3_resnet152_ade(**kwargs):  # 快捷构建：DeepLabV3 + ResNet152 + ADE20K
    return get_deeplabv3('ade20k', 'resnet152', **kwargs)  # 委托 get_deeplabv3，固定数据集与骨干


if __name__ == '__main__':  # 直接运行本文件时的简单自测入口
    model = get_deeplabv3_resnet50_voc()  # 构建 DeepLabV3（ResNet50 + VOC，21 类）
    img = torch.randn(2, 3, 480, 480)  # 构造随机输入：batch=2，RGB 3 通道，480×480
    output = model(img)  # 前向推理，输出 (2, 21, 480, 480)
