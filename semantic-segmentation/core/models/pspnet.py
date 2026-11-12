# ============================================================================
# 【文件说明】pspnet.py —— 实现 PSPNet（Pyramid Scene Parsing Network，金字塔场景
#   解析网络）语义分割模型：包含主模型类 PSPNet、金字塔池化模块 _PyramidPooling、
#   分割头 _PSPHead、1×1 卷积组合块 _PSP1x1Conv，以及 get_psp 系列快捷构建函数。
# 【核心思想】在空洞 ResNet 骨干（output stride=8）末级特征 c4 之后接金字塔池化模块
#   （PPM）：并行执行 1×1、2×2、3×3、6×6 四种尺度的自适应平均池化，分别提取从
#   全局到局部的多尺度场景上下文；每个池化分支经 1×1 卷积降维（2048→512）后双线性
#   上采样回特征图原尺寸，再与骨干原始特征按通道拼接（2048+4×512=4096），最后经
#   3×3 卷积融合 + 1×1 分类器输出逐像素类别预测，从而缓解复杂场景中的上下文依赖、
#   类别混淆与显式关系推断问题。
# 【论文名称】Zhao, Hengshuang, et al. "Pyramid Scene Parsing Network." CVPR 2017.
# 【注意】类 docstring 中的 :class:`nn.BatchNorm` 为笔误（实际类名为 nn.BatchNorm2d），
#   按"只加注释不改代码"原则保持原样。
# ============================================================================
"""Pyramid Scene Parsing Network"""  # 模块英文 docstring（原样保留）：金字塔场景解析网络
import torch  # PyTorch 主库：张量运算、设备指定（torch.device）与权重加载（torch.load）
import torch.nn as nn  # 神经网络子模块：nn.Conv2d / nn.BatchNorm2d / nn.Sequential 等常用层
import torch.nn.functional as F  # 函数式接口：F.interpolate 双线性上采样等无参数操作

from .segbase import SegBaseModel  # 分割模型基类：按 backbone 构建空洞 ResNet 并提供 base_forward 特征提取
from .fcn import _FCNHead  # FCN 辅助分类头：对中间层 c3 计算辅助损失（深度监督）

# __all__：声明本模块对外导出的符号（模型类 + 各数据集/骨干组合的快捷构建函数）
__all__ = ['PSPNet', 'get_psp', 'get_psp_resnet50_voc', 'get_psp_resnet50_ade', 'get_psp_resnet101_voc',
           'get_psp_resnet101_ade', 'get_psp_resnet101_citys', 'get_psp_resnet101_coco']  # __all__ 续行并结束：ADE/Cityscapes/COCO 组合的快捷构建函数名


class PSPNet(SegBaseModel):  # PSPNet 模型类：继承 SegBaseModel，复用骨干网络构建与通用前向逻辑
    # 类 docstring：PSPNet 模型说明与参数（原英文保留）
    r"""Pyramid Scene Parsing Network

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
        Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia.
        "Pyramid scene parsing network." *CVPR*, 2017
    """
    # 【__init__ 参数说明】
    #   nclass：分割类别数（不含背景，由数据集决定）
    #   backbone：骨干网络类型，默认 'resnet50'（可选 resnet50 / resnet101 / resnet152）
    #   aux：是否启用辅助损失（对 c3 层接辅助头做深度监督）
    #   pretrained_base：是否加载 ImageNet 预训练骨干权重
    #   **kwargs：透传给基类与头部的额外参数（如 norm_layer、norm_kwargs）
    def __init__(self, nclass, backbone='resnet50', aux=False, pretrained_base=True, **kwargs):  # 模型构造函数
        super(PSPNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)  # 调用基类构造：初始化骨干网络并记录 aux / nclass
        self.head = _PSPHead(nclass, **kwargs)  # 主分割头：PPM + 卷积分类，将 c4 映射为 (N, nclass, H/8, W/8)
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


# 【函数说明】_PSP1x1Conv —— 构造 "1×1 卷积 + 归一化 + ReLU" 组合层，供 PPM 各分支降维
#   参数：in_channels 输入通道；out_channels 输出通道；norm_layer 归一化层类型；norm_kwargs 归一化层额外参数
def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):  # 返回一个顺序组合模块
    return nn.Sequential(  # 按顺序堆叠三层
        nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1×1 卷积降维（如 2048→512）；后接 BN 故关闭 bias
        norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),  # 归一化层；norm_kwargs 为 None 时展开为空参数
        nn.ReLU(True)  # ReLU 激活（inplace=True 节省显存）
    )  # “1×1 卷积 + BN + ReLU”组合层 Sequential 结束


class _PyramidPooling(nn.Module):  # 金字塔池化模块 PPM：四尺度自适应平均池化并行聚合多尺度上下文
    def __init__(self, in_channels, **kwargs):  # in_channels：输入特征通道数（ResNet stage4 输出为 2048）
        super(_PyramidPooling, self).__init__()  # 调用 nn.Module 基类构造
        out_channels = int(in_channels / 4)  # 每个池化分支降维后的通道数：2048/4 = 512
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)  # 尺度1：1×1 全局平均池化 → (N,C,1,1)，捕获图像级全局上下文
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)  # 尺度2：2×2 池化 → (N,C,2,2)，捕获较粗子区域上下文
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)  # 尺度3：3×3 池化 → (N,C,3,3)，中等尺度上下文
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)  # 尺度4：6×6 池化 → (N,C,6,6)，捕获较细粒度局部上下文
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)  # 分支1 卷积块（对应 avgpool1）：2048→512
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)  # 分支2 卷积块（对应 avgpool2）：2048→512
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)  # 分支3 卷积块（对应 avgpool3）：2048→512
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)  # 分支4 卷积块（对应 avgpool4）：2048→512

    def forward(self, x):  # 前向：x 为骨干输出特征 (N, 2048, H/8, W/8)
        size = x.size()[2:]  # 记录特征图空间尺寸 (H/8, W/8)，作为各分支上采样的目标尺寸
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)  # 分支1：全局池化→1×1卷积降维→上采样回 (H/8, W/8)，通道 2048→512
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)  # 分支2：2×2 池化→降维→上采样回 (H/8, W/8)，通道 2048→512
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)  # 分支3：3×3 池化→降维→上采样回 (H/8, W/8)，通道 2048→512
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)  # 分支4：6×6 池化→降维→上采样回 (H/8, W/8)，通道 2048→512
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)  # 原始特征 + 4 个分支按通道拼接：(N, 2048+4×512=4096, H/8, W/8)


class _PSPHead(nn.Module):  # PSPNet 主分割头：先 PPM 上下文聚合，再卷积块输出类别得分
    # 参数：nclass 类别数；norm_layer 归一化层（默认 BN2d）；norm_kwargs 归一化层额外参数
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_PSPHead, self).__init__()  # 调用基类构造
        self.psp = _PyramidPooling(2048, norm_layer=norm_layer, norm_kwargs=norm_kwargs)  # 金字塔池化模块：【注意】输入通道 2048 为硬编码，仅适配 ResNet 系骨干
        self.block = nn.Sequential(  # PPM 之后的分类卷积块
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),  # 3×3 卷积融合多尺度上下文：4096→512，padding=1 保持尺寸
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),  # 归一化层（512 通道）
            nn.ReLU(True),  # ReLU 激活（inplace）
            nn.Dropout(0.1),  # Dropout，丢弃率 0.1，防止过拟合
            nn.Conv2d(512, nclass, 1)  # 1×1 分类卷积：512→nclass，输出逐像素类别 logits
        )  # 分类卷积块 Sequential 结束

    def forward(self, x):  # 前向：x 为骨干 c4 特征 (N, 2048, H/8, W/8)
        x = self.psp(x)  # 金字塔池化：(N, 2048, H/8, W/8) → (N, 4096, H/8, W/8)
        return self.block(x)  # 分类卷积块：(N, 4096, H/8, W/8) → (N, nclass, H/8, W/8)


# 【工厂函数】get_psp —— 按数据集与骨干网络组合构建 PSPNet，可选加载预训练权重
#   参数：dataset 数据集名称；backbone 骨干类型；pretrained 是否加载完整模型预训练权重（True/版本字符串）；
#         root 权重文件存放根目录；pretrained_base 是否加载 ImageNet 预训练骨干；**kwargs 透传给 PSPNet
def get_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',  # 默认 VOC + ResNet50，不加载完整模型权重
            pretrained_base=True, **kwargs):  # 默认加载 ImageNet 预训练骨干
    # 函数 docstring：get_psp 参数说明与用法示例（原英文保留）
    r"""Pyramid Scene Parsing Network

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    pretrained_base : bool or str, default True
        This will load pretrained backbone network, that was trained on ImageNet.
    Examples
    --------
    >>> model = get_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {  # 数据集名 → 预训练权重文件名中使用的缩写映射
        'pascal_voc': 'pascal_voc',  # PASCAL VOC
        'pascal_aug': 'pascal_aug',  # PASCAL VOC 增广版
        'ade20k': 'ade',  # ADE20K 缩写为 ade
        'coco': 'coco',  # COCO
        'citys': 'citys',  # Cityscapes
    }  # 数据集缩写映射字典结束
    from ..data.dataloader import datasets  # 延迟导入数据集注册表（避免循环导入），用于查询各类别数 NUM_CLASS
    model = PSPNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建 PSPNet
    if pretrained:  # 需要加载完整模型预训练权重时
        from .model_store import get_model_file  # 延迟导入：按模型名定位（必要时下载）权重文件
        device = torch.device(kwargs['local_rank'])  # 【注意】可疑写法：直接索引 kwargs['local_rank']，未传该键会 KeyError；且 local_rank 通常为 int，torch.device(int) 会抛 TypeError，疑似应为 torch.device('cuda', kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('psp_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 拼出权重文件名 psp_{backbone}_{dataset缩写} 并加载
                              map_location=device))  # map_location：将权重张量映射到 local_rank 指定的设备
    return model  # 返回构建（并可能已加载权重）的模型


def get_psp_resnet50_voc(**kwargs):  # 快捷构建：PSPNet + ResNet50 + PASCAL VOC
    return get_psp('pascal_voc', 'resnet50', **kwargs)  # 委托 get_psp，固定数据集与骨干


def get_psp_resnet50_ade(**kwargs):  # 快捷构建：PSPNet + ResNet50 + ADE20K
    return get_psp('ade20k', 'resnet50', **kwargs)  # 委托 get_psp，固定数据集与骨干


def get_psp_resnet101_voc(**kwargs):  # 快捷构建：PSPNet + ResNet101 + PASCAL VOC
    return get_psp('pascal_voc', 'resnet101', **kwargs)  # 委托 get_psp，固定数据集与骨干


def get_psp_resnet101_ade(**kwargs):  # 快捷构建：PSPNet + ResNet101 + ADE20K
    return get_psp('ade20k', 'resnet101', **kwargs)  # 委托 get_psp，固定数据集与骨干


def get_psp_resnet101_citys(**kwargs):  # 快捷构建：PSPNet + ResNet101 + Cityscapes
    return get_psp('citys', 'resnet101', **kwargs)  # 委托 get_psp，固定数据集与骨干


def get_psp_resnet101_coco(**kwargs):  # 快捷构建：PSPNet + ResNet101 + COCO
    return get_psp('coco', 'resnet101', **kwargs)  # 委托 get_psp，固定数据集与骨干


if __name__ == '__main__':  # 直接运行本文件时的简单自测入口
    model = get_psp_resnet50_voc()  # 构建 PSPNet（ResNet50 + VOC，21 类）
    img = torch.randn(4, 3, 480, 480)  # 构造随机输入：batch=4，RGB 3 通道，480×480
    output = model(img)  # 前向推理，输出 (4, 21, 480, 480)
