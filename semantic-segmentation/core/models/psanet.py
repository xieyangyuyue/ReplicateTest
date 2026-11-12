# ============================================================================
# 【文件说明】psanet.py —— 实现 PSANet（Point-wise Spatial Attention Network，点式
#   空间注意力网络）语义分割模型：包含主模型类 PSANet、注意力头 _PSAHead、点式空间
#   注意力模块 _PointwiseSpatialAttention、注意力生成模块 _AttentionGeneration，
#   以及 get_psanet 系列快捷构建函数。
# 【核心思想】不采用 PSPNet 那样固定形状的池化核，而是为任意两个空间位置自适应地
#   学习注意力：collect（收集）分支把全图所有位置的信息加权聚合到当前位置，
#   distribute（分发）分支把当前位置的信息加权传播给全图位置。注意力权重由 1×1
#   卷积从降维特征（2048→512）生成（输出 3600 个通道对应特征图上 60×60=3600 个
#   源位置），经 softmax 归一化后与特征做批量矩阵乘（bmm）完成信息聚合；两分支
#   输出拼接（2×512=1024）后升维至 2048，再与骨干原始特征拼接（4096）并投影分类。
# 【论文名称】Hengshuang Zhao, et al. "PSANet: Point-wise Spatial Attention Network
#   for Scene Parsing." ECCV 2018.
# ============================================================================
"""Point-wise Spatial Attention Network"""  # 模块英文 docstring（原样保留）：点式空间注意力网络
import torch  # PyTorch 主库：张量运算（torch.bmm / torch.softmax）、设备指定与权重加载
import torch.nn as nn  # 神经网络子模块：nn.Conv2d / nn.BatchNorm2d / nn.Sequential 等常用层
import torch.nn.functional as F  # 函数式接口：F.interpolate 双线性上采样等无参数操作

from core.nn import _ConvBNReLU  # 卷积+BN+ReLU 三合一组合层；【注意】此处为绝对导入（其余模型文件多用相对导入），要求以仓库根目录为运行路径
from core.models.segbase import SegBaseModel  # 分割模型基类（绝对导入）：构建空洞 ResNet 并提供 base_forward
from core.models.fcn import _FCNHead  # FCN 辅助分类头（绝对导入）：对中间层 c3 计算辅助损失（深度监督）

# __all__：声明本模块对外导出的符号（模型类 + 各数据集/骨干组合的快捷构建函数）
__all__ = ['PSANet', 'get_psanet', 'get_psanet_resnet50_voc', 'get_psanet_resnet101_voc',
           'get_psanet_resnet152_voc', 'get_psanet_resnet50_citys', 'get_psanet_resnet101_citys',  # __all__ 续行：Cityscapes 各骨干组合的快捷构建函数名
           'get_psanet_resnet152_citys']  # __all__ 列表结束


class PSANet(SegBaseModel):  # PSANet 模型类：继承 SegBaseModel，复用骨干网络构建与通用前向逻辑
    # 类 docstring：PSANet 模型说明与参数（原英文保留）
    r"""PSANet

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
        Hengshuang Zhao, et al. "PSANet: Point-wise Spatial Attention Network for Scene Parsing."
        ECCV-2018.
    """
    # 【__init__ 参数说明】
    #   nclass：分割类别数（不含背景，由数据集决定）
    #   backbone：骨干网络类型；【注意】默认值 'resnet' 不是合法骨干名（基类仅识别 resnet50/101/152），疑似应为 'resnet50'
    #   aux：是否启用辅助损失（对 c3 层接辅助头做深度监督）
    #   pretrained_base：是否加载 ImageNet 预训练骨干权重
    #   **kwargs：透传给基类与头部的额外参数（如 norm_layer）
    def __init__(self, nclass, backbone='resnet', aux=False, pretrained_base=True, **kwargs):  # 模型构造函数
        super(PSANet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)  # 调用基类构造：初始化骨干网络并记录 aux / nclass
        self.head = _PSAHead(nclass, **kwargs)  # 主分割头：点式空间注意力 + 投影分类，将 c4 映射为 (N, nclass, H/8, W/8)
        if aux:  # 若启用辅助损失
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)  # 辅助头：输入 ResNet stage3 特征 c3（1024 通道；通道数硬编码，仅适配 ResNet 系骨干）

        # exclusive：登记本模型新增（非骨干自带）的子模块，供权重加载逻辑区分骨干与新增部分
        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x):  # 前向传播：x 为输入图像张量 (N, 3, H, W)
        size = x.size()[2:]  # 记录输入空间尺寸 (H, W)，用于最后上采样恢复原分辨率
        _, _, c3, c4 = self.base_forward(x)  # 骨干前向：c3=(N, 1024, H/8, W/8)，c4=(N, 2048, H/8, W/8)（dilated 模式下 output stride=8）
        outputs = list()  # 输出收集列表：主输出在前，可选辅助输出在后
        x = self.head(c4)  # 主分割头：(N, 2048, H/8, W/8) → (N, nclass, H/8, W/8)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 双线性上采样 8 倍：(N, nclass, H/8, W/8) → (N, nclass, H, W)
        outputs.append(x)  # 主输出加入列表

        if self.aux:  # 若启用辅助损失
            auxout = self.auxlayer(c3)  # 辅助头：(N, 1024, H/8, W/8) → (N, nclass, H/8, W/8)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)  # 辅助输出同样上采样回原图分辨率：(N, nclass, H, W)
            outputs.append(auxout)  # 辅助输出加入列表
        return tuple(outputs)  # 返回 (主输出,) 或 (主输出, 辅助输出) 元组，供损失函数按序加权计算


class _PSAHead(nn.Module):  # PSANet 主分割头：点式空间注意力聚合 + 卷积投影分类
    # 参数：nclass 类别数；norm_layer 归一化层（默认 BN2d）；**kwargs 额外参数
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_PSAHead, self).__init__()  # 调用 nn.Module 基类构造
        # psa_out_channels = crop_size // 8 ** 2  （原英文注释，原样保留）
        # 【注意】原注释表达式因运算符优先级（** 高于 //）并不能算得 3600，其真实意图应为 (crop_size//8)**2 = (480/8)² = 3600
        self.psa = _PointwiseSpatialAttention(2048, 3600, norm_layer)  # 点式空间注意力模块：输入 2048 通道；3600 = 60×60 个源位置数，【注意】硬编码要求特征图恰为 60×60（即 480×480 输入、output stride=8），其他输入尺寸会因 bmm 形状不匹配而报错

        self.conv_post = _ConvBNReLU(1024, 2048, 1, norm_layer=norm_layer)  # 1×1 卷积块：将注意力输出 1024→2048，与骨干特征通道对齐以便拼接
        self.project = nn.Sequential(  # 分类投影块
            _ConvBNReLU(4096, 512, 3, padding=1, norm_layer=norm_layer),  # 3×3 卷积：4096→512（2048 骨干 + 2048 注意力特征拼接后），padding=1 保持尺寸
            nn.Dropout2d(0.1, False),  # 空间 Dropout，丢弃率 0.1，inplace=False
            nn.Conv2d(512, nclass, 1))  # 1×1 分类卷积：512→nclass，输出逐像素类别 logits

    def forward(self, x):  # 前向：x 为骨干 c4 特征 (N, 2048, H/8, W/8)
        global_feature = self.psa(x)  # 点式空间注意力：(N, 2048, H/8, W/8) → (N, 1024, H/8, W/8)（collect/distribute 各 512 通道拼接）
        out = self.conv_post(global_feature)  # 1×1 卷积升维：(N, 1024, H/8, W/8) → (N, 2048, H/8, W/8)
        out = torch.cat([x, out], dim=1)  # 与骨干原始特征按通道拼接：(N, 2048+2048=4096, H/8, W/8)
        out = self.project(out)  # 投影分类：(N, 4096, H/8, W/8) → (N, nclass, H/8, W/8)

        return out  # 返回主头输出（低分辨率 logits，由模型 forward 上采样回原图）


class _PointwiseSpatialAttention(nn.Module):  # 点式空间注意力模块：collect（收集）与 distribute（分发）两个并行注意力分支
    # 参数：in_channels 输入通道（2048）；out_channels 注意力输出通道（3600，等于特征图位置数 h*w）；norm_layer 归一化层
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_PointwiseSpatialAttention, self).__init__()  # 调用基类构造
        reduced_channels = 512  # 特征压缩后的通道数（注意力计算在 512 维空间进行，降低计算量）
        self.collect_attention = _AttentionGeneration(in_channels, reduced_channels, out_channels, norm_layer)  # 收集分支：为每个位置学习"从全图收集信息"的注意力权重
        self.distribute_attention = _AttentionGeneration(in_channels, reduced_channels, out_channels, norm_layer)  # 分发分支：为每个位置学习"向全图分发信息"的注意力权重

    def forward(self, x):  # 前向：x 为 (N, 2048, H/8, W/8)
        collect_fm = self.collect_attention(x)  # 收集分支输出：(N, 512, H/8, W/8)
        distribute_fm = self.distribute_attention(x)  # 分发分支输出：(N, 512, H/8, W/8)
        psa_fm = torch.cat([collect_fm, distribute_fm], dim=1)  # 两分支按通道拼接：(N, 1024, H/8, W/8)
        return psa_fm  # 返回注意力聚合后的特征


class _AttentionGeneration(nn.Module):  # 单个注意力分支：生成逐对位置注意力权重并用 bmm 完成信息聚合
    # 参数：in_channels 输入通道（2048）；reduced_channels 压缩通道（512）；out_channels 注意力通道（3600=h*w）；norm_layer 归一化层
    def __init__(self, in_channels, reduced_channels, out_channels, norm_layer, **kwargs):
        super(_AttentionGeneration, self).__init__()  # 调用基类构造
        self.conv_reduce = _ConvBNReLU(in_channels, reduced_channels, 1, norm_layer=norm_layer)  # 1×1 卷积块：2048→512 降维，降低注意力计算开销
        self.attention = nn.Sequential(  # 注意力权重生成网络
            _ConvBNReLU(reduced_channels, reduced_channels, 1, norm_layer=norm_layer),  # 1×1 卷积块：512→512
            nn.Conv2d(reduced_channels, out_channels, 1, bias=False))  # 1×1 卷积：512→3600，为每个目标位置生成对全部 h*w 个源位置的（未归一化）注意力权重

        self.reduced_channels = reduced_channels  # 记录压缩通道数，forward 重排形状时使用

    def forward(self, x):  # 前向：x 为 (N, 2048, H/8, W/8)
        reduce_x = self.conv_reduce(x)  # 特征降维：(N, 2048, H/8, W/8) → (N, 512, H/8, W/8)
        attention = self.attention(reduce_x)  # 生成注意力图：(N, 512, H/8, W/8) → (N, 3600, H/8, W/8)
        n, c, h, w = attention.size()  # 读取尺寸：n 批大小，c=3600，h/w 特征图高宽
        attention = attention.view(n, c, -1)  # 展平空间维：(N, 3600, h*w)，第 1 维对应源位置索引
        reduce_x = reduce_x.view(n, self.reduced_channels, -1)  # 展平空间维：(N, 512, h*w)，第 2 维对应源位置索引
        fm = torch.bmm(reduce_x, torch.softmax(attention, dim=1))  # 批量矩阵乘：先对 3600 个源位置的注意力做 softmax 归一化，再加权求和全部位置特征，得 (N, 512, h*w)；【注意】要求 h*w 恰为 3600，否则 bmm 形状不匹配报错
        fm = fm.view(n, self.reduced_channels, h, w)  # 恢复空间形状：(N, 512, h, w)

        return fm  # 返回注意力加权聚合后的特征


# 【工厂函数】get_psanet —— 按数据集与骨干网络组合构建 PSANet，可选加载预训练权重
#   参数：dataset 数据集名称；backbone 骨干类型；pretrained 是否加载完整模型预训练权重；
#         root 权重文件存放根目录；pretrained_base 是否加载 ImageNet 预训练骨干；**kwargs 透传给 PSANet
def get_psanet(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',  # 默认 VOC + ResNet50，不加载完整模型权重
               pretrained_base=True, **kwargs):  # 默认加载 ImageNet 预训练骨干
    acronyms = {  # 数据集名 → 预训练权重文件名中使用的缩写映射
        'pascal_voc': 'pascal_voc',  # PASCAL VOC
        'pascal_aug': 'pascal_aug',  # PASCAL VOC 增广版
        'ade20k': 'ade',  # ADE20K 缩写为 ade
        'coco': 'coco',  # COCO
        'citys': 'citys',  # Cityscapes
    }  # 数据集缩写映射字典结束
    from core.data.dataloader import datasets  # 延迟导入数据集注册表（绝对导入），用于查询各类别数 NUM_CLASS
    model = PSANet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建 PSANet
    if pretrained:  # 需要加载完整模型预训练权重时
        from .model_store import get_model_file  # 延迟导入：按模型名定位（必要时下载）权重文件
        device = torch.device(kwargs['local_rank'])  # 【注意】可疑写法：直接索引 kwargs['local_rank']，未传该键会 KeyError；且 local_rank 通常为 int，torch.device(int) 会抛 TypeError，疑似应为 torch.device('cuda', kwargs['local_rank'])
        # 【注意】疑似复制粘贴 bug：此处权重文件名前缀写成 'deeplabv3_' 而非 'psanet_'，
        # 会去加载 DeepLabV3 的权重并套用于 PSANet（两者结构不同，load_state_dict 将因键不匹配而失败）
        model.load_state_dict(torch.load(get_model_file('deeplabv3_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 拼出权重文件名 deeplabv3_{backbone}_{dataset缩写}（前缀错误，见上）
                              map_location=device))  # map_location：将权重张量映射到 local_rank 指定的设备
    return model  # 返回构建（并可能已加载权重）的模型


def get_psanet_resnet50_voc(**kwargs):  # 快捷构建：PSANet + ResNet50 + PASCAL VOC
    return get_psanet('pascal_voc', 'resnet50', **kwargs)  # 委托 get_psanet，固定数据集与骨干


def get_psanet_resnet101_voc(**kwargs):  # 快捷构建：PSANet + ResNet101 + PASCAL VOC
    return get_psanet('pascal_voc', 'resnet101', **kwargs)  # 委托 get_psanet，固定数据集与骨干


def get_psanet_resnet152_voc(**kwargs):  # 快捷构建：PSANet + ResNet152 + PASCAL VOC
    return get_psanet('pascal_voc', 'resnet152', **kwargs)  # 委托 get_psanet，固定数据集与骨干


def get_psanet_resnet50_citys(**kwargs):  # 快捷构建：PSANet + ResNet50 + Cityscapes
    return get_psanet('citys', 'resnet50', **kwargs)  # 委托 get_psanet，固定数据集与骨干


def get_psanet_resnet101_citys(**kwargs):  # 快捷构建：PSANet + ResNet101 + Cityscapes
    return get_psanet('citys', 'resnet101', **kwargs)  # 委托 get_psanet，固定数据集与骨干


def get_psanet_resnet152_citys(**kwargs):  # 快捷构建：PSANet + ResNet152 + Cityscapes
    return get_psanet('citys', 'resnet152', **kwargs)  # 委托 get_psanet，固定数据集与骨干


if __name__ == '__main__':  # 直接运行本文件时的简单自测入口
    model = get_psanet_resnet50_voc()  # 构建 PSANet（ResNet50 + VOC，21 类）
    img = torch.randn(1, 3, 480, 480)  # 构造随机输入：batch=1，RGB 3 通道，480×480（与注意力模块要求的 60×60 特征图匹配）
    output = model(img)  # 前向推理，输出 (1, 21, 480, 480)
