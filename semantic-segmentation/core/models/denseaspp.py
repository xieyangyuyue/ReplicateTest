# ============================================================================
# 【文件说明】denseaspp.py —— 密集连接空洞卷积金字塔 DenseASPP 分割模型：
#               骨干为 DenseNet 系列（dilated_densenet121/161/169/201），
#               分割头 _DenseASPPHead 为 5 级密集连接的 ASPP 块（空洞率 3/6/12/18/24）。
# 【核心思想】普通 ASPP 的各空洞分支相互独立、并行计算后拼接；DenseASPP 借鉴
#             DenseNet 的密集连接思想：每一级空洞卷积单元（1×1 卷积降到 256 通道
#             → 3×3 空洞卷积降到 64 通道）的输入都是"原始特征 + 此前所有级输出"
#             在通道维的拼接，后级可复用前级不同感受野的特征，在参数量受控的
#             前提下获得更大且更密集的感受野组合；5 级输出（各 64 通道）与原始
#             特征拼接后经 Dropout + 1×1 卷积输出类别 logits。
# 【论文】Maoke Yang, Kunpeng Li, et al. "DenseASPP: Multi-Scale Atrous Spatial
#         Pyramid Pooling..." CVPR 2018
# ============================================================================
import torch  # PyTorch 主库：张量运算、torch.load 等
import torch.nn as nn  # 神经网络模块：Conv2d / BatchNorm2d / Sequential 等
import torch.nn.functional as F  # 函数式接口：interpolate、dropout 等

from .base_models.densenet import *  # 导入膨胀 DenseNet 骨干构造函数（dilated_densenet121/161/169/201 等）
from .fcn import _FCNHead  # FCN 辅助头，用作 aux 分支的分类头

__all__ = ['DenseASPP', 'get_denseaspp', 'get_denseaspp_densenet121_citys',  # 模块对外导出的公开符号列表
           'get_denseaspp_densenet161_citys', 'get_denseaspp_densenet169_citys', 'get_denseaspp_densenet201_citys']  # 其余便捷构造函数名


# DenseASPP 主模型：继承 nn.Module（而非 SegBaseModel，因为骨干是 DenseNet 系列）
class DenseASPP(nn.Module):  # DenseASPP 主模型类
    # ---- __init__ 参数说明 ----
    # nclass: 类别数
    # backbone: DenseNet 骨干类型，默认 'densenet121'
    # aux: 是否启用辅助损失头
    # jpu: 【注意】该参数被接受但从未使用（死参数，保持原样）
    # pretrained_base: 是否加载骨干的 ImageNet 预训练权重
    # dilate_scale: 膨胀倍率（8 或 16），决定骨干输出特征的分辨率（1/8 或 1/16）
    # **kwargs: 透传给骨干构造与辅助头
    def __init__(self, nclass, backbone='densenet121', aux=False, jpu=False,  # 构造函数（续行见下）
                 pretrained_base=True, dilate_scale=8, **kwargs):  # 续行：膨胀倍率默认 8（输出步长 OS=8）
        super(DenseASPP, self).__init__()  # 调用父类构造
        self.nclass = nclass  # 保存类别数
        self.aux = aux  # 保存辅助损失开关
        self.dilate_scale = dilate_scale  # 保存膨胀倍率（骨干最终特征为输入的 1/dilate_scale 分辨率）
        if backbone == 'densenet121':  # 骨干选择：densenet121
            self.pretrained = dilated_densenet121(dilate_scale, pretrained=pretrained_base, **kwargs)  # 膨胀 DenseNet121（最终特征 1024 通道）
        elif backbone == 'densenet161':  # densenet161
            self.pretrained = dilated_densenet161(dilate_scale, pretrained=pretrained_base, **kwargs)  # 膨胀 DenseNet161（2208 通道）
        elif backbone == 'densenet169':  # densenet169
            self.pretrained = dilated_densenet169(dilate_scale, pretrained=pretrained_base, **kwargs)  # 膨胀 DenseNet169（1664 通道）
        elif backbone == 'densenet201':  # densenet201
            self.pretrained = dilated_densenet201(dilate_scale, pretrained=pretrained_base, **kwargs)  # 膨胀 DenseNet201（1920 通道）
        else:  # 未知骨干类型
            raise RuntimeError('unknown backbone: {}'.format(backbone))  # 抛出运行时错误
        in_channels = self.pretrained.num_features  # 骨干最终特征通道数（随 DenseNet 类型不同：1024/2208/1664/1920）

        self.head = _DenseASPPHead(in_channels, nclass)  # DenseASPP 分割头；【注意】未把 kwargs 中的 norm_layer 等传入，自定义归一化配置会被忽略

        if aux:  # 若启用辅助损失
            self.auxlayer = _FCNHead(in_channels, nclass, **kwargs)  # FCN 辅助头（作用于骨干最终特征）

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])  # 独占参数组标记：head（及 auxlayer）训练时可单独设置学习率

    # ---- forward 参数说明 ----
    # x: 输入图像 (N, 3, H, W)
    def forward(self, x):  # 前向：骨干 →（膨胀倍率 16 时上采样回 1/8）→ 头部 → 上采样；可选辅助
        size = x.size()[2:]  # 记录输入空间尺寸 (H, W)
        features = self.pretrained.features(x)  # 调用骨干的 features 属性做前向（DenseNet 风格接口），输出 (N, C, H/s, W/s)
        if self.dilate_scale > 8:  # 若膨胀倍率为 16（特征仅为 1/16 分辨率）
            features = F.interpolate(features, scale_factor=2, mode='bilinear', align_corners=True)  # 双线性上采样 2 倍，回到 1/8 分辨率（与 dilate_scale=8 对齐）
        outputs = []  # 输出收集列表
        x = self.head(features)  # DenseASPP 头 → 主 logits (N, nclass, H/8, W/8)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 双线性上采样回 (N, nclass, H, W)
        outputs.append(x)  # 主输出入列表

        if self.aux:  # 辅助分支
            auxout = self.auxlayer(features)  # FCN 辅助头（与主头共用同一骨干特征）
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)  # 上采样回原尺寸
            outputs.append(auxout)  # 辅助输出入列表
        return tuple(outputs)  # 返回 (主输出,) 或 (主输出, 辅助输出)


# DenseASPP 头：密集 ASPP 块 + Dropout + 1×1 分类卷积
class _DenseASPPHead(nn.Module):  # DenseASPP 分割头：继承 nn.Module
    # ---- __init__ 参数说明 ----
    # in_channels: 骨干输出通道数
    # nclass: 类别数
    # norm_layer / norm_kwargs: 归一化层类型与额外参数（默认 BatchNorm2d / None）
    # **kwargs: 预留参数
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):  # 构造函数
        super(_DenseASPPHead, self).__init__()  # 调用父类构造
        self.dense_aspp_block = _DenseASPPBlock(in_channels, 256, 64, norm_layer, norm_kwargs)  # 密集 ASPP 块：每级 1×1 卷积中间通道 256、空洞卷积输出通道 64
        self.block = nn.Sequential(  # 分类块
            nn.Dropout(0.1),  # Dropout，丢弃概率 0.1
            nn.Conv2d(in_channels + 5 * 64, nclass, 1)  # 1×1 卷积：原始特征 + 5 级 ASPP 输出（各 64 通道）→ nclass
        )  # block 定义结束

    # ---- forward 参数说明 ----
    # x: 骨干特征 (N, in_channels, H, W)
    def forward(self, x):  # 前向：密集 ASPP → 分类
        x = self.dense_aspp_block(x)  # 密集 ASPP：(N, C, H, W) → (N, C + 5*64, H, W)
        return self.block(x)  # 分类 → (N, nclass, H, W)


# DenseASPP 的单级卷积单元：1×1 降维 → BN/ReLU → 3×3 空洞卷积 → BN/ReLU → Dropout
class _DenseASPPConv(nn.Sequential):  # 继承 nn.Sequential（子模块按序自动执行）
    # ---- __init__ 参数说明 ----
    # in_channels: 输入通道（含此前所有密集拼接进来的通道）
    # inter_channels: 1×1 卷积后的中间通道（256）
    # out_channels: 3×3 空洞卷积后的输出通道（64）
    # atrous_rate: 空洞率
    # drop_rate: Dropout 概率（默认 0.1）
    # norm_layer / norm_kwargs: 归一化配置
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,  # 构造函数（续行见下）
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):  # 续行：Dropout 与归一化配置
        super(_DenseASPPConv, self).__init__()  # 调用父类构造
        # 【注意】以下各行末尾的多余逗号使整段构成一个"元组表达式"语句（元素为各 add_module 的返回值 None），
        # 调用仍会依次执行、功能不受影响，但属于不规范写法（按要求保持原样）
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),  # 添加 1×1 降维卷积模块
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),  # 添加 BN 归一化模块（norm_kwargs 为 None 时展开空字典）
        self.add_module('relu1', nn.ReLU(True)),  # 添加就地 ReLU 模块
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),  # 添加 3×3 空洞卷积（padding=空洞率，保持尺寸）
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),  # 添加 BN 归一化模块
        self.add_module('relu2', nn.ReLU(True)),  # 添加就地 ReLU 模块
        self.drop_rate = drop_rate  # 保存 Dropout 概率

    # ---- forward 参数说明 ----
    # x: 输入特征
    def forward(self, x):  # 前向：Sequential 卷积栈 + 额外 Dropout
        features = super(_DenseASPPConv, self).forward(x)  # 按顺序执行 conv1→bn1→relu1→conv2→bn2→relu2
        if self.drop_rate > 0:  # 若配置了 Dropout
            features = F.dropout(features, p=self.drop_rate, training=self.training)  # 按当前训练/推理状态做 Dropout
        return features  # 返回特征


# 密集 ASPP 块：5 级空洞卷积（3/6/12/18/24）密集连接，
# 每级输入都是"原始特征 + 此前各级输出"在通道维的拼接
class _DenseASPPBlock(nn.Module):  # 密集 ASPP 块：继承 nn.Module
    # ---- __init__ 参数说明 ----
    # in_channels: 骨干输出通道数
    # inter_channels1: 每级 1×1 卷积的中间通道（256）
    # inter_channels2: 每级空洞卷积的输出通道（64）
    # norm_layer / norm_kwargs: 归一化配置
    def __init__(self, in_channels, inter_channels1, inter_channels2,  # 构造函数（续行见下）
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):  # 续行：归一化配置
        super(_DenseASPPBlock, self).__init__()  # 调用父类构造
        # 第 1 级：空洞率 3，输入仅原始特征
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,  # 空洞率 3，Dropout 0.1（续行见下）
                                     norm_layer, norm_kwargs)  # 续行：归一化配置
        # 第 2 级：空洞率 6，输入 = 原始特征 + 第 1 级输出（通道 +64）
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,  # 输入通道 +64（续行见下）
                                     norm_layer, norm_kwargs)  # 续行：归一化配置
        # 第 3 级：空洞率 12，输入 = 原始特征 + 前 2 级输出（通道 +128）
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,  # 输入通道 +128（续行见下）
                                      norm_layer, norm_kwargs)  # 续行：归一化配置
        # 第 4 级：空洞率 18，输入 = 原始特征 + 前 3 级输出（通道 +192）
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,  # 输入通道 +192（续行见下）
                                      norm_layer, norm_kwargs)  # 续行：归一化配置
        # 第 5 级：空洞率 24，输入 = 原始特征 + 前 4 级输出（通道 +256）
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,  # 输入通道 +256（续行见下）
                                      norm_layer, norm_kwargs)  # 续行：归一化配置

    # ---- forward 参数说明 ----
    # x: 骨干特征 (N, C, H, W)
    def forward(self, x):  # 前向：逐级密集连接（每级输出与累积特征拼接后送入下一级）
        aspp3 = self.aspp_3(x)  # 第 1 级输出 (N, 64, H, W)
        x = torch.cat([aspp3, x], dim=1)  # 与原始特征拼接 → (N, C+64, H, W)

        aspp6 = self.aspp_6(x)  # 第 2 级输出 (N, 64, H, W)
        x = torch.cat([aspp6, x], dim=1)  # 再拼接 → (N, C+128, H, W)

        aspp12 = self.aspp_12(x)  # 第 3 级输出 (N, 64, H, W)
        x = torch.cat([aspp12, x], dim=1)  # 再拼接 → (N, C+192, H, W)

        aspp18 = self.aspp_18(x)  # 第 4 级输出 (N, 64, H, W)
        x = torch.cat([aspp18, x], dim=1)  # 再拼接 → (N, C+256, H, W)

        aspp24 = self.aspp_24(x)  # 第 5 级输出 (N, 64, H, W)
        x = torch.cat([aspp24, x], dim=1)  # 再拼接 → (N, C+320, H, W) = (N, C+5*64, H, W)

        return x  # 返回密集拼接后的特征


# 便捷构造函数：按数据集 / 骨干 / 预训练选项构建 DenseASPP
# ---- 参数说明 ----
# dataset: 数据集名（决定类别数），默认 'citys'
# backbone: DenseNet 骨干类型，默认 'densenet121'
# pretrained: 是否加载完整模型预训练权重（True 或权重版本号字符串）
# root: 预训练权重文件存放根目录
# pretrained_base: 是否加载骨干的 ImageNet 预训练权重
# **kwargs: 透传给 DenseASPP 构造函数
def get_denseaspp(dataset='citys', backbone='densenet121', pretrained=False,  # 构造入口（续行见下）
                  root='~/.torch/models', pretrained_base=True, **kwargs):  # 续行：权重目录、骨干预训练开关、透传参数
    # 函数 docstring：get_denseaspp 参数说明与用法示例（原英文保留）
    r"""DenseASPP

    Parameters
    ----------
    dataset : str, default citys
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
    >>> model = get_denseaspp(dataset='citys', backbone='densenet121', pretrained=False)
    >>> print(model)
    """  # get_denseaspp 英文 docstring 结束（原样保留）
    acronyms = {  # 数据集名 → 预训练权重文件名中的缩写映射
        'pascal_voc': 'pascal_voc',  # Pascal VOC
        'pascal_aug': 'pascal_aug',  # Pascal VOC 增强版
        'ade20k': 'ade',  # ADE20K → 缩写 ade
        'coco': 'coco',  # COCO
        'citys': 'citys',  # Cityscapes
    }  # 缩写映射结束
    from ..data.dataloader import datasets  # 局部导入数据集注册表（避免循环导入），用于查询类别数
    model = DenseASPP(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建 DenseASPP
    if pretrained:  # 若需加载完整模型预训练权重
        from .model_store import get_model_file  # 局部导入权重文件路径解析工具
        device = torch.device(kwargs['local_rank'])  # 【注意】直接取 kwargs['local_rank']，未传该键时会 KeyError；用 int 构造 device 的写法也较脆弱
        model.load_state_dict(torch.load(get_model_file('denseaspp_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 拼出权重文件名 'denseaspp_<backbone>_<数据集缩写>' 并加载（【注意】torch.load 未指定 weights_only，新版 PyTorch 会告警）
                              map_location=device))  # 将权重张量映射到指定设备
    return model  # 返回构建好的模型


def get_denseaspp_densenet121_citys(**kwargs):  # 便捷构造：Cityscapes + DenseNet121 版 DenseASPP
    return get_denseaspp('citys', 'densenet121', **kwargs)  # 委托 get_denseaspp


def get_denseaspp_densenet161_citys(**kwargs):  # 便捷构造：Cityscapes + DenseNet161 版 DenseASPP
    return get_denseaspp('citys', 'densenet161', **kwargs)  # 委托 get_denseaspp


def get_denseaspp_densenet169_citys(**kwargs):  # 便捷构造：Cityscapes + DenseNet169 版 DenseASPP
    return get_denseaspp('citys', 'densenet169', **kwargs)  # 委托 get_denseaspp


def get_denseaspp_densenet201_citys(**kwargs):  # 便捷构造：Cityscapes + DenseNet201 版 DenseASPP
    return get_denseaspp('citys', 'densenet201', **kwargs)  # 委托 get_denseaspp


if __name__ == '__main__':  # 直接运行本文件时的简易自测入口
    img = torch.randn(2, 3, 480, 480)  # 构造随机输入 (2, 3, 480, 480)
    model = get_denseaspp_densenet121_citys()  # 构建 Cityscapes + DenseNet121 版 DenseASPP
    outputs = model(img)  # 前向推理，验证输出形状
