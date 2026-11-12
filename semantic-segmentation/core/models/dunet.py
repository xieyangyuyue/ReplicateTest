# ============================================================================
# 【文件说明】dunet.py —— DUNet（数据依赖解码 U 形网络）的 PyTorch 实现，
#              论文题为《Decoders Matter for Semantic Segmentation: Data-Dependent
#              Decoding Enables Flexible Feature Aggregation》(CVPR 2019)。
#              基于 SegBaseModel 的空洞 ResNet 骨干（dilated 模式，c3/c4 保持 1/8
#              分辨率）；解码器 FeatureFused 先把 c2/c3 投影并对齐到 c4 尺寸拼接，
#              经 _DUHead 两层 3x3 卷积精炼，最后用 DUpsampling（可学习的
#              数据依赖上采样）一步恢复 8 倍分辨率到原图尺寸。
# 【核心思想】解码器同样重要：传统双线性上采样是固定算子、不可学习。DUpsampling
#            用一个 1x1 卷积直接预测 out×scale² 个通道，再经两次 permute+view
#            把通道维“折叠”回空间维（类 PixelShuffle 重排），使上采样本身可学习、
#            能与编码特征联合优化，实现灵活的特征聚合；原论文另含语义/实例分支，
#            本文件仅实现语义分割部分。
# ============================================================================
"""Decoders Matter for Semantic Segmentation"""  # 原英文 docstring 保留：解码器对语义分割至关重要
import torch  # PyTorch 主库：张量运算与权重加载
import torch.nn as nn  # 神经网络模块：nn.Module、卷积/BN/激活等网络层
import torch.nn.functional as F  # 函数式接口：双线性上采样 F.interpolate 等

from .segbase import SegBaseModel  # 导入分割模型公共基类（提供空洞 ResNet 骨干与 base_forward 前向接口）
from .fcn import _FCNHead  # 导入 FCN 分割头（辅助分支使用）

__all__ = ['DUNet', 'get_dunet', 'get_dunet_resnet50_pascal_voc',  # 模块公开接口：模型类 + 工厂函数
           'get_dunet_resnet101_pascal_voc', 'get_dunet_resnet152_pascal_voc']  # 续行：三种骨干的便捷构造函数


# The model may be wrong because lots of details missing in paper.
class DUNet(SegBaseModel):  # DUNet 主网络（原英文注释保留：因论文细节缺失，实现可能与原文有出入）；继承 SegBaseModel 获得骨干
    # 类 docstring：DUNet 论文引用信息（原英文保留）
    """Decoders Matter for Semantic Segmentation

    Reference:
        Zhi Tian, Tong He, Chunhua Shen, and Youliang Yan.
        "Decoders Matter for Semantic Segmentation:
        Data-Dependent Decoding Enables Flexible Feature Aggregation." CVPR, 2019
    """  # 上方英文 docstring 原样保留：论文引用信息

    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数
    # backbone：骨干类型 'resnet50'/'resnet101'/'resnet152'（三者 c2/c3/c4 通道一致：512/1024/2048）
    # aux：是否启用辅助损失分支（【注意】默认 True，与其他模型默认 False 不同）
    # pretrained_base：是否加载骨干 ImageNet 预训练权重
    # **kwargs：透传给基类与子模块（如 norm_layer）
    def __init__(self, nclass, backbone='resnet50', aux=True, pretrained_base=True, **kwargs):  # 构造函数
        super(DUNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)  # 调用基类构造：构建空洞 ResNet 骨干（未传 jpu → 默认 False → dilated=True，c3/c4 为 1/8 分辨率）
        self.head = _DUHead(2144, **kwargs)  # 主解码头：输入 2144=2048(c4)+48(c3 投影)+48(c2 投影)，输出 256 通道
        self.dupsample = DUpsampling(256, nclass, scale_factor=8, **kwargs)  # 数据依赖上采样：256→nclass，一步 ×8 恢复到全分辨率（1/8 → 1/1）
        if aux:  # 启用辅助分支时
            self.auxlayer = _FCNHead(1024, 256, **kwargs)  # 辅助头：FCN 头作用于 c3（1024 通道，内部先降到 256）→ 输出 256 通道
            self.aux_dupsample = DUpsampling(256, nclass, scale_factor=8, **kwargs)  # 辅助分支的 DUpsampling：256→nclass，×8 上采样

        self.__setattr__('exclusive',  # 登记自有模块名单（骨干 self.pretrained 参数除外），供训练脚本区分参数组
                         ['dupsample', 'head', 'auxlayer', 'aux_dupsample'] if aux else ['dupsample', 'head'])  # aux=True 时含辅助头，否则仅主头与上采样

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像 (N, 3, H, W)
    # 返回：aux=True 时 (主输出, 辅助输出)，均为 (N, nclass, H, W)；否则仅 (主输出,)
    def forward(self, x):  # 前向：骨干提特征 → 融合解码头 → DUpsampling 上采样 → 可选辅助分支
        c1, c2, c3, c4 = self.base_forward(x)  # 空洞骨干四阶段特征：c1 (N,256,H/4)、c2 (N,512,H/8)、c3 (N,1024,H/8)、c4 (N,2048,H/8)（dilated 模式下 c2/c3/c4 同为 1/8；【注意】c1 解包后未被使用）
        outputs = []  # 输出收集列表
        x = self.head(c2, c3, c4)  # 解码头：c2/c3 投影对齐后与 c4 拼接，再卷积精炼 → (N, 256, H/8, W/8)
        x = self.dupsample(x)  # 数据依赖上采样 ×8 → (N, nclass, H, W)
        outputs.append(x)  # 收集主输出

        if self.aux:  # 启用辅助分支时
            auxout = self.auxlayer(c3)  # FCN 头处理 c3 → (N, 256, H/8, W/8)
            auxout = self.aux_dupsample(auxout)  # DUpsampling ×8 → (N, nclass, H, W)
            outputs.append(auxout)  # 收集辅助输出
        return tuple(outputs)  # 返回输出元组


class FeatureFused(nn.Module):  # 特征融合模块：把浅层 c2/c3 投影到统一通道并对齐到 c4 尺寸后拼接
    """Module for fused features"""  # 原英文 docstring 保留：用于特征融合的模块

    # ----------------------- __init__ 参数说明 -----------------------
    # inter_channels：c2/c3 投影后的中间通道数（默认 48）
    # norm_layer：归一化层类型
    # **kwargs：兼容透传（本类中未使用）
    def __init__(self, inter_channels=48, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(FeatureFused, self).__init__()  # 调用父类构造函数
        self.conv2 = nn.Sequential(  # c2 分支：1x1 卷积 + BN + ReLU
            nn.Conv2d(512, inter_channels, 1, bias=False),  # 1x1 卷积 512→48（512 为 ResNet c2 通道数，硬编码，三种骨干均适用）
            norm_layer(inter_channels),  # BN（48 通道）
            nn.ReLU(True)  # 就地 ReLU 激活
        )  # c2 分支 Sequential 结束
        self.conv3 = nn.Sequential(  # c3 分支：1x1 卷积 + BN + ReLU
            nn.Conv2d(1024, inter_channels, 1, bias=False),  # 1x1 卷积 1024→48（1024 为 ResNet c3 通道数，硬编码）
            norm_layer(inter_channels),  # BN（48 通道）
            nn.ReLU(True)  # 就地 ReLU 激活
        )  # c3 分支 Sequential 结束

    # ----------------------- forward 参数说明 -----------------------
    # c2/c3/c4：骨干三阶段特征（dilated 模式下均在 1/8 分辨率）
    # 返回：拼接特征 (N, 2048+2*inter_channels, H/8, W/8) = (N, 2144, H/8, W/8)
    def forward(self, c2, c3, c4):  # 前向：c2/c3 对齐到 c4 尺寸后投影拼接
        size = c4.size()[2:]  # 以 c4 的空间尺寸为融合基准
        c2 = self.conv2(F.interpolate(c2, size, mode='bilinear', align_corners=True))  # c2 双线性对齐到 c4 尺寸后 1x1 投影 → (N, 48, H/8, W/8)（dilated 模式下二者本就同尺寸，插值不改变大小）
        c3 = self.conv3(F.interpolate(c3, size, mode='bilinear', align_corners=True))  # c3 对齐并投影 → (N, 48, H/8, W/8)
        fused_feature = torch.cat([c4, c3, c2], dim=1)  # 通道维拼接：2048+48+48=2144 → (N, 2144, H/8, W/8)
        return fused_feature  # 返回拼接后的多尺度特征


class _DUHead(nn.Module):  # DUNet 解码头：FeatureFused 多尺度拼接 + 两层 3x3 卷积精炼
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：拼接后的输入通道数（DUNet 中为 2144）
    # norm_layer：归一化层类型
    # **kwargs：透传给 FeatureFused
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(_DUHead, self).__init__()  # 调用父类构造函数
        self.fuse = FeatureFused(norm_layer=norm_layer, **kwargs)  # 多尺度特征拼接融合模块
        self.block = nn.Sequential(  # 两层 3x3 卷积块（VGG 风格精炼）
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),  # 3x3 卷积 padding=1（分辨率不变），2144→256，无偏置（后接 BN）
            norm_layer(256),  # BN（256 通道）
            nn.ReLU(True),  # 就地 ReLU 激活
            nn.Conv2d(256, 256, 3, padding=1, bias=False),  # 3x3 卷积 256→256，padding=1
            norm_layer(256),  # BN（256 通道）
            nn.ReLU(True)  # 就地 ReLU 激活
        )  # 精炼卷积块 Sequential 结束

    # ----------------------- forward 参数说明 -----------------------
    # c2/c3/c4：骨干三阶段特征；输出 (N, 256, H/8, W/8)
    def forward(self, c2, c3, c4):  # 前向：拼接融合 → 卷积精炼
        fused_feature = self.fuse(c2, c3, c4)  # → (N, 2144, H/8, W/8)
        out = self.block(fused_feature)  # → (N, 256, H/8, W/8)
        return out  # 返回解码特征


class DUpsampling(nn.Module):  # 数据依赖上采样：1x1 卷积预测 out×scale² 通道后重排到空间维（可学习的上采样算子）
    """DUsampling module"""  # 原英文 docstring 保留（原文即写作 DUsampling，少一个 p）：DUpsampling 模块

    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：输入通道数（DUNet 中为 256）
    # out_channels：输出通道数（nclass）
    # scale_factor：上采样倍数（DUNet 中为 8：1/8 分辨率 → 原尺寸）
    # **kwargs：兼容透传（本类中未使用）
    def __init__(self, in_channels, out_channels, scale_factor=2, **kwargs):  # 构造函数
        super(DUpsampling, self).__init__()  # 调用父类构造函数
        self.scale_factor = scale_factor  # 保存上采样倍数，重排时使用
        self.conv_w = nn.Conv2d(in_channels, out_channels * scale_factor * scale_factor, 1, bias=False)  # 1x1 卷积：256 → nclass*scale²（上采样后的每个像素值均由卷积预测，故称“数据依赖”），无偏置

    # ----------------------- forward 参数说明 -----------------------
    # x：(N, in_channels, H, W)；输出 (N, out_channels, H*scale, W*scale)
    def forward(self, x):  # 前向：卷积扩充通道 → 两次 permute+view 把通道折叠进 H/W 维（类 PixelShuffle 重排）
        x = self.conv_w(x)  # → (N, C, H, W)，其中 C = out_channels * scale²
        n, c, h, w = x.size()  # 拆出批量 N、通道 C、高 H、宽 W

        # N, C, H, W --> N, W, H, C
        x = x.permute(0, 3, 2, 1).contiguous()  # 原英文注释保留：维度重排为 (N, W, H, C)；contiguous 保证后续 view 内存连续
        # （补充说明）通道维换到最后，为下一步把 C 折叠进 H 做准备

        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, h * self.scale_factor, c // self.scale_factor)  # 原英文注释保留：把 C 维的 scale 份折叠进 H 维 → (N, W, H*scale, C/scale)

        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()  # 原英文注释保留：交换 W 与 H*scale 两个维度

        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, h * self.scale_factor, w * self.scale_factor, c // (self.scale_factor * self.scale_factor))  # 原英文注释保留：再把 C/scale 的 scale 份折叠进 W 维 → (N, H*scale, W*scale, C/scale²)

        # N, H * scale, W * scale, C // (scale ** 2) -- > N, C // (scale ** 2), H * scale, W * scale
        x = x.permute(0, 3, 1, 2)  # 原英文注释保留：通道维换回第 1 维 → (N, out_channels, H*scale, W*scale)，即 (N, nclass, H, W)

        return x  # 返回上采样结果


# ----------------------- get_dunet 参数说明 -----------------------
# dataset：数据集名称（决定类别数）
# backbone：骨干类型 'resnet50'/'resnet101'/'resnet152'
# pretrained：是否加载本模型的分割预训练权重
# root：权重文件查找根目录
# pretrained_base：是否加载骨干 ImageNet 预训练权重
# **kwargs：透传；pretrained=True 时须包含 local_rank 用于指定加载设备
def get_dunet(dataset='pascal_voc', backbone='resnet50', pretrained=False,  # 工厂函数：按数据集与骨干构建 DUNet 并可选加载权重
              root='~/.torch/models', pretrained_base=True, **kwargs):  # 续行：默认加载骨干 ImageNet 预训练权重
    acronyms = {  # 数据集名 → 权重文件名缩写映射
        'pascal_voc': 'pascal_voc',  # Pascal VOC 2012
        'pascal_aug': 'pascal_aug',  # VOC 增广集（SBD）
        'ade20k': 'ade',  # ADE20K 场景解析数据集
        'coco': 'coco',  # COCO Stuff
        'citys': 'citys',  # Cityscapes 城市街景（19 类）
    }  # 缩写映射表结束
    from ..data.dataloader import datasets  # 延迟导入数据集注册表（相对导入写法）
    model = DUNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建模型
    if pretrained:  # 若需加载分割预训练权重
        from .model_store import get_model_file  # 延迟导入权重文件路径解析工具
        device = torch.device(kwargs['local_rank'])  # 【注意】可疑写法：kwargs 未包含 local_rank 时会 KeyError；且以 int 直接构造 torch.device 的行为依赖 PyTorch 版本（更稳妥写法为 torch.device('cuda', rank)）
        model.load_state_dict(torch.load(get_model_file('dunet_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 拼出权重文件名（如 dunet_resnet50_pascal_voc.pth）并读取
                              map_location=device))  # 将权重映射到指定设备
    return model  # 返回构建/加载完成的模型


def get_dunet_resnet50_pascal_voc(**kwargs):  # 便捷构造函数：ResNet50 骨干 + Pascal VOC
    return get_dunet('pascal_voc', 'resnet50', **kwargs)  # 委托给 get_dunet


def get_dunet_resnet101_pascal_voc(**kwargs):  # 便捷构造函数：ResNet101 骨干 + Pascal VOC
    return get_dunet('pascal_voc', 'resnet101', **kwargs)  # 委托给 get_dunet


def get_dunet_resnet152_pascal_voc(**kwargs):  # 便捷构造函数：ResNet152 骨干 + Pascal VOC
    return get_dunet('pascal_voc', 'resnet152', **kwargs)  # 委托给 get_dunet


if __name__ == '__main__':  # 直接运行本文件时的小型自测：前向一次验证网络可跑通
    img = torch.randn(2, 3, 256, 256)  # 构造随机输入 (2, 3, 256, 256)
    model = get_dunet_resnet50_pascal_voc()  # 构建 ResNet50 + VOC 版模型（工厂默认加载骨干预训练权重）
    outputs = model(img)  # 前向得到输出元组（主/辅输出均为 (2, 21, 256, 256)，21 为 VOC 类别数）
