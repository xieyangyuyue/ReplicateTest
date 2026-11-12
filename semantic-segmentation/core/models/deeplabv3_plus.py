# ============================================================================
# 【文件说明】deeplabv3_plus.py —— 实现 DeepLabV3+（DeepLabV3Plus）语义分割模型：
#   采用 Xception（深度可分离卷积）骨干 + 编码器-解码器结构，包含主模型类
#   DeepLabV3Plus、解码融合头 _DeepLabHead，以及快捷构建函数 get_deeplabv3_plus。
# 【核心思想】编码器为空洞 Xception（dilated=True 时 output stride=8）：高层特征
#   （Exit Flow 输出，2048 通道）送入 ASPP（空洞率 12/24/36）提取多尺度上下文得
#   256 通道特征；解码器将该特征上采样 2 倍至低层特征（Entry Flow block1 输出，
#   1/4 分辨率、128 通道）的尺寸，低层特征先经 3×3 卷积压缩为 48 通道以抑制冗余，
#   随后二者拼接（256+48=304）并经两个 3×3 卷积块 + 1×1 分类器输出，
#   兼顾深层语义与浅层空间细节、显著改善分割边界。
# 【论文名称】Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable
#   Convolution for Semantic Image Segmentation." ECCV 2018.
# 【注意】1) 论文解码器使用深度可分离卷积，本实现的 _ConvBNReLU 为普通卷积组合层；
#   2) get_deeplabv3_plus 的 backbone 参数实际被忽略（模型固定使用 Xception）。
# ============================================================================
import torch  # PyTorch 主库：张量运算、设备指定（torch.device）与权重加载（torch.load）
import torch.nn as nn  # 神经网络子模块：nn.Module / nn.Sequential / nn.Conv2d 等常用层
import torch.nn.functional as F  # 函数式接口：F.interpolate 双线性上采样等无参数操作

from .base_models.xception import get_xception  # Xception 骨干构建函数（Aligned Xception，内部使用深度可分离卷积）
from .deeplabv3 import _ASPP  # 复用 DeepLabV3 的 ASPP 模块（空洞空间金字塔池化）
from .fcn import _FCNHead  # FCN 辅助分类头：对中间层 c3 计算辅助损失（深度监督）
from ..nn import _ConvBNReLU  # 卷积+BN+ReLU 三合一组合层（普通卷积，非论文中的深度可分离卷积）

__all__ = ['DeepLabV3Plus', 'get_deeplabv3_plus', 'get_deeplabv3_plus_xception_voc']  # __all__：模块对外导出的符号列表


class DeepLabV3Plus(nn.Module):  # DeepLabV3+ 模型类：【注意】直接继承 nn.Module（而非 SegBaseModel），自带 Xception 骨干与自定义 base_forward
    # 类 docstring：DeepLabV3Plus 模型说明与参数（原英文保留）
    r"""DeepLabV3Plus
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'xception').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """
    # 【__init__ 参数说明】
    #   nclass：分割类别数（不含背景，由数据集决定）
    #   backbone：骨干类型参数（默认 'xception'）；【注意】该参数实际未被使用，模型固定使用 Xception
    #   aux：是否启用辅助损失（默认 True，与 PSPNet/DeepLabV3 默认 False 不同）
    #   pretrained_base：是否加载 ImageNet 预训练骨干权重
    #   dilated：骨干是否使用空洞卷积；True 时 output stride=8（特征更精细），False 时 output stride=32
    #   **kwargs：透传给骨干、头部与 ASPP 的额外参数（norm_layer、norm_kwargs 等）
    def __init__(self, nclass, backbone='xception', aux=True, pretrained_base=True, dilated=True, **kwargs):  # 模型构造函数
        super(DeepLabV3Plus, self).__init__()  # 调用 nn.Module 基类构造
        self.aux = aux  # 记录是否启用辅助损失
        self.nclass = nclass  # 记录分割类别数
        output_stride = 8 if dilated else 32  # 输出特征相对输入的下采样倍数：空洞模式 8（高层特征保持 1/8 分辨率），普通模式 32

        self.pretrained = get_xception(pretrained=pretrained_base, output_stride=output_stride, **kwargs)  # 构建 Xception 骨干（可加载 ImageNet 预训练权重，按 output_stride 配置空洞率与步长）

        # deeplabv3 plus  （原英文注释，原样保留）
        self.head = _DeepLabHead(nclass, **kwargs)  # 主分割头：ASPP 编码 + 低层特征解码融合，输出 (N, nclass, H/4, W/4)
        if aux:  # 若启用辅助损失
            self.auxlayer = _FCNHead(728, nclass, **kwargs)  # 辅助头：输入 Middle Flow 输出 c3（728 通道、1/8 分辨率；通道数硬编码，仅适配 Xception）

    def base_forward(self, x):  # 骨干网络前向：手动展开 Xception 三个阶段，返回 (低层特征 c1, 中层特征 c3, 高层特征 c4)
        # Entry flow  （原英文注释，原样保留）
        x = self.pretrained.conv1(x)  # 首层 3×3 卷积：3→32 通道，stride=2，(N, 3, H, W) → (N, 32, H/2, W/2)
        x = self.pretrained.bn1(x)  # 批归一化（32 通道）
        x = self.pretrained.relu(x)  # ReLU 激活

        x = self.pretrained.conv2(x)  # 第二层 3×3 卷积：32→64 通道，stride=1，(N, 64, H/2, W/2)
        x = self.pretrained.bn2(x)  # 批归一化（64 通道）
        x = self.pretrained.relu(x)  # ReLU 激活

        x = self.pretrained.block1(x)  # Entry block1：64→128 通道，stride=2，降到 1/4 分辨率 (N, 128, H/4, W/4)
        # add relu here  （原英文注释，原样保留）
        x = self.pretrained.relu(x)  # block1 后额外补一次 ReLU（与骨干原始实现对齐）
        low_level_feat = x  # 低层特征 c1：(N, 128, H/4, W/4)，供解码器融合空间细节

        x = self.pretrained.block2(x)  # Entry block2：128→256 通道，stride=2，降到 1/8 分辨率
        x = self.pretrained.block3(x)  # Entry block3：256→728 通道（dilated 时 stride=1，保持 1/8 分辨率）

        # Middle flow  （原英文注释，原样保留）
        x = self.pretrained.midflow(x)  # 中间流：16 个 728 通道深度可分离卷积块串联，通道与分辨率均不变
        mid_level_feat = x  # 中层特征 c3：(N, 728, H/8, W/8)，供辅助损失头使用

        # Exit flow  （原英文注释，原样保留）
        x = self.pretrained.block20(x)  # Exit block20：728→1024 通道（dilated 时 stride=1，保持 1/8 分辨率）
        x = self.pretrained.relu(x)  # ReLU 激活
        x = self.pretrained.conv3(x)  # 深度可分离卷积：1024→1536 通道
        x = self.pretrained.bn3(x)  # 批归一化（1536 通道）
        x = self.pretrained.relu(x)  # ReLU 激活

        x = self.pretrained.conv4(x)  # 深度可分离卷积：1536→1536 通道
        x = self.pretrained.bn4(x)  # 批归一化（1536 通道）
        x = self.pretrained.relu(x)  # ReLU 激活

        x = self.pretrained.conv5(x)  # 深度可分离卷积：1536→2048 通道
        x = self.pretrained.bn5(x)  # 批归一化（2048 通道）
        x = self.pretrained.relu(x)  # ReLU 激活
        return low_level_feat, mid_level_feat, x  # 返回 (c1, c3, c4)：c4 为高层特征 (N, 2048, H/8, W/8)（dilated 模式下）

    def forward(self, x):  # 前向传播：x 为输入图像张量 (N, 3, H, W)
        size = x.size()[2:]  # 记录输入空间尺寸 (H, W)，用于最后上采样恢复原分辨率
        c1, c3, c4 = self.base_forward(x)  # 骨干前向：c1=(N, 128, H/4, W/4)、c3=(N, 728, H/8, W/8)、c4=(N, 2048, H/8, W/8)
        outputs = list()  # 输出收集列表：主输出在前，可选辅助输出在后
        x = self.head(c4, c1)  # 主头（ASPP 编码 + 与低层特征 c1 解码融合）：→ (N, nclass, H/4, W/4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 双线性上采样 4 倍：(N, nclass, H/4, W/4) → (N, nclass, H, W)
        outputs.append(x)  # 主输出加入列表
        if self.aux:  # 若启用辅助损失
            auxout = self.auxlayer(c3)  # 辅助头：(N, 728, H/8, W/8) → (N, nclass, H/8, W/8)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)  # 辅助输出上采样回原图分辨率：(N, nclass, H, W)
            outputs.append(auxout)  # 辅助输出加入列表
        return tuple(outputs)  # 返回 (主输出, 辅助输出) 元组，供损失函数按序加权计算


class _DeepLabHead(nn.Module):  # DeepLabV3+ 头部：ASPP 编码器 + 低层特征融合解码器
    # 参数：nclass 类别数；c1_channels 低层特征通道数（Xception block1 输出为 128）；norm_layer 归一化层（默认 BN2d）；**kwargs 额外参数
    def __init__(self, nclass, c1_channels=128, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DeepLabHead, self).__init__()  # 调用 nn.Module 基类构造
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer, **kwargs)  # ASPP 编码器：输入 2048 通道、空洞率 12/24/36（对应 output stride=8）、输出 256 通道；【注意】_ASPP 的 norm_kwargs 为必填参数但此处未显式传入，仅当外部 kwargs 恰好包含 norm_kwargs 才能成功构造，否则抛 TypeError
        self.c1_block = _ConvBNReLU(c1_channels, 48, 3, padding=1, norm_layer=norm_layer)  # 低层特征处理：3×3 卷积 128→48 通道，压缩低层冗余信息、保留边界细节（论文设定 48 通道）
        self.block = nn.Sequential(  # 融合后的解码卷积块
            _ConvBNReLU(304, 256, 3, padding=1, norm_layer=norm_layer),  # 3×3 卷积：304(=256 ASPP 输出 + 48 低层特征)→256，padding=1 保持尺寸
            nn.Dropout(0.5),  # Dropout，丢弃率 0.5，防止过拟合
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer),  # 3×3 卷积：256→256，进一步细化融合特征与边界
            nn.Dropout(0.1),  # Dropout，丢弃率 0.1
            nn.Conv2d(256, nclass, 1))  # 1×1 分类卷积：256→nclass，输出逐像素类别 logits

    def forward(self, x, c1):  # 前向：x 为高层特征 (N, 2048, H/8, W/8)，c1 为低层特征 (N, 128, H/4, W/4)
        size = c1.size()[2:]  # 以低层特征尺寸 (H/4, W/4) 作为上采样目标（解码对齐基准）
        c1 = self.c1_block(c1)  # 低层特征降维：(N, 128, H/4, W/4) → (N, 48, H/4, W/4)
        x = self.aspp(x)  # ASPP 编码聚合多尺度上下文：(N, 2048, H/8, W/8) → (N, 256, H/8, W/8)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 双线性上采样 2 倍恢复分辨率：(N, 256, H/8, W/8) → (N, 256, H/4, W/4)，与低层特征空间对齐
        return self.block(torch.cat([x, c1], dim=1))  # 高低层特征按通道拼接后解码：(N, 304, H/4, W/4) → (N, nclass, H/4, W/4)


# 【工厂函数】get_deeplabv3_plus —— 按数据集（骨干固定为 Xception）构建 DeepLabV3+，可选加载预训练权重
#   参数：dataset 数据集名称；backbone 骨干类型（【注意】实际被忽略，固定使用 Xception）；pretrained 是否加载完整模型预训练权重；
#         root 权重文件存放根目录；pretrained_base 是否加载 ImageNet 预训练骨干；**kwargs 透传给 DeepLabV3Plus
def get_deeplabv3_plus(dataset='pascal_voc', backbone='xception', pretrained=False, root='~/.torch/models',  # 默认 VOC + Xception，不加载完整模型权重
                       pretrained_base=True, **kwargs):  # 默认加载 ImageNet 预训练骨干
    acronyms = {  # 数据集名 → 预训练权重文件名中使用的缩写映射
        'pascal_voc': 'pascal_voc',  # PASCAL VOC
        'pascal_aug': 'pascal_aug',  # PASCAL VOC 增广版
        'ade20k': 'ade',  # ADE20K 缩写为 ade
        'coco': 'coco',  # COCO
        'citys': 'citys',  # Cityscapes
    }  # 数据集缩写映射字典结束
    from ..data.dataloader import datasets  # 延迟导入数据集注册表（避免循环导入），用于查询各类别数 NUM_CLASS
    model = DeepLabV3Plus(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建 DeepLabV3+（backbone 参数在内部被忽略）
    if pretrained:  # 需要加载完整模型预训练权重时
        from .model_store import get_model_file  # 延迟导入：按模型名定位（必要时下载）权重文件
        device = torch.device(kwargs['local_rank'])  # 【注意】可疑写法：直接索引 kwargs['local_rank']，未传该键会 KeyError；且 local_rank 通常为 int，torch.device(int) 会抛 TypeError，疑似应为 torch.device('cuda', kwargs['local_rank'])
        model.load_state_dict(  # 加载权重并覆盖模型参数
            torch.load(get_model_file('deeplabv3_plus_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 拼出权重文件名 deeplabv3_plus_{backbone}_{dataset缩写} 并读取
                map_location=device))  # map_location：将权重张量映射到 local_rank 指定的设备
    return model  # 返回构建（并可能已加载权重）的模型


def get_deeplabv3_plus_xception_voc(**kwargs):  # 快捷构建：DeepLabV3+ + Xception + PASCAL VOC
    return get_deeplabv3_plus('pascal_voc', 'xception', **kwargs)  # 委托 get_deeplabv3_plus，固定数据集与骨干


if __name__ == '__main__':  # 直接运行本文件时的简单自测入口
    model = get_deeplabv3_plus_xception_voc()  # 构建 DeepLabV3+（Xception + VOC，21 类，默认启用辅助头）
