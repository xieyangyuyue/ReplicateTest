# ============================================================================
# 【文件说明】basic.py —— 语义分割网络的基础构件库：提供 Conv+BN+ReLU/PReLU
#             组合模块（_ConvBNReLU/_ConvBNPReLU/_ConvBN/_BNPReLU）、PSPNet 的
#             金字塔池化模块（_PSPModule，本文件内为废弃遗留版本）、MobileNet
#             深度卷积块（_DepthwiseConv）与 MobileNetV2 倒残差块（InvertedResidual）。
# 【核心思想】将高频出现的 "卷积 → 归一化 → 激活" 三件套封装为可复用类，并统一
#             构造接口（norm_layer 可替换为 SyncBN 等）；深度可分离卷积把标准卷积
#             拆成 "逐通道卷积 + 逐点 1x1 卷积" 两步，大幅降低参数量与计算量；
#             倒残差块先 1x1 卷积升维、再深度卷积提特征、最后 1x1 卷积降维
#             （与 ResNet 残差的"先降后升"相反），形状匹配时叠加恒等捷径。
# ============================================================================
"""Basic Module for Semantic Segmentation"""  # 原有 docstring：语义分割基础模块
import torch  # 导入 PyTorch 主库，提供张量与随机数等基础能力
import torch.nn as nn  # 导入神经网络子模块（Conv2d、BatchNorm2d、ReLU 等）
import torch.nn.functional as F  # 导入函数式接口（bilinear 插值 interpolate 等无状态操作）

__all__ = ['_ConvBNPReLU', '_ConvBN', '_BNPReLU', '_ConvBNReLU', '_DepthwiseConv', 'InvertedResidual']  # `import *` 时导出的符号白名单；【注意】_PSPModule 未列入（实际使用的是 core/models/espnet.py 中自带的版本，本文件的 _PSPModule 疑似废弃遗留代码）


class _ConvBNReLU(nn.Module):  # 基础三件套模块：2D 卷积 + BN + ReLU/ReLU6
    # 形参说明: in_channels 输入通道数; out_channels 输出通道数; kernel_size 卷积核尺寸
    # 形参说明: stride 步长(默认1); padding 零填充(默认0); dilation 空洞卷积率(默认1)
    # 形参说明: groups 分组卷积组数(默认1即普通卷积); relu6 为 True 时用 ReLU6 替代 ReLU
    # 形参说明: norm_layer 归一化层构造函数(默认 BatchNorm2d, 可换 SyncBN 等); **kwargs 兼容额外参数
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造参数续行：空洞率/分组/ReLU6 开关/归一化层/额外关键字参数
        super(_ConvBNReLU, self).__init__()  # 调用父类 nn.Module 的构造函数完成模块注册
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)  # 2D 卷积层；bias=False 因紧随其后的 BN 自带平移参数 beta，省去冗余偏置
        self.bn = norm_layer(out_channels)  # 批归一化：稳定各通道分布，加速收敛
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)  # 激活层；relu6=True 时用 ReLU6(上限6, 适合移动端量化)；【注意】以位置参数传 inplace 在新版 PyTorch 已废弃，建议写 inplace=True

    def forward(self, x):  # 前向传播：x 为特征图 (N, in_channels, H, W)
        x = self.conv(x)  # 卷积: (N, in_channels, H, W) -> (N, out_channels, H', W')，H'/W' 由 stride/padding/dilation 决定
        x = self.bn(x)  # 归一化: 形状不变
        x = self.relu(x)  # 非线性激活: 形状不变（inplace=True 原地修改节省显存）
        return x  # 返回激活后的特征图 (N, out_channels, H', W')


class _ConvBNPReLU(nn.Module):  # 三件套变体：2D 卷积 + BN + PReLU（参数化 ReLU，负半轴斜率可学习）
    # 形参说明: 与 _ConvBNReLU 基本一致，仅无 relu6 参数（固定使用 PReLU 激活）
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造参数续行：空洞率/分组/归一化层/额外关键字参数（PReLU 变体无 relu6 参数）
        super(_ConvBNPReLU, self).__init__()  # 初始化父类 nn.Module
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)  # 2D 卷积层（同样不带偏置，由 BN 的 beta 补偿）
        self.bn = norm_layer(out_channels)  # 批归一化层
        self.prelu = nn.PReLU(out_channels)  # PReLU：每个通道一个可学习斜率，对小幅负值仍保留梯度信息

    def forward(self, x):  # 前向传播：x 形状 (N, in_channels, H, W)
        x = self.conv(x)  # 卷积: (N, in_channels, H, W) -> (N, out_channels, H', W')
        x = self.bn(x)  # 归一化: 形状不变
        x = self.prelu(x)  # PReLU 激活: 形状不变
        return x  # 返回 (N, out_channels, H', W')


class _ConvBN(nn.Module):  # 二件套模块：2D 卷积 + BN（不含激活，常见于网络末端 logits 前）
    # 形参说明: 与 _ConvBNReLU 完全一致（仅无激活相关参数）
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造参数续行：空洞率/分组/归一化层/额外关键字参数（无激活变体）
        super(_ConvBN, self).__init__()  # 初始化父类 nn.Module
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)  # 2D 卷积层（bias=False）
        self.bn = norm_layer(out_channels)  # 批归一化层

    def forward(self, x):  # 前向传播：x 形状 (N, in_channels, H, W)
        x = self.conv(x)  # 卷积: (N, in_channels, H, W) -> (N, out_channels, H', W')
        x = self.bn(x)  # 归一化: 形状不变
        return x  # 返回未激活的特征图 (N, out_channels, H', W')


class _BNPReLU(nn.Module):  # 二件套模块：BN + PReLU（不含卷积，常作独立的后处理单元）
    # 形参说明: out_channels 通道数; norm_layer 归一化层构造函数; **kwargs 兼容额外参数
    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_BNPReLU, self).__init__()  # 初始化父类 nn.Module
        self.bn = norm_layer(out_channels)  # 批归一化层
        self.prelu = nn.PReLU(out_channels)  # 逐通道可学习斜率的 PReLU

    def forward(self, x):  # 前向传播：x 形状 (N, C, H, W)
        x = self.bn(x)  # 归一化: 形状不变
        x = self.prelu(x)  # 激活: 形状不变
        return x  # 返回处理后的特征图


# -----------------------------------------------------------------
#                      For PSPNet
# -----------------------------------------------------------------
# （中文补充）以下为 PSPNet 专用模块
class _PSPModule(nn.Module):  # PSPNet 金字塔池化模块：多尺度全局上下文聚合；【注意】本类未被 __all__ 导出且含明显 bug（见下），实际使用的是 core/models/espnet.py 中的同名实现
    # 形参说明: in_channels 输入通道数; sizes 各池化分支输出尺寸，默认 (1,2,3,6) 分别捕获全局与 1/2、1/3、1/6 尺度上下文
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), **kwargs):
        super(_PSPModule, self).__init__()  # 初始化父类 nn.Module
        out_channels = int(in_channels / 4)  # 每个分支统一压缩到 in_channels/4，四个分支拼接后通道数恰为 in_channels
        self.avgpools = nn.ModuleList()  # 存放各分支自适应平均池化层的模块列表
        self.convs = nn.ModuleList()  # 存放各分支 1x1 卷积的模块列表
        for size in sizes:  # 遍历每个金字塔分支的目标尺寸
            self.avgpool.append(nn.AdaptiveAvgPool2d(size))  # 【注意】bug：属性名缺少复数 s（应为 self.avgpools.append），运行到此会因 self.avgpool 未定义/无法动态注册而抛出 AttributeError
            self.convs.append(_ConvBNReLU(in_channels, out_channels, 1, **kwargs))  # 各分支 1x1 卷积降维: (N,C,size,size) -> (N,C/4,size,size)

    def forward(self, x):  # 前向传播：x 形状 (N, in_channels, H, W)
        size = x.size()[2:]  # 记录输入空间尺寸 (H, W)，供各分支上采样对齐
        feats = [x]  # 特征收集列表，先放入原始输入（保留细节信息）
        for (avgpool, conv) in enumerate(zip(self.avgpools, self.convs)):  # 【注意】bug：enumerate 返回 (下标, 元组)，无法解包成 (avgpool, conv) 两个变量，正确写法应为 for avgpool, conv in zip(...)；且因上一处 bug，self.avgpools 恒为空列表
            feats.append(F.interpolate(conv(avgpool(x)), size, mode='bilinear', align_corners=True))  # 池化→卷积→双线性上采样回原尺寸: (N,C/4,size,size) -> (N,C/4,H,W)
        return torch.cat(feats, dim=1)  # 沿通道维拼接原始特征与四个分支: (N, C + 4*(C/4), H, W) = (N, 2C, H, W)


# -----------------------------------------------------------------
#                      For MobileNet
# -----------------------------------------------------------------
# （中文补充）以下为 MobileNet 专用模块
class _DepthwiseConv(nn.Module):  # MobileNet 的深度卷积块 conv_dw：Depthwise 3x3 + Pointwise 1x1
    """conv_dw in MobileNet"""  # 原有 docstring：MobileNet 中的深度卷积块

    # 形参说明: in_channels/out_channels 输入输出通道数; stride 步长(1 或 2); norm_layer 归一化层构造函数
    def __init__(self, in_channels, out_channels, stride, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DepthwiseConv, self).__init__()  # 初始化父类 nn.Module
        self.conv = nn.Sequential(  # 将两步卷积按序组装为 Sequential
            _ConvBNReLU(in_channels, in_channels, 3, stride, 1, groups=in_channels, norm_layer=norm_layer),  # Depthwise: 3x3 逐通道卷积(groups=通道数)负责空间滤波，可能下采样: (N,Cin,H,W) -> (N,Cin,H/stride,W/stride)
            _ConvBNReLU(in_channels, out_channels, 1, norm_layer=norm_layer))  # Pointwise: 1x1 卷积负责跨通道信息融合与通道数变换: -> (N,Cout,H/stride,W/stride)

    def forward(self, x):  # 前向传播：x 形状 (N, in_channels, H, W)
        return self.conv(x)  # 依次通过 depthwise 与 pointwise 卷积，输出 (N, out_channels, H/stride, W/stride)


# -----------------------------------------------------------------
#                      For MobileNetV2
# -----------------------------------------------------------------
# （中文补充）以下为 MobileNetV2 专用模块
class InvertedResidual(nn.Module):  # MobileNetV2 倒残差块：细-粗-细结构（与 ResNet 残差的粗-细-粗相反）
    # 形参说明: in_channels/out_channels 输入输出通道数; stride 步长(仅允许 1 或 2)
    # 形参说明: expand_ratio 中间升维倍率; norm_layer 归一化层构造函数; **kwargs 兼容额外参数
    def __init__(self, in_channels, out_channels, stride, expand_ratio, norm_layer=nn.BatchNorm2d, **kwargs):
        super(InvertedResidual, self).__init__()  # 初始化父类 nn.Module
        assert stride in [1, 2]  # 断言步长只能是 1（保持分辨率）或 2（下采样）
        self.use_res_connect = stride == 1 and in_channels == out_channels  # 残差捷径条件：不下采样且输入输出通道数相同

        layers = list()  # 临时列表，收集本块的子层
        inter_channels = int(round(in_channels * expand_ratio))  # 升维后的中间通道数 = round(in_channels * expand_ratio)
        if expand_ratio != 1:  # 升维倍率不为 1 时才需要 1x1 升维卷积
            # pw
            layers.append(_ConvBNReLU(in_channels, inter_channels, 1, relu6=True, norm_layer=norm_layer))  # （原注释 pw 保留）逐点卷积升维 + ReLU6: (N,Cin,H,W) -> (N,Cm,H,W)
        layers.extend([  # 一次性追加后续子层
            # dw
            _ConvBNReLU(inter_channels, inter_channels, 3, stride, 1,  # （原注释 dw 保留）3x3 深度卷积起始于本行
                        groups=inter_channels, relu6=True, norm_layer=norm_layer),  # groups=Cm 逐通道卷积 + ReLU6，本块唯一下采样点: (N,Cm,H,W) -> (N,Cm,H/stride,W/stride)
            # pw-linear
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),  # （原注释 pw-linear 保留）1x1 逐点卷积降维，线性输出不加激活: -> (N,Cout,H',W')
            norm_layer(out_channels)])  # 降维后仅接 BN（"线性瓶颈"设计，保留信息）
        self.conv = nn.Sequential(*layers)  # 将子层按序组装为 Sequential

    def forward(self, x):  # 前向传播：x 形状 (N, in_channels, H, W)
        if self.use_res_connect:  # 满足残差连接条件
            return x + self.conv(x)  # 输出 = 恒等捷径 x + 主分支输出（逐元素相加，通道与空间尺寸一致）
        else:  # 不满足残差连接条件（stride≠1 或通道数不一致）分支
            return self.conv(x)  # 形状不匹配时直接返回主分支输出


if __name__ == '__main__':  # 直接运行本文件时执行的简易自测代码
    x = torch.randn(1, 32, 64, 64)  # 构造随机输入张量 (N=1, C=32, H=64, W=64)
    model = InvertedResidual(32, 64, 2, 1)  # 实例化倒残差块：32→64 通道，stride=2 下采样，expand_ratio=1（不升维）
    out = model(x)  # 前向推理，输出形状 (1, 64, 32, 32)
