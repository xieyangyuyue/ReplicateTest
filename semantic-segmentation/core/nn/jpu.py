# ============================================================================
# 【文件说明】jpu.py —— JPU（Joint Pyramid Upsampling，联合金字塔上采样）模块，
#             出自 FastFCN 论文（Rethinking Dilated Convolution in the Backbone
#             for Semantic Segmentation），作为 FPN 式解码头的替代，接在骨干网络
#             之后做多尺度特征融合（本仓库中由 core/models/segbase.py 在
#             jpu=True 时启用，输入 ResNet 的 c1/c2/c3/c4 四阶段特征）。
# 【核心思想】把骨干网最后三个 stage（1/8、1/16、1/32 分辨率）的特征分别用 3x3
#             卷积统一到 width 通道，全部上采样对齐到 1/8 尺寸后拼接；再并行送入
#             四个空洞率分别为 1/2/4/8 的深度可分离卷积分支提取多尺度上下文并拼接
#             输出。相比逐级反卷积/FPN 的上采样，"一次对齐 + 并行空洞卷积"显著
#             降低了计算量与显存占用。
# ============================================================================
"""Joint Pyramid Upsampling"""  # 原有 docstring：联合金字塔上采样
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络子模块（Conv2d、BatchNorm2d、Sequential 等）
import torch.nn.functional as F  # 导入函数式接口（双线性插值 interpolate 等）

__all__ = ['JPU']  # 对外导出符号白名单（仅 JPU；SeparableConv2d 为内部实现，不导出）


class SeparableConv2d(nn.Module):  # 深度可分离卷积：Depthwise 卷积 + BN + Pointwise 1x1 卷积
    # 形参说明: inplanes 输入通道数; planes 输出通道数; kernel_size 卷积核尺寸(默认3)
    # 形参说明: stride 步长(默认1); padding 填充(默认1); dilation 空洞率(默认1)
    # 形参说明: bias 卷积偏置开关(默认False); norm_layer 归一化层构造函数(默认 BatchNorm2d)
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=False, norm_layer=nn.BatchNorm2d):  # 构造参数续行：空洞率/卷积偏置开关/归一化层
        super(SeparableConv2d, self).__init__()  # 初始化父类 nn.Module
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)  # Depthwise 逐通道卷积：groups=inplanes 使每个通道独立卷积，仅做空间滤波，通道数不变
        self.bn = norm_layer(inplanes)  # depthwise 卷积后的批归一化
        self.pointwise = nn.Conv2d(inplanes, planes, 1, bias=bias)  # Pointwise 1x1 卷积：跨通道线性组合，完成通道数变换 inplanes→planes

    def forward(self, x):  # 前向传播：x 形状 (N, inplanes, H, W)
        x = self.conv(x)  # depthwise 卷积: (N, inplanes, H, W) -> (N, inplanes, H', W')
        x = self.bn(x)  # 归一化: 形状不变
        x = self.pointwise(x)  # pointwise 卷积: (N, inplanes, H', W') -> (N, planes, H', W')
        return x  # 返回输出特征图


# copy from: https://github.com/wuhuikai/FastFCN/blob/master/encoding/nn/customize.py  # 原注释保留：代码出处为 FastFCN 官方仓库
# （中文补充）JPU 主模块：融合骨干网后三个 stage 的特征，输出多尺度上下文特征
class JPU(nn.Module):  # 联合金字塔上采样模块
    # 形参说明: in_channels 骨干网各 stage 的输出通道数列表（取后三个使用，如 ResNet 的 [512, 1024, 2048]）
    # 形参说明: width 统一后的通道宽度(默认512); norm_layer 归一化层构造函数; **kwargs 预留扩展参数
    def __init__(self, in_channels, width=512, norm_layer=nn.BatchNorm2d, **kwargs):
        super(JPU, self).__init__()  # 初始化父类 nn.Module

        self.conv5 = nn.Sequential(  # 处理最深 stage（in_channels[-1]，1/32 分辨率）的卷积块
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),  # 3x3 卷积把通道统一到 width，padding=1 保持空间尺寸
            norm_layer(width),  # 批归一化
            nn.ReLU(True))  # ReLU 激活（inplace=True）
        self.conv4 = nn.Sequential(  # 处理中间 stage（in_channels[-2]，1/16 分辨率）的卷积块
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),  # 3x3 卷积统一通道到 width
            norm_layer(width),  # 批归一化
            nn.ReLU(True))  # ReLU 激活
        self.conv3 = nn.Sequential(  # 处理较浅 stage（in_channels[-3]，1/8 分辨率）的卷积块
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),  # 3x3 卷积统一通道到 width
            norm_layer(width),  # 批归一化
            nn.ReLU(True))  # ReLU 激活

        self.dilation1 = nn.Sequential(  # 空洞分支 1：dilation=1，感受野 3x3，负责局部细节
            SeparableConv2d(3 * width, width, 3, padding=1, dilation=1, bias=False),  # 深度可分离卷积，padding=dilation 保持空间尺寸不变
            norm_layer(width),  # 批归一化
            nn.ReLU(True))  # ReLU 激活
        self.dilation2 = nn.Sequential(  # 空洞分支 2：dilation=2，中等感受野，捕获中尺度上下文
            SeparableConv2d(3 * width, width, 3, padding=2, dilation=2, bias=False),  # 深度可分离空洞卷积
            norm_layer(width),  # 批归一化
            nn.ReLU(True))  # ReLU 激活
        self.dilation3 = nn.Sequential(  # 空洞分支 3：dilation=4，较大感受野
            SeparableConv2d(3 * width, width, 3, padding=4, dilation=4, bias=False),  # 深度可分离空洞卷积
            norm_layer(width),  # 批归一化
            nn.ReLU(True))  # ReLU 激活
        self.dilation4 = nn.Sequential(  # 空洞分支 4：dilation=8，最大感受野，近似全局上下文
            SeparableConv2d(3 * width, width, 3, padding=8, dilation=8, bias=False),  # 深度可分离空洞卷积
            norm_layer(width),  # 批归一化
            nn.ReLU(True))  # ReLU 激活

    def forward(self, *inputs):  # 前向传播：inputs 为骨干网由浅到深的多 stage 特征（segbase.py 中传入 c1,c2,c3,c4 四个，实际使用后三个）
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]  # 后三个 stage 分别过对应卷积块统一到 width 通道，得到 [1/32, 1/16, 1/8] 三种分辨率的特征列表
        size = feats[-1].size()[2:]  # 以最浅的 1/8 特征（feats[-1]，即 conv3 输出）的空间尺寸为对齐基准
        feats[-2] = F.interpolate(feats[-2], size, mode='bilinear', align_corners=True)  # 1/16 特征双线性上采样 2 倍，对齐到 1/8 尺寸
        feats[-3] = F.interpolate(feats[-3], size, mode='bilinear', align_corners=True)  # 1/32 特征双线性上采样 4 倍，对齐到 1/8 尺寸
        feat = torch.cat(feats, dim=1)  # 通道维拼接三路特征: (N, 3*width, H/8, W/8)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)],  # 将 1/8 对齐后的拼接特征并行送入四个不同空洞率的卷积分支
                         dim=1)  # 四个不同空洞率的分支并行提取多尺度上下文后拼接: (N, 4*width, H/8, W/8)

        return inputs[0], inputs[1], inputs[2], feat  # 返回骨干网前三个原始特征 (c1,c2,c3) + JPU 融合特征 feat（供后续分割头使用）
