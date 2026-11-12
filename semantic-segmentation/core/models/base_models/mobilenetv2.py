# ============================================================================
# 【文件说明】mobilenetv2.py —— MobileNet V1 / V2 轻量级骨干网络
#     定义 MobileNet（V1：深度可分离卷积堆叠）与 MobileNetV2（倒残差 + 线性瓶颈）
#     两个网络类，并提供多种宽度倍数（1.0 / 0.75 / 0.5 / 0.25）的工厂函数。
# 【核心思想】V1：把标准卷积拆成"逐通道 3×3 深度卷积 + 1×1 逐点卷积"
#     （_DepthwiseConv），大幅降低参数量与计算量；V2：倒残差块 InvertedResidual
#     ——与 ResNet 残差"两端宽中间窄"相反，先 1×1 升维（扩张倍数 t，最大 6），
#     再 3×3 深度卷积（stride 控制是否下采样），最后 1×1 降维；降维层之后
#     不加激活（"线性瓶颈"，避免 ReLU/ReLU6 在低维空间损失信息）；
#     仅当 stride=1 且输入输出通道一致时才有残差跳跃连接。
# 【工程位置】core/models/base_models/ 下的骨干网络之一；上层分割模型经
#     get_mobilenet / get_mobilenet_v2（或 mobilenet*_x 工厂函数）创建，
#     其 features 各阶段的输出被用作编码器多尺度特征。
# ============================================================================
"""MobileNet and MobileNetV2."""  # 原模块英文 docstring（原样保留）
import torch  # PyTorch 主库（模块类型与 __main__ 自测使用）
import torch.nn as nn  # 神经网络模块：Conv2d/BatchNorm2d/Linear/Sequential/Dropout2d 等

from core.nn import _ConvBNReLU, _DepthwiseConv, InvertedResidual  # 公共层：Conv+BN+ReLU、深度可分离卷积（DW+PW）、MobileNetV2 倒残差块（均定义于 core/nn/basic.py）

__all__ = ['MobileNet', 'MobileNetV2', 'get_mobilenet', 'get_mobilenet_v2',  # 模块公开符号清单（限制 from ... import * 的范围）
           'mobilenet1_0', 'mobilenet_v2_1_0', 'mobilenet0_75', 'mobilenet_v2_0_75',  # V1/V2 宽度倍数 1.0 与 0.75 的工厂函数
           'mobilenet0_5', 'mobilenet_v2_0_5', 'mobilenet0_25', 'mobilenet_v2_0_25']  # V1/V2 宽度倍数 0.5 与 0.25 的工厂函数


# MobileNet V1：深度可分离卷积版本的轻量分类网络（本工程中用作分割骨干之一）
class MobileNet(nn.Module):
    # 参数：num_classes 分类数（默认 ImageNet 1000 类）；multiplier 通道宽度倍数（<1 时压缩通道）；
    # norm_layer 归一化层构造器；**kwargs 兼容上层传入的额外参数
    def __init__(self, num_classes=1000, multiplier=1.0, norm_layer=nn.BatchNorm2d, **kwargs):
        super(MobileNet, self).__init__()  # 初始化 nn.Module 基类（完成子模块注册）
        conv_dw_setting = [  # 深度卷积各阶段配置，每行为 [输出通道 c, 重复块数 n, 首块步长 s]
            [64, 1, 1],  # 1 个块、64 通道、不下采样（停留在 1/2 分辨率）
            [128, 2, 2],  # 2 个块、128 通道、首块下采样到 1/4
            [256, 2, 2],  # 2 个块、256 通道、下采样到 1/8
            [512, 6, 2],  # 6 个块、512 通道、下采样到 1/16（特征主力阶段）
            [1024, 2, 2]]  # 2 个块、1024 通道、下采样到 1/32
        input_channels = int(32 * multiplier) if multiplier > 1.0 else 32  # 首层输出通道：仅当倍数 >1 时放大，缩小时保持 32（与官方实现一致）
        features = [_ConvBNReLU(3, input_channels, 3, 2, 1, norm_layer=norm_layer)]  # 首层标准 3×3 卷积（stride=2）：(N,3,H,W) → (N,32,H/2,W/2)

        for c, n, s in conv_dw_setting:  # 遍历每个深度卷积阶段
            out_channels = int(c * multiplier)  # 该阶段输出通道按宽度倍数缩放
            for i in range(n):  # 阶段内堆叠 n 个深度可分离卷积块
                stride = s if i == 0 else 1  # 仅阶段内第一个块用步长 s（负责下采样），其余为 1
                features.append(_DepthwiseConv(input_channels, out_channels, stride, norm_layer))  # 深度卷积 + 逐点卷积：stride=2 时空间减半
                input_channels = out_channels  # 更新下一块的输入通道
        features.append(nn.AdaptiveAvgPool2d(1))  # 全局自适应平均池化：(N,C,H',W') → (N,C,1,1)
        self.features = nn.Sequential(*features)  # 串联为完整特征提取器

        self.classifier = nn.Linear(int(1024 * multiplier), num_classes)  # 分类全连接：(N,1024×multiplier) → (N,num_classes)

        # weight initialization
        for m in self.modules():  # 遍历全部子模块统一初始化权重
            if isinstance(m, nn.Conv2d):  # 卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out')  # Kaiming 正态初始化（按输出扇出 fan_out）
                if m.bias is not None:  # 若带偏置
                    nn.init.zeros_(m.bias)  # 偏置置零
            elif isinstance(m, nn.BatchNorm2d):  # BN 层
                nn.init.ones_(m.weight)  # 缩放因子 γ 初始化为 1
                nn.init.zeros_(m.bias)  # 平移因子 β 初始化为 0
            elif isinstance(m, nn.Linear):  # 全连接层
                nn.init.normal_(m.weight, 0, 0.01)  # 权重按 N(0, 0.01²) 初始化
                nn.init.zeros_(m.bias)  # 偏置置零

    # 前向传播：x (N,3,H,W) → 分类 logits (N,num_classes)
    def forward(self, x):
        x = self.features(x)  # 特征提取 + 全局池化：(N,3,H,W) → (N,C,1,1)
        x = self.classifier(x.view(x.size(0), x.size(1)))  # (N,C,1,1) 展平为 (N,C) 后全连接分类 → (N,num_classes)
        return x  # 返回分类 logits


# MobileNetV2：倒残差 + 线性瓶颈（本工程轻量分割任务的主力骨干）
class MobileNetV2(nn.Module):
    # 参数：num_classes 分类数；multiplier 通道宽度倍数；norm_layer 归一化层；**kwargs 兼容上层额外参数
    def __init__(self, num_classes=1000, multiplier=1.0, norm_layer=nn.BatchNorm2d, **kwargs):
        super(MobileNetV2, self).__init__()  # 初始化 nn.Module 基类
        inverted_residual_setting = [  # 倒残差块配置，每行为 [扩张倍数 t, 输出通道 c, 块数 n, 首块步长 s]
            # t, c, n, s
            [1, 16, 1, 1],  # t=1 不升维（输入仅 32 维）：输出 16 通道，1/2 → 1/4
            [6, 24, 2, 2],  # t=6 升维 6 倍：输出 24 通道，下采样到 1/8
            [6, 32, 3, 2],  # 输出 32 通道，下采样到 1/16
            [6, 64, 4, 2],  # 输出 64 通道，下采样到 1/32
            [6, 96, 3, 1],  # 输出 96 通道，保持 1/32（不下采样，深层细化）
            [6, 160, 3, 2],  # 输出 160 通道，下采样到 1/64
            [6, 320, 1, 1]]  # 输出 320 通道，保持 1/64
        # building first layer
        input_channels = int(32 * multiplier) if multiplier > 1.0 else 32  # 首层输出通道：仅放大时缩放（缩小保持 32）
        last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280  # 末端 1×1 卷积输出通道（基准 1280）
        features = [_ConvBNReLU(3, input_channels, 3, 2, 1, relu6=True, norm_layer=norm_layer)]  # 首层 3×3 卷积（stride=2、ReLU6 激活）：(N,3,H,W) → (N,32,H/2,W/2)

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:  # 遍历倒残差配置逐阶段构建
            out_channels = int(c * multiplier)  # 该阶段输出通道按宽度倍数缩放
            for i in range(n):  # 阶段内堆叠 n 个倒残差块
                stride = s if i == 0 else 1  # 仅首块用步长 s；stride=1 且通道一致时块内才启用残差连接
                features.append(InvertedResidual(input_channels, out_channels, stride, t, norm_layer))  # 倒残差块：1×1 升维(t 倍) → 3×3 深度卷积 → 1×1 线性降维（无激活）
                input_channels = out_channels  # 更新输入通道
        # building last several layers
        features.append(_ConvBNReLU(input_channels, last_channels, 1, relu6=True, norm_layer=norm_layer))  # 末端 1×1 卷积升维到 1280（ReLU6）：(N,320,H',W') → (N,1280,H',W')
        features.append(nn.AdaptiveAvgPool2d(1))  # 全局自适应平均池化：(N,1280,H',W') → (N,1280,1,1)
        self.features = nn.Sequential(*features)  # 串联为特征提取器

        self.classifier = nn.Sequential(  # 分类头
            nn.Dropout2d(0.2),  # 随机失活【注意】可疑写法：此处输入已是 2D (N,C)，Dropout2d 面向 4D 特征图设计，效果等价于普通 Dropout
            nn.Linear(last_channels, num_classes))  # 全连接分类：(N,1280) → (N,num_classes)

        # weight initialization
        for m in self.modules():  # 遍历全部子模块初始化权重
            if isinstance(m, nn.Conv2d):  # 卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out')  # Kaiming 正态初始化（fan_out）
                if m.bias is not None:  # 若带偏置
                    nn.init.zeros_(m.bias)  # 偏置置零
            elif isinstance(m, nn.BatchNorm2d):  # BN 层
                nn.init.ones_(m.weight)  # γ 置 1
                nn.init.zeros_(m.bias)  # β 置 0
            elif isinstance(m, nn.Linear):  # 全连接层
                nn.init.normal_(m.weight, 0, 0.01)  # 权重按 N(0, 0.01²) 初始化
                if m.bias is not None:  # 若带偏置
                    nn.init.zeros_(m.bias)  # 偏置置零

    # 前向传播：x (N,3,H,W) → 分类 logits (N,num_classes)
    def forward(self, x):
        x = self.features(x)  # 特征提取 + 全局池化：(N,3,H,W) → (N,1280,1,1)
        x = self.classifier(x.view(x.size(0), x.size(1)))  # 展平为 (N,1280) → Dropout → 全连接 → (N,num_classes)
        return x  # 返回分类 logits


# Constructor
# MobileNet V1 通用构造器：multiplier 宽度倍数；pretrained 预训练开关；root 权重目录（本函数中未使用）
def get_mobilenet(multiplier=1.0, pretrained=False, root='~/.torch/models', **kwargs):
    model = MobileNet(multiplier=multiplier, **kwargs)  # 按宽度倍数构造 MobileNet V1

    if pretrained:  # 若请求预训练权重
        raise ValueError("Not support pretrained")  # 本工程未提供 V1 预训练权重，直接抛异常
    return model  # 返回随机初始化的模型


# MobileNetV2 通用构造器：参数含义同 get_mobilenet（root 同样未被使用）
def get_mobilenet_v2(multiplier=1.0, pretrained=False, root='~/.torch/models', **kwargs):
    model = MobileNetV2(multiplier=multiplier, **kwargs)  # 按宽度倍数构造 MobileNetV2

    if pretrained:  # 若请求预训练权重
        raise ValueError("Not support pretrained")  # 本工程未提供 V2 预训练权重，直接抛异常
    return model  # 返回随机初始化的模型


def mobilenet1_0(**kwargs):  # V1 宽度倍数 1.0（标准宽度）工厂函数
    return get_mobilenet(1.0, **kwargs)  # 委托通用构造器


def mobilenet_v2_1_0(**kwargs):  # V2 宽度倍数 1.0（标准宽度）
    return get_mobilenet_v2(1.0, **kwargs)  # 委托通用构造器


def mobilenet0_75(**kwargs):  # V1 宽度倍数 0.75
    return get_mobilenet(0.75, **kwargs)  # 委托通用构造器


def mobilenet_v2_0_75(**kwargs):  # V2 宽度倍数 0.75
    return get_mobilenet_v2(0.75, **kwargs)  # 委托通用构造器


def mobilenet0_5(**kwargs):  # V1 宽度倍数 0.5
    return get_mobilenet(0.5, **kwargs)  # 委托通用构造器


def mobilenet_v2_0_5(**kwargs):  # V2 宽度倍数 0.5
    return get_mobilenet_v2(0.5, **kwargs)  # 委托通用构造器


def mobilenet0_25(**kwargs):  # V1 宽度倍数 0.25
    return get_mobilenet(0.25, **kwargs)  # 委托通用构造器


def mobilenet_v2_0_25(**kwargs):  # V2 宽度倍数 0.25
    return get_mobilenet_v2(0.25, **kwargs)  # 委托通用构造器


if __name__ == '__main__':  # 直接运行本文件时做实例化自测
    model = mobilenet0_5()  # 构造 0.5 倍宽度的 MobileNet V1（仅实例化，不做前向）
