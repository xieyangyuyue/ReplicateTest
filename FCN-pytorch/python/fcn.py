# -*- coding: utf-8 -*-
# ============================================================================
# 【文件说明】fcn.py —— FCN（Fully Convolutional Networks，全卷积网络）模型定义
#
# 本文件是整个项目的核心，实现了语义分割经典论文《Fully Convolutional
# Networks for Semantic Segmentation》(CVPR 2015) 中的四种网络结构：
#   1. FCN32s —— 只用最深一层特征（pool5），一次性上采样 32 倍；
#   2. FCN16s —— 融合 pool4 的跳跃连接（细节更丰富）；
#   3. FCN8s  —— 再融合 pool3 的跳跃连接（效果最好的一版）；
#   4. FCNs   —— 融合 pool1~pool5 全部五层特征；
# 以及作为编码器（backbone / 特征提取器）的 VGGNet 类。
#
# 【核心思想】
#   - 把分类网络 VGG 的全连接层改造成 1×1 卷积 → 网络可接受任意尺寸输入，
#     输出与输入等大的"逐像素类别得分图"（密集预测）；
#   - 深层特征语义强但分辨率低 → 用转置卷积（俗称反卷积）逐级上采样恢复分辨率；
#   - 浅层特征分辨率高、细节丰富 → 通过"跳跃连接"与深层上采样特征逐元素相加。
# ============================================================================

from __future__ import print_function  # 向 Python2 兼容 print() 函数式写法

import torch                     # PyTorch 主库：张量运算与自动求导
import torch.nn as nn            # 神经网络模块：层、容器、损失函数（nn.Module 等）
import torch.optim as optim      # 优化器模块（本文件末尾自测代码用到 SGD）
from torchvision import models   # torchvision 预训练模型库：用于下载 VGG 的 ImageNet 权重
from torchvision.models.vgg import VGG  # torchvision 官方 VGG 模型基类，下方 VGGNet 继承它


class FCN32s(nn.Module):  # FCN-32s 网络：仅用 pool5（缩小 32 倍）的特征，连续上采样 32 倍恢复原尺寸

    def __init__(self, pretrained_net, n_class):  # 构造参数：预训练 VGG 编码器、分割类别数
        super().__init__()                # 调用父类 nn.Module 的构造函数（注册子模块所必需）
        self.n_class = n_class            # 保存类别数，供最后的分类卷积层确定输出通道数
        self.pretrained_net = pretrained_net  # 保存 VGG 编码器，forward 时用它提取各阶段特征
        self.relu    = nn.ReLU(inplace=True)  # ReLU 激活函数；inplace=True 原地计算，节省显存
        # ---------- 解码器（上采样部分）：5 组 [转置卷积 + BN] ----------
        # 转置卷积第 1 层：512→512 通道；3×3 卷积核，stride=2 将空间尺寸放大 2 倍；
        # padding=1 与 output_padding=1 配合，保证放大后尺寸恰好翻倍（H/32 → H/16）
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)  # 第 1 层转置卷积后的批归一化（512 个通道）
        # 转置卷积第 2 层：512→256 通道，空间尺寸 H/16 → H/8
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)  # 第 2 层的批归一化（256 个通道）
        # 转置卷积第 3 层：256→128 通道，空间尺寸 H/8 → H/4
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)  # 第 3 层的批归一化（128 个通道）
        # 转置卷积第 4 层：128→64 通道，空间尺寸 H/4 → H/2
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)   # 第 4 层的批归一化（64 个通道）
        # 转置卷积第 5 层：64→32 通道，空间尺寸 H/2 → H（恢复到原图大小）
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)   # 第 5 层的批归一化（32 个通道）
        # 分类器：1×1 卷积，把每个像素位置 32 维特征映射为 n_class 维得分（逐像素分类）
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):  # 前向传播：x 为输入图像张量，形状 (N, 3, H, W)
        output = self.pretrained_net(x)  # 用 VGG 编码器提取特征，返回含 x1~x5 各阶段特征的字典
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)  # 取最深层（pool5 后）特征：512 通道、分辨率 1/32

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)  # 上采样×2 → 1/16，再 ReLU、BN
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)   # 继续上采样 → 1/8
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)   # → 1/4
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)    # → 1/2
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)        # → 原图分辨率
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)  # 1×1 卷积输出各类得分图

        return score  # size=(N, n_class, x.H/1, x.W/1)  # 返回得分图：每个像素对每一类各有一个分数


class FCN16s(nn.Module):  # FCN-16s 网络：融合 pool4 特征（跳跃连接），从 1/16 分辨率开始精化

    def __init__(self, pretrained_net, n_class):  # 构造参数：预训练 VGG 编码器、分割类别数
        super().__init__()                # 调用父类构造函数
        self.n_class = n_class            # 保存类别数
        self.pretrained_net = pretrained_net  # 保存 VGG 编码器
        self.relu    = nn.ReLU(inplace=True)  # ReLU 激活（原地操作省显存）
        # 转置卷积第 1 层：512→512 通道，尺寸放大 2 倍（H/32 → H/16）
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)  # 第 1 层的批归一化
        # 转置卷积第 2 层：512→256 通道，尺寸再放大 2 倍
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)  # 第 2 层的批归一化
        # 转置卷积第 3 层：256→128 通道
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)  # 第 3 层的批归一化
        # 转置卷积第 4 层：128→64 通道
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)   # 第 4 层的批归一化
        # 转置卷积第 5 层：64→32 通道
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)   # 第 5 层的批归一化
        # 分类器：1×1 卷积输出 n_class 维逐像素得分
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):  # 前向传播：x 形状 (N, 3, H, W)
        output = self.pretrained_net(x)  # 编码器提取各阶段特征
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)  # pool5 特征：最深、语义最强、分辨率最低
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)  # pool4 特征：比 x5 浅一层，用于跳跃连接

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)  # x5 上采样×2，与 x4 尺寸对齐
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)  # 跳跃连接：与 x4 逐元素相加后做 BN
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)   # 继续上采样 → 1/8
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)   # → 1/4
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)    # → 1/2
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)        # → 原图分辨率
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)  # 输出各类得分图

        return score  # size=(N, n_class, x.H/1, x.W/1)  # 返回得分图


class FCN8s(nn.Module):  # FCN-8s 网络：同时融合 pool4 与 pool3 特征，论文中效果最好的一版

    def __init__(self, pretrained_net, n_class):  # 构造参数：预训练 VGG 编码器、分割类别数
        super().__init__()                # 调用父类构造函数
        self.n_class = n_class            # 保存类别数
        self.pretrained_net = pretrained_net  # 保存 VGG 编码器
        self.relu    = nn.ReLU(inplace=True)  # ReLU 激活（原地操作）
        # 转置卷积第 1 层：512→512 通道，尺寸放大 2 倍
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)  # 第 1 层的批归一化
        # 转置卷积第 2 层：512→256 通道，尺寸放大 2 倍
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)  # 第 2 层的批归一化
        # 转置卷积第 3 层：256→128 通道
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)  # 第 3 层的批归一化
        # 转置卷积第 4 层：128→64 通道
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)   # 第 4 层的批归一化
        # 转置卷积第 5 层：64→32 通道
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)   # 第 5 层的批归一化
        # 分类器：1×1 卷积输出 n_class 维逐像素得分
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):  # 前向传播：x 形状 (N, 3, H, W)
        output = self.pretrained_net(x)  # 编码器提取各阶段特征
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)  # pool5 特征（1/32 分辨率）
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)  # pool4 特征（1/16 分辨率）
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)   # pool3 特征（1/8 分辨率）

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)  # x5 上采样×2
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)  # 第一次跳跃连接：+x4
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)   # 再上采样×2
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)  # 第二次跳跃连接：+x3
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)   # → 1/4
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)    # → 1/2
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)        # → 原图分辨率
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)  # 输出各类得分图

        return score  # size=(N, n_class, x.H/1, x.W/1)  # 返回得分图


class FCNs(nn.Module):  # FCNs 网络：融合 pool1~pool5 全部五层跳跃连接的完整版

    def __init__(self, pretrained_net, n_class):  # 构造参数：预训练 VGG 编码器、分割类别数
        super().__init__()                # 调用父类构造函数
        self.n_class = n_class            # 保存类别数
        self.pretrained_net = pretrained_net  # 保存 VGG 编码器
        self.relu    = nn.ReLU(inplace=True)  # ReLU 激活（原地操作）
        # 转置卷积第 1 层：512→512 通道，尺寸放大 2 倍
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)  # 第 1 层的批归一化
        # 转置卷积第 2 层：512→256 通道
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)  # 第 2 层的批归一化
        # 转置卷积第 3 层：256→128 通道
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)  # 第 3 层的批归一化
        # 转置卷积第 4 层：128→64 通道
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)   # 第 4 层的批归一化
        # 转置卷积第 5 层：64→32 通道
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)   # 第 5 层的批归一化
        # 分类器：1×1 卷积输出 n_class 维逐像素得分
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):  # 前向传播：x 形状 (N, 3, H, W)
        output = self.pretrained_net(x)  # 编码器提取各阶段特征
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)  # pool5 特征（1/32 分辨率、512 通道）
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)  # pool4 特征（1/16 分辨率、512 通道）
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)   # pool3 特征（1/8 分辨率、256 通道）
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)   # pool2 特征（1/4 分辨率、128 通道）
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)    # pool1 特征（1/2 分辨率、64 通道）

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)  # x5 上采样×2 后 BN
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)  # 跳跃连接：+x4
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)   # 上采样×2 后 BN
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)  # 跳跃连接：+x3
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)   # 上采样×2 后 BN
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)  # 跳跃连接：+x2
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)    # 上采样×2 后 BN
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)  # 跳跃连接：+x1
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)        # 上采样×2 → 原图分辨率
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)  # 输出各类得分图

        return score  # size=(N, n_class, x.H/1, x.W/1)  # 返回得分图


class VGGNet(VGG):  # VGG 特征提取网络：继承 torchvision 的 VGG，改造 forward 以输出各池化阶段特征
    # 构造参数：pretrained=是否加载 ImageNet 预训练权重；model=VGG 型号（vgg11/13/16/19）；
    #           requires_grad=参数是否可训练（False=冻结编码器）；remove_fc=是否删除全连接层；show_params=是否打印参数
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))  # 用 cfg 配置构建卷积层序列，初始化父类 VGG
        self.ranges = ranges[model]  # 记录该型号 VGG 每个阶段（相邻池化层之间）的层索引范围

        if pretrained:  # 若需要加载预训练权重
            # 【新环境适配】原代码：exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)
            # 其中 pretrained=True 是 torchvision<0.13 的旧参数（新版仅告警、未来会移除），
            # 且 exec 拼接代码不便于阅读。此处改为等价的新旧兼容写法：
            # 优先用 weights 枚举（torchvision>=0.13），旧版回退 pretrained=True。
            # VGG16_Weights.IMAGENET1K_V1 与原 pretrained=True 加载的是同一份权重。
            weights_cls = getattr(models, model.upper() + '_Weights', None)  # 如 models.VGG16_Weights（新版才存在，旧版返回 None）
            if weights_cls is not None:  # torchvision>=0.13：新 API
                vgg_pretrained = getattr(models, model)(weights=weights_cls.IMAGENET1K_V1)  # 下载并构建带 ImageNet 权重的 VGG
            else:  # 旧版 torchvision：回退到旧参数写法
                vgg_pretrained = getattr(models, model)(pretrained=True)  # 旧 API（torchvision<0.13）
            self.load_state_dict(vgg_pretrained.state_dict())  # 把 ImageNet 权重复制进当前网络（迁移学习）

        if not requires_grad:  # 若要冻结编码器参数（只训练解码器时）
            for param in super().parameters():  # 遍历父类（编码器）的所有参数
                param.requires_grad = False      # 关闭梯度，反向传播时不更新

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier  # 删除 VGG 原始的全连接分类头：分割任务用不到，可省显存

        if show_params:  # 若需要查看参数详情
            for name, param in self.named_parameters():  # 遍历所有具名参数
                print(name, param.size())                 # 打印参数名和形状

    def forward(self, x):  # 前向传播：改造后的 VGG，返回各池化阶段的特征字典
        output = {}  # 用字典保存 5 个池化阶段的输出

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        # 获取每个最大池化层的输出（VGG 网络共有 5 个池化阶段）
        for idx in range(len(self.ranges)):  # 遍历 5 个阶段
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):  # 执行该阶段索引范围内的所有层
                x = self.features[layer](x)  # 逐层前向（卷积/ReLU/池化）
            output["x%d"%(idx+1)] = x  # 以 'x1'~'x5' 为键，保存该阶段（池化后）的特征图

        return output  # 返回 {'x1':..., 'x2':..., 'x3':..., 'x4':..., 'x5':...}


ranges = {  # 各型号 VGG 的"阶段→层索引区间"表：每个元组覆盖 从上一池化层之后 到 下一池化层（含） 的层
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),   # vgg11：共 21 层（8 卷积+8 ReLU+5 池化）
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),   # vgg13：共 25 层（10 卷积+10 ReLU+5 池化）
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),   # vgg16：共 31 层（13 卷积+13 ReLU+5 池化）
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))    # vgg19：共 37 层（16 卷积+16 ReLU+5 池化）
}  # ranges 字典定义结束

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# 从 torchvision 官方源码裁剪来的 VGG 结构配置：数字=卷积输出通道数，'M'=2×2 最大池化层
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],                                # 8 卷积 + 5 池化
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],                       # 10 卷积 + 5 池化
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],        # 13 卷积 + 5 池化
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],  # 16 卷积 + 5 池化
}  # cfg 字典定义结束

def make_layers(cfg, batch_norm=False):  # 根据 cfg 配置构建 VGG 的卷积层序列；batch_norm=是否插入 BN 层
    layers = []              # 层列表容器
    in_channels = 3          # 当前输入通道数（RGB 图像初始为 3）
    for v in cfg:            # 逐项遍历配置列表
        if v == 'M':         # 遇到 'M'：插入 2×2、步长 2 的最大池化（空间尺寸减半）
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 向层列表添加最大池化层
        else:                # 遇到数字：插入输出通道为 v 的 3×3 卷积（padding=1 保持尺寸不变）
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)  # 创建当前层的卷积
            if batch_norm:   # 若启用 BN：卷积 → BN → ReLU
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]  # 三个层一起加入列表
            else:            # 否则：卷积 → ReLU
                layers += [conv2d, nn.ReLU(inplace=True)]  # 两个层一起加入列表
            in_channels = v  # 下一层的输入通道数 = 当前卷积的输出通道数
    return nn.Sequential(*layers)  # 把层列表打包为顺序执行容器并返回


if __name__ == "__main__":  # 直接运行本文件时执行以下自测代码
    batch_size, n_class, h, w = 10, 20, 160, 160  # 测试超参：批 10、20 类、输入 160×160

    # test output size
    vgg_model = VGGNet(requires_grad=True)  # 构建 VGG 编码器（默认 vgg16、加载预训练权重、可训练）
    input = torch.autograd.Variable(torch.randn(batch_size, 3, 224, 224))  # 随机输入 (10,3,224,224)；Variable 为旧版包装，新版直接用 Tensor
    output = vgg_model(input)  # 前向传播得到特征字典
    assert output['x5'].size() == torch.Size([batch_size, 512, 7, 7])  # 断言最深层尺寸：224/32=7 → (10,512,7,7)

    fcn_model = FCN32s(pretrained_net=vgg_model, n_class=n_class)  # 构建 FCN32s 模型
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))  # 随机输入 (10,3,160,160)
    output = fcn_model(input)  # 前向传播
    assert output.size() == torch.Size([batch_size, n_class, h, w])  # 断言输出与输入等大（全卷积的特点）

    fcn_model = FCN16s(pretrained_net=vgg_model, n_class=n_class)  # 构建 FCN16s 模型
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))  # 随机输入
    output = fcn_model(input)  # 前向传播
    assert output.size() == torch.Size([batch_size, n_class, h, w])  # 断言输出尺寸正确

    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class)  # 构建 FCN8s 模型
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))  # 随机输入
    output = fcn_model(input)  # 前向传播
    assert output.size() == torch.Size([batch_size, n_class, h, w])  # 断言输出尺寸正确

    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)  # 构建 FCNs 模型
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))  # 随机输入
    output = fcn_model(input)  # 前向传播
    assert output.size() == torch.Size([batch_size, n_class, h, w])  # 断言输出尺寸正确

    print("Pass size check")  # 所有尺寸检查通过

    # test a random batch, loss should decrease
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)  # 重建 FCNs 用于训练自测
    criterion = nn.BCELoss()  # 二值交叉熵损失（需先过 sigmoid）
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)  # SGD 优化器：学习率 0.001、动量 0.9
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))  # 固定一份随机输入
    y = torch.autograd.Variable(torch.randn(batch_size, n_class, h, w), requires_grad=False)  # 随机"标签"（不需要梯度）
    for iter in range(10):  # 迭代 10 轮，观察损失是否下降
        optimizer.zero_grad()  # 梯度清零
        output = fcn_model(input)  # 前向传播
        output = torch.sigmoid(output)  # 【新环境适配】sigmoid 激活（原代码 nn.functional.sigmoid 为旧写法，新版已弃用，改用 torch.sigmoid）
        loss = criterion(output, y)  # 计算损失
        loss.backward()  # 反向传播求梯度
        print("iter{}, loss {}".format(iter, loss.item()))  # 【新环境适配】打印损失（原 loss.data[0] 在新版报 IndexError，改用 .item()）
        optimizer.step()  # 更新参数
