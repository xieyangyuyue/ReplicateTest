# ============================================================================
# 【文件说明】resnet.py —— 经典 ResNet 骨干网络（torchvision 风格实现），
#              提供 resnet18/34/50/101/152 工厂函数。在分割工程中作为编码器
#              （backbone）使用，例如 BiSENet 通过
#              `from core.models.base_models.resnet import resnet18` 取其
#              stage1~stage4 特征；作编码器时只前向到 layer4，不走末尾的
#              avgpool/fc 分类头。
# 【核心思想】残差学习：每个残差块输出 F(x) + x，恒等捷径让梯度直达浅层，
#              使上百层网络可训练。两种残差块：BasicBlock（3×3×2，用于
#              resnet18/34）与 Bottleneck（1×1 降维 → 3×3 卷积 → 1×1 升维
#              至 4 倍通道，用于 resnet50+）。下采样（stride=2）放在各 stage
#              首块 3×3 卷积上（即 ResNet v1.5 结构）。
# 【注意】     本实现不支持 dilated（空洞卷积）模式，输出步长固定为 32；
#              分割任务需要输出步长 8 的骨干时请改用 resnetv1b.py /
#              resnext.py 中的 dilated 变体。
# ============================================================================
import torch.nn as nn  # PyTorch 神经网络模块：层定义、容器、参数初始化等
import torch.utils.model_zoo as model_zoo  # 预训练权重下载工具【注意】load_url 在新版 PyTorch 已废弃，新代码建议用 torch.hub.load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',  # from xxx import * 时对外导出的符号清单（ResNet 类 + 各深度工厂函数）
           'resnet152']  # 清单续行：resnet152

model_urls = {  # 各深度 ResNet 对应的 ImageNet 预训练权重下载地址表（pytorch.org 官方托管，URL 尾串为校验哈希）
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',  # ResNet-18 权重地址
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',  # ResNet-34 权重地址
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',  # ResNet-50 权重地址
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',  # ResNet-101 权重地址
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',  # ResNet-152 权重地址
}  # 权重地址表结束


# ----------------------- conv3x3 参数说明 -----------------------
# in_planes：输入通道数；out_planes：输出通道数；stride：步长（>1 时该卷积承担下采样）
def conv3x3(in_planes, out_planes, stride=1):  # 封装带 padding 的 3×3 卷积（无 bias，后接 BN）
    """3x3 convolution with padding"""  # 原英文 docstring 保留：带填充的 3×3 卷积
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,  # 3×3 卷积，stride 控制是否下采样
                     padding=1, bias=False)  # padding=1 使 stride=1 时分辨率不变；无 bias（紧随的 BN 自带偏置）


# ----------------------- conv1x1 参数说明 -----------------------
# in_planes：输入通道数；out_planes：输出通道数；stride：步长（用于捷径投影下采样）
def conv1x1(in_planes, out_planes, stride=1):  # 封装 1×1 卷积：仅做通道升降维（捷径投影时兼下采样）
    """1x1 convolution"""  # 原英文 docstring 保留：1×1 卷积
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)  # 1×1 卷积，stride>1 时同时完成空间下采样


class BasicBlock(nn.Module):  # 基础残差块：两个 3×3 卷积 + 捷径相加（resnet18/34 使用）
    expansion = 1  # 输出通道相对 planes 的放大倍数：BasicBlock 不扩通道，恒为 1

    # ----------------------- BasicBlock.__init__ 参数说明 -----------------------
    # inplanes：块输入通道数；planes：块内卷积通道数（= 输出通道数，因 expansion=1）
    # stride：下采样步长（放在 conv1 的 3×3 上，仅 stage 首块 >1）
    # downsample：捷径投影子模块（输入输出通道/分辨率不一致时才非 None，否则恒等捷径）
    # norm_layer：归一化层构造函数（默认 BatchNorm2d，可换 GroupNorm 等适配小 batch）
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):  # 构造两卷积残差块
        super(BasicBlock, self).__init__()  # 调用父类 nn.Module 构造函数，初始化模块树
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)  # 3×3 卷积①：通道变换，stride>1 时同时把分辨率减半
        self.bn1 = norm_layer(planes)  # 卷积①后的批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活（inplace 省显存）
        self.conv2 = conv3x3(planes, planes)  # 3×3 卷积②：stride=1、通道不变，继续提取空间特征
        self.bn2 = norm_layer(planes)  # 卷积②后的批归一化（相加之前不再激活）
        self.downsample = downsample  # 保存捷径投影（None 表示直接恒等捷径）
        self.stride = stride  # 记录步长（仅存档，forward 中不使用）

    def forward(self, x):  # 前向：卷积→BN→ReLU ×2 + 捷径相加
        identity = x  # 保存输入作为捷径分支（identity 分支）

        out = self.conv1(x)  # 3×3 卷积①（stride>1 时 H、W 减半）
        out = self.bn1(out)  # BN①
        out = self.relu(out)  # ReLU①

        out = self.conv2(out)  # 3×3 卷积②（分辨率、通道不变）
        out = self.bn2(out)  # BN②

        if self.downsample is not None:  # 通道数或分辨率发生变化时需要投影捷径
            identity = self.downsample(x)  # 1×1 卷积 + BN 把捷径对齐到输出形状

        out += identity  # 残差相加：F(x) + x
        out = self.relu(out)  # 相加之后再激活

        return out  # 返回块输出 (N, planes*1, H/stride, W/stride)


class Bottleneck(nn.Module):  # 瓶颈残差块：1×1→3×3→1×1 三卷积结构（resnet50/101/152 使用）
    expansion = 4  # 输出通道 = planes × 4（末个 1×1 卷积升维 4 倍）

    # ----------------------- Bottleneck.__init__ 参数说明 -----------------------
    # inplanes：块输入通道数；planes：中间瓶颈通道数（输出为 planes*expansion）
    # stride：下采样步长（放在 conv2 的 3×3 上，v1.5 结构；仅 stage 首块 >1）
    # downsample：捷径投影子模块（通道/分辨率不一致时非 None）
    # norm_layer：归一化层构造函数
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):  # 构造三卷积瓶颈块
        super(Bottleneck, self).__init__()  # 调用父类构造函数
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)  # 1×1 卷积①：通道降维（如 256→64），压缩后续 3×3 计算量
        self.bn1 = norm_layer(planes)  # BN①
        self.conv2 = conv3x3(planes, planes, stride)  # 3×3 卷积②：唯一做空间卷积的层，stride>1 时下采样
        self.bn2 = norm_layer(planes)  # BN②
        self.conv3 = conv1x1(planes, planes * self.expansion)  # 1×1 卷积③：通道升维 4 倍（64→256），恢复特征宽度
        self.bn3 = norm_layer(planes * self.expansion)  # BN③：残差分支最后一个 BN（zero_init_residual 会把它的 γ 置 0）
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活
        self.downsample = downsample  # 保存捷径投影
        self.stride = stride  # 记录步长（存档用）

    def forward(self, x):  # 前向：三卷积瓶颈结构 + 捷径相加
        identity = x  # 保存输入作为捷径分支

        out = self.conv1(x)  # 1×1 降维：(N, inplanes, H, W) → (N, planes, H, W)
        out = self.bn1(out)  # BN①
        out = self.relu(out)  # ReLU①

        out = self.conv2(out)  # 3×3 空间卷积（stride>1 时 H、W 减半）
        out = self.bn2(out)  # BN②
        out = self.relu(out)  # ReLU②

        out = self.conv3(out)  # 1×1 升维：(N, planes, H, W) → (N, planes*4, H, W)
        out = self.bn3(out)  # BN③

        if self.downsample is not None:  # 尺寸/通道不匹配时启用投影捷径
            identity = self.downsample(x)  # 1×1 卷积 + BN 对齐捷径形状

        out += identity  # 残差相加：F(x) + x
        out = self.relu(out)  # 相加之后再激活

        return out  # 返回 (N, planes*4, H/stride, W/stride)


class ResNet(nn.Module):  # ResNet 主体：stem（conv1+bn1+relu+maxpool）+ 4 个残差 stage + 分类头

    # ----------------------- ResNet.__init__ 参数说明 -----------------------
    # block：残差块类型（BasicBlock 或 Bottleneck）
    # layers：四个 stage 各自的残差块数量（如 [3,4,6,3] 即 resnet50）
    # num_classes：分类头输出类别数（ImageNet 默认 1000；作分割编码器时不使用该头）
    # zero_init_residual：是否将残差分支最后一个 BN 的 γ 零初始化（利于训练初期稳定）
    # norm_layer：归一化层构造函数，默认 nn.BatchNorm2d
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=nn.BatchNorm2d):  # 按配置构建完整 ResNet
        super(ResNet, self).__init__()  # 调用父类构造函数
        self.inplanes = 64  # 通道数游标：记录"当前输出通道"，供 _make_layer 逐 stage 更新（stem 输出 64）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  # stem 的 7×7 卷积：RGB 3 通道 → 64 通道，stride=2 下采样
                               bias=False)  # 无 bias（后接 BN）
        self.bn1 = norm_layer(64)  # stem 的批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU 激活
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 3×3 最大池化 stride=2：stem 后再下采样一半（累计 1/4）
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)  # stage1：基准 64 通道、stride=1 → 输出 64*E，分辨率仍 1/4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)  # stage2：128 通道、stride=2 → 分辨率 1/8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)  # stage3：256 通道、stride=2 → 分辨率 1/16
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)  # stage4：512 通道、stride=2 → 分辨率 1/32（输出 512*E）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应全局平均池化：任意输入尺寸 → 1×1（分类头专用）
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 分类全连接层：512*E → num_classes（分割作编码器时不使用）

        for m in self.modules():  # 遍历所有子模块统一做参数初始化
            if isinstance(m, nn.Conv2d):  # 卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming 正态初始化（按 fan_out 保持 ReLU 前后方差）
            elif isinstance(m, nn.BatchNorm2d):  # 批归一化层
                nn.init.constant_(m.weight, 1)  # 缩放因子 γ 初始化为 1
                nn.init.constant_(m.bias, 0)  # 偏移 β 初始化为 0

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:  # 可选开关：残差分支末 BN 零初始化
            for m in self.modules():  # 再次遍历所有模块
                if isinstance(m, Bottleneck):  # 瓶颈块 → 置零 bn3
                    nn.init.constant_(m.bn3.weight, 0)  # γ=0 → 初始时残差分支输出 0，整块退化为恒等映射
                elif isinstance(m, BasicBlock):  # 基础块 → 置零 bn2
                    nn.init.constant_(m.bn2.weight, 0)  # 同上，块初始等价恒等映射

    # ----------------------- _make_layer 参数说明 -----------------------
    # block：残差块类型；planes：本 stage 基准通道数；blocks：本 stage 块数量
    # stride：本 stage 首块的下采样步长（其余块恒为 1）；norm_layer：归一化层构造函数
    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=nn.BatchNorm2d):  # 堆叠 blocks 个残差块构成一个 stage
        downsample = None  # 捷径投影，默认 None（恒等捷径）
        if stride != 1 or self.inplanes != planes * block.expansion:  # 需要下采样，或输入通道 ≠ 输出通道时
            downsample = nn.Sequential(  # 捷径投影 = 1×1 卷积 + BN 的顺序容器
                conv1x1(self.inplanes, planes * block.expansion, stride),  # 1×1 卷积：对齐通道数并同步下采样
                norm_layer(planes * block.expansion),  # 投影后的 BN
            )  # 捷径投影 Sequential 定义结束

        layers = []  # 本 stage 的残差块列表
        layers.append(block(self.inplanes, planes, stride, downsample))  # 首块：携带 stride 与投影捷径（stage 内唯一可能下采样的块）
        self.inplanes = planes * block.expansion  # 更新通道游标为本 stage 的输出通道
        for _ in range(1, blocks):  # 其余 blocks-1 个块（下划线表示不使用循环变量）
            layers.append(block(self.inplanes, planes))  # 常规块：恒等捷径、stride=1

        return nn.Sequential(*layers)  # 将块列表串成顺序容器返回

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像张量 (N, 3, H, W)；返回：分类 logits (N, num_classes)
    # 形状以 Bottleneck（E=expansion=4）为例；BasicBlock 时各 stage 通道为 64/128/256/512
    def forward(self, x):  # 前向：stem → 4 个 stage → 全局池化 → 分类
        x = self.conv1(x)  # 7×7 卷积 stride=2：(N, 3, H, W) → (N, 64, H/2, W/2)
        x = self.bn1(x)  # BN（形状不变）
        x = self.relu(x)  # ReLU（形状不变）
        x = self.maxpool(x)  # 最大池化 stride=2 → (N, 64, H/4, W/4)

        x = self.layer1(x)  # stage1（stride=1）→ (N, 256, H/4, W/4)
        x = self.layer2(x)  # stage2（stride=2）→ (N, 512, H/8, W/8)
        x = self.layer3(x)  # stage3（stride=2）→ (N, 1024, H/16, W/16)
        x = self.layer4(x)  # stage4（stride=2）→ (N, 2048, H/32, W/32)

        x = self.avgpool(x)  # 全局平均池化 → (N, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # 展平 → (N, 2048)
        x = self.fc(x)  # 全连接 → (N, num_classes)

        return x  # 返回分类 logits


# 工厂函数：构建 ResNet-18（BasicBlock × [2,2,2,2]）
# pretrained=True 时经 model_zoo.load_url 下载 ImageNet 权重并加载（strict 默认 True，结构不一致会直接报错）
def resnet18(pretrained=False, **kwargs):  # **kwargs 透传给 ResNet（如 norm_layer、num_classes）
    # resnet18 工厂函数 docstring：构建 ResNet-18 模型说明（原英文保留）
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)  # 用基础块、每 stage 2 块构建 18 层网络
    if pretrained:  # 需要加载预训练权重
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))  # 下载并加载 ResNet-18 的 ImageNet 权重【注意】load_url 为废弃 API
    return model  # 返回构建好的模型


# 工厂函数：构建 ResNet-34（BasicBlock × [3,4,6,3]）
# pretrained=True 时经 model_zoo.load_url 下载 ImageNet 权重并加载（strict 默认 True）
def resnet34(pretrained=False, **kwargs):  # **kwargs 透传给 ResNet
    # resnet34 工厂函数 docstring：构建 ResNet-34 模型说明（原英文保留）
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)  # 用基础块、块数 [3,4,6,3] 构建 34 层网络
    if pretrained:  # 需要加载预训练权重
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))  # 下载并加载 ResNet-34 的 ImageNet 权重【注意】load_url 为废弃 API
    return model  # 返回构建好的模型


# 工厂函数：构建 ResNet-50（Bottleneck × [3,4,6,3]，分割工程中最常用的经典骨干）
# pretrained=True 时经 model_zoo.load_url 下载 ImageNet 权重并加载（strict 默认 True）
def resnet50(pretrained=False, **kwargs):  # **kwargs 透传给 ResNet
    # resnet50 工厂函数 docstring：构建 ResNet-50 模型说明（原英文保留）
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)  # 用瓶颈块、块数 [3,4,6,3] 构建 50 层网络
    if pretrained:  # 需要加载预训练权重
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))  # 下载并加载 ResNet-50 的 ImageNet 权重【注意】load_url 为废弃 API
    return model  # 返回构建好的模型


# 工厂函数：构建 ResNet-101（Bottleneck × [3,4,23,3]，stage3 加深到 23 块）
# pretrained=True 时经 model_zoo.load_url 下载 ImageNet 权重并加载（strict 默认 True）
def resnet101(pretrained=False, **kwargs):  # **kwargs 透传给 ResNet
    # resnet101 工厂函数 docstring：构建 ResNet-101 模型说明（原英文保留）
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)  # 用瓶颈块、块数 [3,4,23,3] 构建 101 层网络
    if pretrained:  # 需要加载预训练权重
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))  # 下载并加载 ResNet-101 的 ImageNet 权重【注意】load_url 为废弃 API
    return model  # 返回构建好的模型


# 工厂函数：构建 ResNet-152（Bottleneck × [3,8,36,3]，最深的经典配置）
# pretrained=True 时经 model_zoo.load_url 下载 ImageNet 权重并加载（strict 默认 True）
def resnet152(pretrained=False, **kwargs):  # **kwargs 透传给 ResNet
    # resnet152 工厂函数 docstring：构建 ResNet-152 模型说明（原英文保留）
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)  # 用瓶颈块、块数 [3,8,36,3] 构建 152 层网络
    if pretrained:  # 需要加载预训练权重
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))  # 下载并加载 ResNet-152 的 ImageNet 权重【注意】load_url 为废弃 API
    return model  # 返回构建好的模型


if __name__ == '__main__':  # 直接运行本文件时的简易自测入口
    import torch  # 导入 torch 用于构造随机输入张量
    img = torch.randn(4, 3, 224, 224)  # 构造 batch=4 的随机图像张量 (4, 3, 224, 224)
    model = resnet50(True)  # 构建 ResNet-50 并加载 ImageNet 预训练权重【注意】pretrained=True 会联网下载权重，离线环境将失败
    output = model(img)  # 前向推理 → 输出分类 logits (4, 1000)
