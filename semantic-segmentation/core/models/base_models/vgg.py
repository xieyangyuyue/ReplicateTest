# ============================================================================
# 【文件说明】vgg.py —— VGG 系列经典卷积骨干网络（VGG11/13/16/19 及 BN 变体）
#     提供 VGG 网络类、make_layers 特征构建函数、cfg 网络配置表，以及
#     vgg11 ~ vgg19_bn 共 8 个工厂函数，支持加载 ImageNet 预训练权重。
# 【核心思想】用堆叠的小卷积核（3×3、stride=1、padding=1）替代大感受野大核：
#     两层 3×3 的感受野等效一层 5×5，三层等效 7×7，参数更少且非线性更强。
#     网络按"卷积组 + 2×2 最大池化"分为五个阶段，通道数 64→128→256→512→512
#     阶梯式翻倍。cfg 配置表中数字代表卷积层输出通道、'M' 代表最大池化层，
#     make_layers 据此逐层构建"Conv(+BN)+ReLU"序列。VGG16（配置 'D'）是
#     FCN 系列分割模型最常用的骨干，本工程取其 features 卷积组作编码器，
#     pool3/pool4 等阶段前的特征作为分割头的多尺度输入。
# 【工程位置】core/models/base_models/ 下的骨干网络之一，由上层分割模型
#     （如 fcn.py）通过 vgg16(pretrained=True) 等工厂函数实例化。
# ============================================================================
import torch  # PyTorch 主库（__main__ 自测构造随机输入用）
import torch.nn as nn  # 神经网络模块：Conv2d/MaxPool2d/ReLU/Linear/Sequential 等
import torch.utils.model_zoo as model_zoo  # 预训练权重下载工具【注意】model_zoo.load_url 属废弃 API，新版 PyTorch 建议改用 torch.hub.load_state_dict_from_url

__all__ = [  # 模块公开符号清单
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',  # VGG 类与各工厂函数（前半部分）
    'vgg19_bn', 'vgg19',  # __all__ 清单续行：vgg19_bn、vgg19
]  # __all__ 清单结束

model_urls = {  # 各 VGG 变体的 ImageNet 预训练权重下载地址映射
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',  # VGG11 权重
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',  # VGG13 权重
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',  # VGG16 权重（FCN 分割常用）
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',  # VGG19 权重
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',  # VGG11 + BN 权重
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',  # VGG13 + BN 权重
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',  # VGG16 + BN 权重
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',  # VGG19 + BN 权重
}  # 权重地址表结束


# VGG 主网络：features 卷积特征 + avgpool（7×7）+ classifier 三层全连接分类头
class VGG(nn.Module):
    # 参数：features 由 make_layers 构建的卷积特征提取器；num_classes 分类数；init_weights 是否执行随机初始化
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()  # 初始化 nn.Module 基类
        self.features = features  # 保存外部传入的卷积特征部分
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # 自适应平均池化到 7×7（兼容任意输入尺寸，配合分类头的 512*7*7）
        self.classifier = nn.Sequential(  # 分类头：三层全连接
            nn.Linear(512 * 7 * 7, 4096),  # (N,25088) → (N,4096)
            nn.ReLU(True),  # 原位激活
            nn.Dropout(),  # 随机失活（默认 p=0.5）
            nn.Linear(4096, 4096),  # (N,4096) → (N,4096)
            nn.ReLU(True),  # 原位激活
            nn.Dropout(),  # 随机失活（默认 p=0.5）
            nn.Linear(4096, num_classes)  # (N,4096) → (N,num_classes)
        )  # 分类头 Sequential 定义结束
        if init_weights:  # 需要随机初始化时（加载预训练权重时会传 False 跳过）
            self._initialize_weights()  # 执行权重初始化

    # 前向传播：x (N,3,H,W) → 分类 logits (N,num_classes)
    def forward(self, x):
        x = self.features(x)  # 卷积特征：(N,3,H,W) → (N,512,H/32,W/32)
        x = self.avgpool(x)  # 自适应池化：(N,512,H/32,W/32) → (N,512,7,7)
        x = x.view(x.size(0), -1)  # 展平：(N,512,7,7) → (N,25088)
        x = self.classifier(x)  # 分类头：(N,25088) → (N,num_classes)
        return x  # 返回分类 logits

    # 权重初始化：按模块类型分别初始化
    def _initialize_weights(self):
        for m in self.modules():  # 遍历全部子模块
            if isinstance(m, nn.Conv2d):  # 卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming 正态初始化（fan_out、relu）
                if m.bias is not None:  # 若带偏置
                    nn.init.constant_(m.bias, 0)  # 偏置置 0
            elif isinstance(m, nn.BatchNorm2d):  # BN 层
                nn.init.constant_(m.weight, 1)  # γ 置 1
                nn.init.constant_(m.bias, 0)  # β 置 0
            elif isinstance(m, nn.Linear):  # 全连接层
                nn.init.normal_(m.weight, 0, 0.01)  # 权重按 N(0, 0.01²) 初始化
                nn.init.constant_(m.bias, 0)  # 偏置置 0


# 按 cfg 配置构建卷积特征序列：数字 → 3×3 卷积(+BN)+ReLU；'M' → 2×2 最大池化
def make_layers(cfg, batch_norm=False):
    layers = []  # 层收集列表
    in_channels = 3  # 输入为 RGB 三通道
    for v in cfg:  # 逐项遍历配置
        if v == 'M':  # 'M' 表示最大池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 2×2 最大池化（stride=2），空间减半
        else:  # 数字表示卷积层输出通道
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)  # 3×3 卷积（stride=1、padding=1，空间不变）
            if batch_norm:  # BN 变体
                layers += (conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True))  # 卷积+BN+ReLU【注意】可疑写法：列表 += 元组（等价于 extend，结果正确，但与下面 += 列表的风格不一致）
            else:  # 普通 VGG
                layers += [conv2d, nn.ReLU(inplace=True)]  # 卷积+ReLU
            in_channels = v  # 更新下一层的输入通道
    return nn.Sequential(*layers)  # 串联为特征提取器


# VGG 网络配置表：A=VGG11、B=VGG13、D=VGG16、E=VGG19；数字为卷积输出通道，'M' 为最大池化
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG11：8 个卷积层
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG13：10 个卷积层
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],  # VGG16：13 个卷积层（FCN 系列分割骨干常用）
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],  # VGG19：16 个卷积层
}  # cfg 网络配置表结束


# VGG11 工厂函数（配置 A）：pretrained 为 True 时加载 ImageNet 预训练权重
def vgg11(pretrained=False, **kwargs):
    # vgg11 工厂函数 docstring：VGG 11 层模型（配置 A）说明（原英文保留）
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:  # 加载预训练权重时
        kwargs['init_weights'] = False  # 关闭随机初始化（权重随后被覆盖）
    model = VGG(make_layers(cfg['A']), **kwargs)  # 按配置 A 构建特征部分并组装 VGG
    if pretrained:  # 需要预训练权重时
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))  # 下载并加载 VGG11 的 ImageNet 权重
    return model  # 返回模型


# VGG11 + BN 工厂函数（配置 A，卷积后接 BatchNorm）
def vgg11_bn(pretrained=False, **kwargs):
    # vgg11_bn 工厂函数 docstring：VGG 11 层 + BN 模型说明（原英文保留）
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:  # 加载预训练权重时
        kwargs['init_weights'] = False  # 关闭随机初始化
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)  # 按配置 A + BN 构建特征部分
    if pretrained:  # 需要预训练权重时
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))  # 下载并加载 VGG11+BN 的权重
    return model  # 返回模型


# VGG13 工厂函数（配置 B）
def vgg13(pretrained=False, **kwargs):
    # vgg13 工厂函数 docstring：VGG 13 层模型（配置 B）说明（原英文保留）
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:  # 加载预训练权重时
        kwargs['init_weights'] = False  # 关闭随机初始化
    model = VGG(make_layers(cfg['B']), **kwargs)  # 按配置 B 构建特征部分
    if pretrained:  # 需要预训练权重时
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))  # 下载并加载 VGG13 的权重
    return model  # 返回模型


# VGG13 + BN 工厂函数（配置 B，卷积后接 BatchNorm）
def vgg13_bn(pretrained=False, **kwargs):
    # vgg13_bn 工厂函数 docstring：VGG 13 层 + BN 模型说明（原英文保留）
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:  # 加载预训练权重时
        kwargs['init_weights'] = False  # 关闭随机初始化
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)  # 按配置 B + BN 构建特征部分
    if pretrained:  # 需要预训练权重时
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))  # 下载并加载 VGG13+BN 的权重
    return model  # 返回模型


# VGG16 工厂函数（配置 D）——本工程 FCN 等分割模型最常用的骨干入口
def vgg16(pretrained=False, **kwargs):
    # vgg16 工厂函数 docstring：VGG 16 层模型（配置 D）说明（原英文保留）
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:  # 加载预训练权重时
        kwargs['init_weights'] = False  # 关闭随机初始化
    model = VGG(make_layers(cfg['D']), **kwargs)  # 按配置 D 构建特征部分
    if pretrained:  # 需要预训练权重时
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))  # 下载并加载 VGG16 的 ImageNet 权重
    return model  # 返回模型


# VGG16 + BN 工厂函数（配置 D，卷积后接 BatchNorm）
def vgg16_bn(pretrained=False, **kwargs):
    # vgg16_bn 工厂函数 docstring：VGG 16 层 + BN 模型说明（原英文保留）
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:  # 加载预训练权重时
        kwargs['init_weights'] = False  # 关闭随机初始化
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)  # 按配置 D + BN 构建特征部分
    if pretrained:  # 需要预训练权重时
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))  # 下载并加载 VGG16+BN 的权重
    return model  # 返回模型


# VGG19 工厂函数（配置 E）
def vgg19(pretrained=False, **kwargs):
    # vgg19 工厂函数 docstring：VGG 19 层模型（配置 E）说明（原英文保留）
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:  # 加载预训练权重时
        kwargs['init_weights'] = False  # 关闭随机初始化
    model = VGG(make_layers(cfg['E']), **kwargs)  # 按配置 E 构建特征部分
    if pretrained:  # 需要预训练权重时
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))  # 下载并加载 VGG19 的权重
    return model  # 返回模型


# VGG19 + BN 工厂函数（配置 E，卷积后接 BatchNorm）
def vgg19_bn(pretrained=False, **kwargs):
    # vgg19_bn 工厂函数 docstring：VGG 19 层 + BN 模型说明（原英文保留）
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:  # 加载预训练权重时
        kwargs['init_weights'] = False  # 关闭随机初始化
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)  # 按配置 E + BN 构建特征部分
    if pretrained:  # 需要预训练权重时
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))  # 下载并加载 VGG19+BN 的权重
    return model  # 返回模型


if __name__ == '__main__':  # 直接运行本文件时做前向自测
    img = torch.randn((4, 3, 480, 480))  # 随机输入 (4,3,480,480)
    model = vgg16(pretrained=False)  # 构建 VGG16（不加载预训练权重）
    out = model(img)  # 前向传播 → (4,1000)
