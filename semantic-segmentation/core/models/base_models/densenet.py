# ============================================================================
# 【文件说明】densenet.py —— DenseNet 系列（DenseNet-121/161/169/201）骨干网络，
#             以及面向密集预测的空洞版 DilatedDenseNet（删除后段 Transition 的池化
#             并提高空洞率，使输出步长降为 8 或 16）。
# 【核心思想】DenseNet 的“密集连接”：每个 _DenseLayer 的输出与输入在通道维 concat，
#             因此 _DenseBlock 内第 i 层的输入通道数 = Block 入口通道数 + i*growth_rate
#             （growth rate 控制每层新增通道数 k）；早期特征被反复复用，梯度流动更
#             顺畅、参数更省。相邻 DenseBlock 之间用 _Transition（1x1 卷积把通道压缩
#             到一半 + 2x2 平均池化下采样）衔接，即 DenseNet-BC 的压缩设计。
#             工程位置：core/models/base_models/，作为编码器被分割模型（如 ICNet）调用。
# ============================================================================

import re  # 正则库：用于把旧版预训练权重的 state_dict 键名（norm.1 等）改写成新版键名
import torch  # PyTorch 主库：张量运算
import torch.nn as nn  # 神经网络模块库：Conv2d/BatchNorm2d/Sequential 等
import torch.nn.functional as F  # 函数式接口：dropout、relu、adaptive_avg_pool2d
import torch.utils.model_zoo as model_zoo  # 按 URL 下载并加载预训练权重；【注意】该接口已不推荐，新版可用 torch.hub.load_state_dict_from_url

from collections import OrderedDict  # 有序字典：保证 features 内子模块按插入顺序执行（键名与官方权重对齐）

__all__ = ['DenseNet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',  # 模块导出符号（标准版工厂函数）
           'dilated_densenet121', 'dilated_densenet161', 'dilated_densenet169', 'dilated_densenet201']  # 模块导出符号（空洞版工厂函数）

model_urls = {  # 各 DenseNet 变体的官方 ImageNet 预训练权重下载地址
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',  # DenseNet-121 权重
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',  # DenseNet-169 权重
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',  # DenseNet-201 权重
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',  # DenseNet-161 权重
}  # 权重地址表结束


class _DenseLayer(nn.Sequential):  # DenseNet 单层（瓶颈结构）：BN-ReLU-Conv1x1 → BN-ReLU-Conv3x3，输出与输入 concat
    # 参数说明：num_input_features——本层输入通道数（= Block 入口通道 + 已堆叠层数*growth_rate）
    #           growth_rate——每层新增通道数 k；bn_size——瓶颈倍率（1x1 卷积输出通道 = bn_size*k）
    #           drop_rate——dropout 概率；dilation——3x3 卷积空洞率；norm_layer——归一化层构造函数
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation=1, norm_layer=nn.BatchNorm2d):  # 初始化
        super(_DenseLayer, self).__init__()  # 父类 nn.Sequential 初始化
        self.add_module('norm1', norm_layer(num_input_features)),  # BN1（作用在输入上）；【注意】行尾多余逗号使该语句成为无副作用的“元组表达式”（继承自 torchvision 旧代码的写法，无实际影响）
        self.add_module('relu1', nn.ReLU(True)),  # ReLU1（inplace）
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, 1, 1, bias=False)),  # 1x1 瓶颈卷积：通道 in→bn_size*k（默认 4k）
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),  # BN2
        self.add_module('relu2', nn.ReLU(True)),  # ReLU2
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, 3, 1, dilation, dilation, bias=False)),  # 3x3 卷积：通道 bn_size*k→k，stride=1、padding=dilation 保持尺寸
        self.drop_rate = drop_rate  # 记录 dropout 概率（forward 中使用）

    def forward(self, x):  # 前向：x 是此前所有层输出的 concat
        new_features = super(_DenseLayer, self).forward(x)  # 顺序执行 norm1→relu1→conv1→norm2→relu2→conv2，得到本层新增特征 (N,k,H,W)
        if self.drop_rate > 0:  # 配置了 dropout
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)  # 按 train/eval 模式做 dropout
        return torch.cat([x, new_features], 1)  # 密集连接核心：通道维 concat，输出通道 = 输入通道 + k


class _DenseBlock(nn.Sequential):  # DenseNet 密集块：堆叠 num_layers 个 _DenseLayer，层层 concat
    # 参数说明：num_layers——块内层数；num_input_features——进入本块的通道数；bn_size——瓶颈倍率
    #           growth_rate——每层新增通道 k；drop_rate——dropout 概率；dilation——空洞率；norm_layer——归一化层构造函数
    def __init__(self, num_layers, num_input_features, bn_size,  # 构造函数（参数含义见上方注释）
                 growth_rate, drop_rate, dilation=1, norm_layer=nn.BatchNorm2d):  # 续行：其余超参数
        super(_DenseBlock, self).__init__()  # 父类初始化
        for i in range(num_layers):  # 逐层构建（i 从 0 开始）
            layer = _DenseLayer(num_input_features + i * growth_rate,  # 第 i 层输入通道 = Block 入口通道 + i*k（前面各层输出的 concat 累积）
                                growth_rate, bn_size, drop_rate, dilation, norm_layer)  # 其余超参原样透传
            self.add_module('denselayer%d' % (i + 1), layer)  # 子模块命名为 denselayer1..N（与官方权重键名一致）


class _Transition(nn.Sequential):  # 过渡层：BN-ReLU-Conv1x1（通道压缩）+ AvgPool2d（空间减半）
    # 参数说明：num_input_features——输入通道；num_output_features——输出通道（调用处传 num_features//2，即压缩到一半）
    #           norm_layer——归一化层构造函数
    def __init__(self, num_input_features, num_output_features, norm_layer=nn.BatchNorm2d):  # 初始化
        super(_Transition, self).__init__()  # 父类初始化
        self.add_module('norm', norm_layer(num_input_features))  # BN
        self.add_module('relu', nn.ReLU(True))  # ReLU
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, 1, 1, bias=False))  # 1x1 卷积：通道 in→out（DenseNet-BC 的 compression=0.5 压缩）
        self.add_module('pool', nn.AvgPool2d(2, 2))  # 2x2 平均池化 stride=2：空间尺寸减半


# Net
# DenseNet 主体
class DenseNet(nn.Module):  # DenseNet 分类网络主体（分割任务中被当作编码器使用）

    # 参数说明：growth_rate——每层新增通道 k；block_config——各 DenseBlock 的层数元组
    #           num_init_features——stem 输出通道数；bn_size——瓶颈倍率（默认 4）
    #           drop_rate——dropout 概率；num_classes——分类数；norm_layer——归一化层构造函数
    #           **kwargs——接收并忽略额外关键字参数
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), num_init_features=64,  # 构造函数（参数含义见上方注释）
                 bn_size=4, drop_rate=0, num_classes=1000, norm_layer=nn.BatchNorm2d, **kwargs):  # 续行：其余超参数
        super(DenseNet, self).__init__()  # 父类初始化

        # First convolution
        # stem：7x7 卷积 + BN + ReLU + 3x3 最大池化，空间缩至 1/4
        self.features = nn.Sequential(OrderedDict([  # 用 OrderedDict 组织特征提取链，键名与官方权重对齐
            ('conv0', nn.Conv2d(3, num_init_features, 7, 2, 3, bias=False)),  # 7x7 卷积 stride=2、padding=3：通道 3→num_init_features，空间减半
            ('norm0', norm_layer(num_init_features)),  # BN
            ('relu0', nn.ReLU(True)),  # ReLU
            ('pool0', nn.MaxPool2d(3, 2, 1)),  # 3x3 最大池化 stride=2：空间再减半（至此 /4）
        ]))  # stem 特征链（OrderedDict→Sequential）定义结束

        # Each denseblock
        # 依次搭建 DenseBlock 与 Transition
        num_features = num_init_features  # 当前通道计数（进入第一个 Block 前的通道数）
        for i, num_layers in enumerate(block_config):  # 遍历每个 Block 的层数配置
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate, norm_layer=norm_layer)  # 构建第 i+1 个 DenseBlock（【注意】未传 dilation，用默认 1）
            self.features.add_module('denseblock%d' % (i + 1), block)  # 命名为 denseblock1..4
            num_features = num_features + num_layers * growth_rate  # Block 结束后通道 = 入口通道 + 层数*k（concat 累加）
            if i != len(block_config) - 1:  # 非最后一个 Block 时，后面接一个过渡层
                trans = _Transition(num_features, num_features // 2, norm_layer=norm_layer)  # 过渡层：通道压缩到一半（compression=0.5）
                self.features.add_module('transition%d' % (i + 1), trans)  # 命名为 transition1..3
                num_features = num_features // 2  # 更新通道计数为压缩后的值
        self.num_features = num_features  # 记录最终特征通道数（如 DenseNet-121 为 1024）

        # Final batch norm
        # 收尾 BN
        self.features.add_module('norm5', norm_layer(num_features))  # 最后一个 BN，键名 norm5（与官方权重对齐）

        # Linear layer
        # 分类头
        self.classifier = nn.Linear(num_features, num_classes)  # 全连接分类层 num_features→num_classes

        # Official init from torch repo.
        # 官方风格的参数初始化
        for m in self.modules():  # 遍历所有子模块
            if isinstance(m, nn.Conv2d):  # 卷积层
                nn.init.kaiming_normal_(m.weight)  # Kaiming 正态初始化权重
            elif isinstance(m, nn.BatchNorm2d):  # BN 层（【注意】若 norm_layer 换成非 BN 类型，该分支不会命中）
                nn.init.constant_(m.weight, 1)  # weight 初始化为 1
                nn.init.constant_(m.bias, 0)  # bias 初始化为 0
            elif isinstance(m, nn.Linear):  # 全连接层
                nn.init.constant_(m.bias, 0)  # bias 初始化为 0

    def forward(self, x):  # 前向传播：输入 (N,3,H,W)
        features = self.features(x)  # 完整特征链：(N,3,H,W)→(N,num_features,H/32,W/32)（空洞版为 /8 或 /16）
        out = F.relu(features, True)  # 收尾 BN 后的 ReLU（【注意】inplace 以位置参数传入）：形状不变
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)  # 全局平均池化并展平：(N,num_features,H',W')→(N,num_features)
        out = self.classifier(out)  # 分类：(N,num_features)→(N,num_classes)
        return out  # 返回 logits


class DilatedDenseNet(DenseNet):  # 空洞版 DenseNet：删除后段池化、提高空洞率，输出更高分辨率特征图（供密集预测）
    # 参数说明：前 7 个参数同 DenseNet；dilate_scale——目标输出步长（仅支持 8 或 16）；norm_layer——归一化层构造函数
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16), num_init_features=64,  # 构造函数（参数含义见上方注释）
                 bn_size=4, drop_rate=0, num_classes=1000, dilate_scale=8, norm_layer=nn.BatchNorm2d, **kwargs):  # 续行：其余超参数
        super(DilatedDenseNet, self).__init__(growth_rate, block_config, num_init_features,  # 先按标准 DenseNet 搭建，再做空洞改造（【注意】父类调用全部使用位置参数）
                                              bn_size, drop_rate, num_classes, norm_layer)  # 续行：与上一行同属一条语句
        assert (dilate_scale == 8 or dilate_scale == 16), "dilate_scale can only set as 8 or 16"  # 只允许输出步长 8/16
        from functools import partial  # 局部导入 partial：把 _conv_dilate 包装成单参数函数供 apply 使用
        if dilate_scale == 8:  # 输出步长 8 的改造方案
            self.features.denseblock3.apply(partial(self._conv_dilate, dilate=2))  # denseblock3 内所有 3x3 卷积空洞率改为 2
            self.features.denseblock4.apply(partial(self._conv_dilate, dilate=4))  # denseblock4 内所有 3x3 卷积空洞率改为 4
            del self.features.transition2.pool  # 删除 transition2 的池化子模块（Sequential 少了 pool 即不再下采样，分辨率保持 /8）
            del self.features.transition3.pool  # 删除 transition3 的池化子模块（配合 denseblock4 的空洞 4）
        elif dilate_scale == 16:  # 输出步长 16 的改造方案
            self.features.denseblock4.apply(partial(self._conv_dilate, dilate=2))  # denseblock4 内 3x3 卷积空洞率改为 2
            del self.features.transition3.pool  # 仅删除 transition3 的池化（分辨率保持 /16）

    def _conv_dilate(self, m, dilate):  # 模块改造函数（配合 nn.Module.apply 递归遍历子模块调用）
        classname = m.__class__.__name__  # 取模块类名
        if classname.find('Conv') != -1:  # 只处理名字含 Conv 的卷积类模块
            if m.kernel_size == (3, 3):  # 只改 3x3 卷积（1x1 瓶颈卷积不受影响）
                m.padding = (dilate, dilate)  # padding 同步改为 dilate，保证特征图尺寸不变
                m.dilation = (dilate, dilate)  # 空洞率改为 dilate


# Specification
# 各层数规格：层数 → (num_init_features, growth_rate, [各 DenseBlock 层数])
densenet_spec = {121: (64, 32, [6, 12, 24, 16]),  # DenseNet-121：stem 64 通道，k=32，4 个 Block 分别 6/12/24/16 层
                 161: (96, 48, [6, 12, 36, 24]),  # DenseNet-161：stem 96 通道，k=48，4 个 Block 分别 6/12/36/24 层
                 169: (64, 32, [6, 12, 32, 32]),  # DenseNet-169：stem 64 通道，k=32，4 个 Block 分别 6/12/32/32 层
                 201: (64, 32, [6, 12, 48, 32])}  # DenseNet-201：stem 64 通道，k=32，4 个 Block 分别 6/12/48/32 层


# Constructor
# 工厂函数：按层数构建 DenseNet，并可加载官方 ImageNet 预训练权重
def get_densenet(num_layers, pretrained=False, **kwargs):  # num_layers——121/161/169/201；pretrained——是否加载预训练权重；**kwargs——透传给 DenseNet
    # 下方为原始英文 docstring，原样保留（【注意】其中提到 root 参数，但函数签名并未定义该参数）
    r"""Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.

    Parameters
    ----------
    num_layers : int
        Number of layers for the variant of densenet. Options are 121, 161, 169, 201.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default $TORCH_HOME/models
        Location for keeping the model parameters.
    """  # 原始 docstring 结束
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]  # 按层数查规格表取出三要素
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)  # 构建对应规格的 DenseNet
    if pretrained:  # 需要加载预训练权重
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        # （上方四行为原代码英文注释，原样保留；【注意】“pervious”为原文拼写错误，应为 previous）
        pattern = re.compile(  # 编译正则：匹配旧版键名 ...denselayerN.(norm|relu|conv).(1|2).(参数名)
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')  # 正则模式串与 re.compile 调用闭合（整条语句结束）
        state_dict = model_zoo.load_url(model_urls['densenet%d' % num_layers])  # 从官方 URL 下载并读入权重
        for key in list(state_dict.keys()):  # 遍历所有键（list() 先拷贝，避免边遍历边删）
            res = pattern.match(key)  # 尝试匹配旧键名
            if res:  # 命中旧命名
                new_key = res.group(1) + res.group(2)  # 拼出新键名（去掉中间的点：norm.1 → norm1）
                state_dict[new_key] = state_dict[key]  # 把权重迁移到新键
                del state_dict[key]  # 删除旧键
        model.load_state_dict(state_dict)  # 载入改写后的权重
    return model  # 返回模型


def get_dilated_densenet(num_layers, dilate_scale, pretrained=False, **kwargs):  # 构建空洞版；【注意】**kwargs 未透传给 DilatedDenseNet，norm_layer 等参数会被静默忽略
    num_init_features, growth_rate, block_config = densenet_spec[num_layers]  # 按层数查规格表
    model = DilatedDenseNet(growth_rate, block_config, num_init_features, dilate_scale=dilate_scale)  # 构建空洞版 DenseNet（输出步长 dilate_scale）
    if pretrained:  # 加载预训练权重（键名改写流程与 get_densenet 相同）
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        # （上方四行为原代码英文注释，原样保留；【注意】“pervious”为原文拼写错误，应为 previous）
        pattern = re.compile(  # 编译正则：匹配旧版键名（同上）
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')  # 正则模式串与 re.compile 调用闭合（整条语句结束）
        state_dict = model_zoo.load_url(model_urls['densenet%d' % num_layers])  # 从官方 URL 下载并读入权重
        for key in list(state_dict.keys()):  # 遍历所有键
            res = pattern.match(key)  # 尝试匹配旧键名
            if res:  # 命中旧命名
                new_key = res.group(1) + res.group(2)  # 拼出新键名
                state_dict[new_key] = state_dict[key]  # 迁移权重
                del state_dict[key]  # 删除旧键
        model.load_state_dict(state_dict)  # 载入改写后的权重
    return model  # 返回模型


def densenet121(**kwargs):  # DenseNet-121 工厂函数
    return get_densenet(121, **kwargs)  # 构建 121 层规格（k=32，Block 6/12/24/16）


def densenet161(**kwargs):  # DenseNet-161 工厂函数
    return get_densenet(161, **kwargs)  # 构建 161 层规格（k=48，Block 6/12/36/24）


def densenet169(**kwargs):  # DenseNet-169 工厂函数
    return get_densenet(169, **kwargs)  # 构建 169 层规格（k=32，Block 6/12/32/32）


def densenet201(**kwargs):  # DenseNet-201 工厂函数
    return get_densenet(201, **kwargs)  # 构建 201 层规格（k=32，Block 6/12/48/32）


def dilated_densenet121(dilate_scale, **kwargs):  # 空洞版 DenseNet-121 工厂函数（dilate_scale——输出步长 8 或 16）
    return get_dilated_densenet(121, dilate_scale, **kwargs)  # 构建空洞版 121


def dilated_densenet161(dilate_scale, **kwargs):  # 空洞版 DenseNet-161 工厂函数
    return get_dilated_densenet(161, dilate_scale, **kwargs)  # 构建空洞版 161


def dilated_densenet169(dilate_scale, **kwargs):  # 空洞版 DenseNet-169 工厂函数
    return get_dilated_densenet(169, dilate_scale, **kwargs)  # 构建空洞版 169


def dilated_densenet201(dilate_scale, **kwargs):  # 空洞版 DenseNet-201 工厂函数
    return get_dilated_densenet(201, dilate_scale, **kwargs)  # 构建空洞版 201


if __name__ == '__main__':  # 直接运行本文件时的 smoke test（只做前向，不训练）
    img = torch.randn(2, 3, 224, 224)  # 构造随机输入 (2,3,224,224)
    model = dilated_densenet121(8)  # 构建输出步长 8 的空洞版 DenseNet-121（不加载预训练权重）
    outputs = model(img)  # 前向推理：输出 (2,1000) 分类 logits
