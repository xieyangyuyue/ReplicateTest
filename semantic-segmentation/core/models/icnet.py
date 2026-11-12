# ============================================================================
# 【文件说明】icnet.py —— 实现 ICNet（Image Cascade Network，图像级联网络），
#              工程位置 core/models/icnet.py，包含 ICNet 主网络、金字塔池化模块
#              （PPM）、级联特征融合单元（CFF）、_ICHead 头部及 get_icnet 工厂函数。
# 【核心思想】把同一幅图缩放成三个尺度的级联输入：
#            1/4 小图走完整（空洞）ResNet + 金字塔池化 PPM 提取高层语义（sub4 分支）；
#            1/2 中图共享同一 ResNet 权重、只取到 c2 层，得到高一级分辨率的特征（sub2 分支）；
#            原图只过 3 层轻量卷积保细节（sub1 分支）；随后用两个 CFF 单元自粗到细
#            逐级融合（低分辨率分支上采样对齐后与高分辨率分支相加），
#            以极小的计算代价兼顾高分辨率精度与实时性。
# 【论文名称】H. Zhao, X. Qi, X. Shen, J. Shi, J. Jia. "ICNet for Real-Time
#            Semantic Segmentation on High-Resolution Images." ECCV 2018
#            (arXiv:1704.08545).
# ============================================================================
"""Image Cascade Network"""  # 模块级英文 docstring，原样保留
import torch  # PyTorch 主库：权重加载 torch.load、设备指定 torch.device
import torch.nn as nn  # 神经网络模块库：Conv2d、BatchNorm2d、Sequential 等
import torch.nn.functional as F  # 函数式接口：F.interpolate 上/下采样、F.adaptive_avg_pool2d 自适应池化

from .segbase import SegBaseModel  # 导入分割模型公共基类（构建 ResNet 骨干并提供 base_forward 接口）

__all__ = ['ICNet', 'get_icnet', 'get_icnet_resnet50_citys',  # 模块公开接口列表开始（与下一行共同构成完整列表）
           'get_icnet_resnet101_citys', 'get_icnet_resnet152_citys']  # 模块公开接口（列表跨两行书写）


class ICNet(SegBaseModel):  # ICNet 主网络类：继承基类获得（空洞）ResNet 骨干
    """Image Cascade Network"""  # 原英文类 docstring，原样保留

    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数
    # backbone：ResNet 骨干类型，默认 'resnet50'（另支持 'resnet101' / 'resnet152'）
    # aux：辅助损失开关（本类未单独实现辅助分支，仅接口保留）
    # jpu：JPU 开关【注意】该参数未透传给基类，实际恒为空洞骨干（dilated, 输出步长 8）模式
    # pretrained_base：是否加载骨干的 ImageNet 预训练权重
    # **kwargs：透传给基类与各子模块（如 norm_layer）
    def __init__(self, nclass, backbone='resnet50', aux=False, jpu=False, pretrained_base=True, **kwargs):  # 构造函数
        super(ICNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)  # 基类构造：构建空洞 ResNet（layer3/4 用空洞卷积，c3/c4 保持 1/8 分辨率）
        self.conv_sub1 = nn.Sequential(  # sub1 分支：处理原图的轻量卷积栈（3 次 stride-2 下采样，共 1/8）
            _ConvBNReLU(3, 32, 3, 2, **kwargs),  # 3x3 卷积 stride=2（默认 padding=1）：3 -> 32 通道，分辨率 1/2
            _ConvBNReLU(32, 32, 3, 2, **kwargs),  # 3x3/s2：32 -> 32 通道，分辨率 1/4
            _ConvBNReLU(32, 64, 3, 2, **kwargs)  # 3x3/s2：32 -> 64 通道，分辨率 1/8
        )  # conv_sub1 轻量卷积栈结束

        self.ppm = PyramidPoolingModule()  # 金字塔池化模块：挂在 sub4 分支（骨干 c4 特征）之后做多尺度上下文增强

        self.head = _ICHead(nclass, **kwargs)  # ICNet 头部：两个级联特征融合（CFF）单元 + 最终分类卷积

        self.__setattr__('exclusive', ['conv_sub1', 'head'])  # 训练时"独占新参数"的子模块列表（骨干 self.pretrained 使用预训练权重，不在此列）

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像 (N, 3, H, W)
    # 返回：4 个输出的元组，分辨率依次为 1x、1/4、1/8、1/16（第 0 个为主输出，其余为训练期辅助监督）
    def forward(self, x):  # 前向：三路级联输入 -> PPM -> 逐级 CFF 融合
        # sub 1
        x_sub1 = self.conv_sub1(x)  # 原图分支：(N,3,H,W) -> (N,64,H/8,W/8)

        # sub 2
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)  # 原图双线性下采样 1/2：-> (N,3,H/2,W/2)
        _, x_sub2, _, _ = self.base_forward(x_sub2)  # 骨干前向取第 2 个特征 c2：512 通道、(H/2)/8 = H/16 分辨率（骨干为空洞模式、输出步长 8）

        # sub 4
        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)  # 原图双线性下采样 1/4：-> (N,3,H/4,W/4)
        _, _, _, x_sub4 = self.base_forward(x_sub4)  # 骨干前向取第 4 个特征 c4：2048 通道、(H/4)/8 = H/32 分辨率
        # add PyramidPoolingModule
        x_sub4 = self.ppm(x_sub4)   # PPM 多尺度上下文融合（加法式），形状不变 (N,2048,H/32,W/32)
        outputs = self.head(x_sub1, x_sub2, x_sub4)  # 级联融合头部：返回 4 个尺度的预测图列表

        return tuple(outputs)  # 转元组返回，与其他模型的多输出接口统一
class PyramidPoolingModule(nn.Module):  # 金字塔池化模块（本文件为"逐元素相加"的轻量变体）
    # ----------------------- __init__ 参数说明 -----------------------
    # pyramids：自适应平均池化的分箱尺寸列表 [1,2,3,6]，对应 1x1/2x2/3x3/6x6 四个尺度
    def __init__(self, pyramids=[1,2,3,6]):  # 构造函数【注意】可变对象（列表）作默认参数是 Python 风险写法，所幸本类未修改该列表
        super(PyramidPoolingModule, self).__init__()  # 父类构造
        self.pyramids = pyramids  # 保存池化分箱尺寸列表

    def forward(self, input):  # 前向：多尺度池化 -> 上采样回原尺寸 -> 逐元素累加【注意】形参名 input 遮蔽了 Python 内建函数 input
        feat = input  # feat 作为累加器，初始值为输入特征
        height, width = input.shape[2:]  # 取输入特征的高、宽
        for bin_size in self.pyramids:  # 依次以 1/2/3/6 的分箱尺寸做池化
            x = F.adaptive_avg_pool2d(input, output_size=bin_size)  # 自适应平均池化到 bin_size x bin_size（任意输入尺寸均可）
            x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)  # 双线性上采样回原特征尺寸
            feat  = feat + x  # 逐元素累加【注意】与 PSPNet 原版的"拼接 + 1x1 降维"不同，此处直接相加（更省参数，适合实时网络）
        return feat  # 返回累加后的特征，形状与输入相同
class _ICHead(nn.Module):  # ICNet 头部：两个 CFF 单元自粗到细级联融合 + 最终分类卷积
    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数
    # norm_layer：归一化层类型，默认 nn.BatchNorm2d
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(_ICHead, self).__init__()  # 父类构造，初始化模块注册表
        #self.cff_12 = CascadeFeatureFusion(512, 64, 128, nclass, norm_layer, **kwargs)
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, nclass, norm_layer, **kwargs)  # CFF 单元 2：低分辨率输入 128 通道（来自 cff_24 的输出）、高分辨率输入 64 通道（sub1 分支），输出 128 通道（上一行为被注释的旧版定义，原样保留）
        self.cff_24 = CascadeFeatureFusion(2048, 512, 128, nclass, norm_layer, **kwargs)  # CFF 单元 1：低分辨率输入 2048 通道（sub4 经 PPM）、高分辨率输入 512 通道（sub2），输出 128 通道

        self.conv_cls = nn.Conv2d(128, nclass, 1, bias=False)  # 最终分类卷积：1x1，128 -> nclass 通道，无 bias

    # ----------------------- forward 参数说明 -----------------------
    # x_sub1：(N,64,H/8,W/8) 原图轻量分支特征
    # x_sub2：(N,512,H/16,W/16) 中尺度（1/2 图）骨干 c2 特征
    # x_sub4：(N,2048,H/32,W/32) 小尺度（1/4 图）骨干 c4 + PPM 特征
    # 返回：列表 [1x, 1/4, 1/8, 1/16] 四个尺度的预测图（第 0 个为主输出）
    def forward(self, x_sub1, x_sub2, x_sub4):  # 前向：两次 CFF 融合，再逐级上采样输出
        outputs = list()  # 输出收集列表
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)  # 第一次融合：sub4 上采样到 sub2 尺寸（H/16）后相加 -> 融合特征 (N,128,H/16) + 辅助分类图 (N,nclass,H/16)
        outputs.append(x_24_cls)  # 收集最粗尺度（1/16）辅助输出
        #x_cff_12, x_12_cls = self.cff_12(x_sub2, x_sub1)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)  # 第二次融合：cff_24 结果上采样到 sub1 尺寸（H/8）后与 sub1 相加 -> (N,128,H/8) + 辅助分类图（上一行为被注释的旧版调用，原样保留）
        outputs.append(x_12_cls)  # 收集 1/8 尺度辅助输出

        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)  # 融合特征上采样 2 倍：H/8 -> H/4
        up_x2 = self.conv_cls(up_x2)  # 1x1 分类卷积：-> (N,nclass,H/4,W/4)，主输出
        outputs.append(up_x2)  # 收集主输出（1/4）
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear', align_corners=True)  # 主输出再上采样 4 倍：H/4 -> H（全分辨率）
        outputs.append(up_x8)  # 收集全分辨率输出
        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()  # 反转列表：由 [1/16, 1/8, 1/4, 1x] 变为 [1x, 1/4, 1/8, 1/16]，主输出（最大尺度）排在首位

        return outputs  # 返回 4 个尺度的输出列表


class _ConvBNReLU(nn.Module):  # "卷积 + BN + ReLU"基础单元【注意】与 core/nn/basic.py 中的同名类功能重复，此处为文件内私有副本（供 conv_sub1 使用）
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels / out_channels：输入/输出通道数
    # kernel_size / stride / padding / dilation：卷积核尺寸/步长/填充/空洞率（本文件按 3x3/s2/p1 使用）
    # groups：分组卷积组数（默认 1 即普通卷积）
    # norm_layer：归一化层类型；bias：卷积偏置开关（默认 False）
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, norm_layer=nn.BatchNorm2d, bias=False, **kwargs):  # 构造函数（参数跨两行书写）
        super(_ConvBNReLU, self).__init__()  # 父类构造
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)  # 通用卷积：按位置依次传入 kernel/stride/padding/dilation/groups/bias
        self.bn = norm_layer(out_channels)  # BN
        self.relu = nn.ReLU(True)  # 就地 ReLU

    def forward(self, x):  # 前向：卷积 -> BN -> ReLU
        x = self.conv(x)  # 卷积，通道 in -> out
        x = self.bn(x)  # 批归一化
        x = self.relu(x)  # ReLU 激活
        return x  # 返回结果


class CascadeFeatureFusion(nn.Module):  # 级联特征融合单元（CFF）：低分辨率语义上采样对齐后与高分辨率细节相加
    """CFF Unit"""  # 原英文 docstring，原样保留

    # ----------------------- __init__ 参数说明 -----------------------
    # low_channels：低分辨率（语义）分支输入通道数
    # high_channels：高分辨率（细节）分支输入通道数
    # out_channels：融合后的输出通道数（ICNet 中为 128）
    # nclass：类别数（低分辨率分支的辅助分类器使用）
    # norm_layer：归一化层类型，默认 nn.BatchNorm2d
    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(CascadeFeatureFusion, self).__init__()  # 父类构造
        self.conv_low = nn.Sequential(  # 低分辨率分支：空洞卷积 + BN
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),  # 3x3 卷积，空洞率 2、padding 2（尺寸不变）：low -> 128 通道
            norm_layer(out_channels)  # BN
        )  # conv_low 低分辨率分支子模块结束
        self.conv_high = nn.Sequential(  # 高分辨率分支：1x1 卷积 + BN（本分支保持分辨率不变）
            nn.Conv2d(high_channels, out_channels, 1, bias=False),  # 1x1 卷积：high -> 128 通道
            norm_layer(out_channels)  # BN
        )  # conv_high 高分辨率分支子模块结束
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)  # 辅助分类器：对低分辨率分支特征做 1x1 分类（训练期深层监督）

    # ----------------------- forward 参数说明 -----------------------
    # x_low：低分辨率特征（将被上采样到 x_high 的尺寸）
    # x_high：高分辨率特征（融合的对齐基准）
    # 返回：(融合特征 x, 低分辨率分支的辅助分类图 x_low_cls)
    def forward(self, x_low, x_high):  # 前向：上采样对齐 -> 双分支卷积 -> 相加融合
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)  # 低分辨率分支双线性上采样到高分辨率分支尺寸（级联融合的关键对齐步骤）
        x_low = self.conv_low(x_low)  # 空洞卷积变换：-> (N,128,H_high,W_high)
        x_high = self.conv_high(x_high)  # 1x1 卷积变换：-> (N,128,H_high,W_high)
        x = x_low + x_high  # 逐元素相加，完成级联融合
        x = F.relu(x, inplace=True)  # 就地 ReLU
        x_low_cls = self.conv_low_cls(x_low)  # 对融合前的低分辨率分支做辅助分类：-> (N,nclass,H_high,W_high)

        return x, x_low_cls  # 返回融合特征与辅助分类图


# ----------------------- get_icnet 参数说明 -----------------------
# dataset：数据集名称（默认 'citys' 即 Cityscapes）
# backbone：ResNet 骨干类型（默认 'resnet50'）
# pretrained：是否加载本仓库提供的模型权重文件
# root：权重文件的下载/缓存目录
# pretrained_base：是否加载骨干的 ImageNet 预训练权重
# **kwargs：透传给模型构造（如 local_rank、norm_layer）
def get_icnet(dataset='citys', backbone='resnet50', pretrained=False, root='~/.torch/models',
              pretrained_base=True, **kwargs):  # ICNet 工厂函数（参数跨两行书写）
    acronyms = {  # 数据集名 -> 预训练权重文件名中的缩写映射表
        'pascal_voc': 'pascal_voc',  # Pascal VOC 数据集缩写
        'pascal_aug': 'pascal_aug',  # Pascal VOC 增强标注集缩写
        'ade20k': 'ade',  # ADE20K 场景解析数据集缩写
        'coco': 'coco',  # COCO 数据集缩写
        'citys': 'citys',  # Cityscapes 街景数据集缩写
    }  # acronyms 字典结束
    from ..data.dataloader import datasets  # 延迟导入数据集注册表（相对导入）
    model = ICNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按类别数与骨干类型构建 ICNet
    if pretrained:  # 需要加载预训练权重时
        from .model_store import get_model_file  # 延迟导入权重文件定位/下载工具
        device = torch.device(kwargs['local_rank'])  # 【注意】直接取 kwargs['local_rank']，未传该键时会抛 KeyError
        model.load_state_dict(torch.load(get_model_file('icnet_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 权重名含骨干（如 icnet_resnet50_citys）【注意】torch.load 未指定 weights_only，新版 PyTorch 会有告警
                              map_location=device))  # 权重张量映射到指定设备（跨两行的调用）
    return model  # 返回构建好的模型


def get_icnet_resnet50_citys(**kwargs):  # Cityscapes + ResNet50 快捷工厂函数
    return get_icnet('citys', 'resnet50', **kwargs)  # 固定数据集与骨干类型


def get_icnet_resnet101_citys(**kwargs):  # Cityscapes + ResNet101 快捷工厂函数
    return get_icnet('citys', 'resnet101', **kwargs)  # 固定数据集与骨干类型


def get_icnet_resnet152_citys(**kwargs):  # Cityscapes + ResNet152 快捷工厂函数
    return get_icnet('citys', 'resnet152', **kwargs)  # 固定数据集与骨干类型


if __name__ == '__main__':  # 模块自测入口：直接运行本文件时执行
    img = torch.randn(1, 3, 256, 256)  # 构造随机输入 (1,3,256,256)
    model = get_icnet_resnet50_citys()  # 构建 Cityscapes + ResNet50 版 ICNet
    outputs = model(img)  # 前向传播一次（返回 4 个尺度的输出元组）
