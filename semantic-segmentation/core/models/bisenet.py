# ============================================================================
# 【文件说明】bisenet.py —— 双边（双路径）分割网络 BiSeNet 的 PyTorch 实现。
#              模型由三部分组成：空间路径 SpatialPath（保留空间细节）、
#              上下文路径 ContextPath（ResNet18 骨干 + 注意力精炼模块 ARM + 全局上下文）、
#              特征融合模块 FeatureFusion（FFM，通道注意力加权融合两路特征），最后接轻量分割头。
#              工程位置：core/models/bisenet.py，骨干仅支持 resnet18。
# 【核心思想】论文《BiSeNet: Bilateral Segmentation Network for Real-time Semantic
#            Segmentation》(ECCV 2018, Changqian Yu et al.)。实时分割的矛盾：
#            大步长下采样丢失空间细节、小步长又耗时。BiSeNet 用一条浅层小步长网络
#            专管空间细节（1/8 分辨率、少量通道），用一条大步长轻量骨干专管语义上下文
#            （快速降到 1/32 再自顶向下精炼），两路并行、FFM 融合，兼顾速度与精度。
# ============================================================================
"""Bilateral Segmentation Network"""  # 原英文 docstring 保留：双边（双路径）分割网络
import torch  # PyTorch 主库：张量运算（torch.cat / torch.load / torch.device 等）
import torch.nn as nn  # 神经网络模块：nn.Module 基类及卷积/BN/激活/池化等网络层
import torch.nn.functional as F  # 函数式接口：双线性上采样 F.interpolate 等

from core.models.base_models.resnet import resnet18  # 导入 ResNet18 骨干构造函数（上下文路径的骨干）
from core.nn import _ConvBNReLU  # 导入通用“卷积+BN+ReLU”基础模块（定义在 core/nn/basic.py）

__all__ = ['BiSeNet', 'get_bisenet', 'get_bisenet_resnet18_citys']  # 模块公开接口：模型类 + 两个工厂函数


class BiSeNet(nn.Module):  # BiSeNet 主网络：空间路径 + 上下文路径 + 特征融合 + 分割头
    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数（含背景），决定分割头输出通道数
    # backbone：骨干类型，本实现仅支持 'resnet18'，其他取值会抛异常
    # aux：是否启用辅助损失分支（对上下文路径两个中间输出各接一个分割头做深监督）
    # jpu：未使用，仅为与其他模型统一的接口参数而保留【注意】
    # pretrained_base：是否加载骨干的 ImageNet 预训练权重
    # **kwargs：透传给子模块（如 norm_layer 归一化层类型）
    def __init__(self, nclass, backbone='resnet18', aux=False, jpu=False, pretrained_base=True, **kwargs):  # 构造函数：搭建双路径结构与输出头
        super(BiSeNet, self).__init__()  # 调用父类 nn.Module 构造函数，初始化模块树
        self.aux = aux  # 保存辅助分支开关，forward 中据此决定输出个数
        self.spatial_path = SpatialPath(3, 128, **kwargs)  # 空间路径：3→128 通道，连续 3 次 stride=2 下采样，输出 1/8 分辨率细节特征
        self.context_path = ContextPath(backbone, pretrained_base, **kwargs)  # 上下文路径：ResNet18 骨干 + 2 个 ARM 精炼模块 + 全局平均池化上下文
        self.ffm = FeatureFusion(256, 256, 4, **kwargs)  # 特征融合模块 FFM：空间路(128)+上下文路(128) 拼接共 256 通道 → 融合输出 256；reduction=4 为注意力瓶颈通道压缩比
        self.head = _BiSeHead(256, 64, nclass, **kwargs)  # 主分割头：256→64→nclass
        if aux:  # 启用辅助监督时，为上下文路径两个尺度的特征各建一个分割头
            self.auxlayer1 = _BiSeHead(128, 256, nclass, **kwargs)  # 辅助头 1：作用于 1/16 分辨率上下文特征（128 通道），瓶颈通道 256
            self.auxlayer2 = _BiSeHead(128, 256, nclass, **kwargs)  # 辅助头 2：作用于 1/8 分辨率上下文特征（128 通道），瓶颈通道 256

        self.__setattr__('exclusive',  # 登记“模型自有模块”名单（区别于预训练骨干），供训练脚本区分参数组（如只对 exclusive 做权重衰减）
                         ['spatial_path', 'context_path', 'ffm', 'head', 'auxlayer1', 'auxlayer2'] if aux else [  # aux=True 时的名单：含两个辅助头
                             'spatial_path', 'context_path', 'ffm', 'head'])  # aux=False 时的名单：仅四个主干模块

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像张量 (N, 3, H, W)
    # 返回：元组 (主输出[, 辅助输出1, 辅助输出2])，每个输出均为 (N, nclass, H, W)
    def forward(self, x):  # 前向传播：双路径并行提取 → FFM 融合 → 分割头 → 上采样回原尺寸
        size = x.size()[2:]  # 记录输入空间尺寸 (H, W)，供最终上采样对齐
        spatial_out = self.spatial_path(x)  # 空间路径输出 → (N, 128, H/8, W/8)，保留空间细节
        context_out = self.context_path(x)  # 上下文路径输出列表：[(N,128,H/16,W/16), (N,128,H/8,W/8)]（浅→深）
        fusion_out = self.ffm(spatial_out, context_out[-1])  # FFM 融合：空间路 (N,128,H/8) 与上下文路最深输出 (N,128,H/8) 拼接并做通道注意力加权 → (N, 256, H/8, W/8)
        outputs = []  # 输出收集列表：主输出在前，辅助输出在后
        x = self.head(fusion_out)  # 主分割头 → (N, nclass, H/8, W/8)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 双线性上采样（align_corners=True 对齐角点）→ (N, nclass, H, W)
        outputs.append(x)  # 收集主输出

        if self.aux:  # 启用辅助损失分支时
            auxout1 = self.auxlayer1(context_out[0])  # 辅助头 1 作用于 1/16 分辨率上下文特征 → (N, nclass, H/16, W/16)
            auxout1 = F.interpolate(auxout1, size, mode='bilinear', align_corners=True)  # 上采样回原尺寸 → (N, nclass, H, W)
            outputs.append(auxout1)  # 收集辅助输出 1
            auxout2 = self.auxlayer2(context_out[1])  # 辅助头 2 作用于 1/8 分辨率上下文特征 → (N, nclass, H/8, W/8)
            auxout2 = F.interpolate(auxout2, size, mode='bilinear', align_corners=True)  # 上采样回原尺寸 → (N, nclass, H, W)
            outputs.append(auxout2)  # 收集辅助输出 2
        return tuple(outputs)  # 返回 (主输出,) 或 (主输出, 辅助1, 辅助2) 元组


class _BiSeHead(nn.Module):  # BiSeNet 轻量分割头：3x3 卷积降维 + Dropout + 1x1 分类卷积
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：输入通道数（主头 256 / 辅助头 128）
    # inter_channels：瓶颈中间通道数（主头 64 / 辅助头 256）
    # nclass：分割类别数
    # norm_layer：归一化层类型，默认 BatchNorm2d
    # **kwargs：兼容上层透传（本类中未使用）
    def __init__(self, in_channels, inter_channels, nclass, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(_BiSeHead, self).__init__()  # 调用父类构造函数
        self.block = nn.Sequential(  # 顺序组合的分割头块
            _ConvBNReLU(in_channels, inter_channels, 3, 1, 1, norm_layer=norm_layer),  # 3x3 卷积 stride=1 padding=1（分辨率不变），in→inter 通道，带 BN+ReLU
            nn.Dropout(0.1),  # Dropout p=0.1，防过拟合（分割头常用值）
            nn.Conv2d(inter_channels, nclass, 1)  # 1x1 卷积投影到类别数 → (N, nclass, H', W') 得分图
        )  # 分割头 Sequential 块结束

    # ----------------------- forward 参数说明 -----------------------
    # x：输入特征 (N, in_channels, H', W')；输出 (N, nclass, H', W')
    def forward(self, x):  # 前向：直接过 block
        x = self.block(x)  # 通道 in→inter→nclass，空间分辨率不变
        return x  # 返回分割得分图


class SpatialPath(nn.Module):  # 空间路径：浅层小步长网络，负责保留空间细节
    """Spatial path"""  # 原英文 docstring 保留：空间路径

    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：输入通道数（RGB 图像为 3）
    # out_channels：输出通道数（BiSeNet 中为 128）
    # norm_layer：归一化层类型
    # **kwargs：透传给 _ConvBNReLU
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(SpatialPath, self).__init__()  # 调用父类构造函数
        inter_channels = 64  # 中间通道数 64：空间路径用少量通道换取推理速度
        self.conv7x7 = _ConvBNReLU(in_channels, inter_channels, 7, 2, 3, norm_layer=norm_layer)  # 7x7 卷积 stride=2 padding=3 → (N, 64, H/2, W/2)，大核扩大感受野
        self.conv3x3_1 = _ConvBNReLU(inter_channels, inter_channels, 3, 2, 1, norm_layer=norm_layer)  # 3x3 卷积 stride=2 padding=1 → (N, 64, H/4, W/4)
        self.conv3x3_2 = _ConvBNReLU(inter_channels, inter_channels, 3, 2, 1, norm_layer=norm_layer)  # 3x3 卷积 stride=2 padding=1 → (N, 64, H/8, W/8)
        self.conv1x1 = _ConvBNReLU(inter_channels, out_channels, 1, 1, 0, norm_layer=norm_layer)  # 1x1 卷积升维 64→128，分辨率保持 1/8

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像 (N, 3, H, W)；输出 (N, out_channels, H/8, W/8)
    def forward(self, x):  # 前向：连续三次 stride=2 下采样 + 1x1 升维
        x = self.conv7x7(x)  # → (N, 64, H/2, W/2)
        x = self.conv3x3_1(x)  # → (N, 64, H/4, W/4)
        x = self.conv3x3_2(x)  # → (N, 64, H/8, W/8)
        x = self.conv1x1(x)  # → (N, 128, H/8, W/8)

        return x  # 返回 1/8 分辨率的空间细节特征


class _GlobalAvgPooling(nn.Module):  # 全局平均池化上下文模块：把整幅图压缩成全局上下文向量再广播回原尺寸
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：输入通道数（ResNet18 layer4 输出为 512）
    # out_channels：输出通道数（128）
    # norm_layer：归一化层类型（此处按位置参数传入）
    # **kwargs：兼容透传（本类中未使用）
    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):  # 构造函数
        super(_GlobalAvgPooling, self).__init__()  # 调用父类构造函数
        self.gap = nn.Sequential(  # 顺序组合：全局池化 → 1x1 卷积降维 → BN → ReLU
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化到 1x1 → (N, in_channels, 1, 1)
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1x1 卷积 512→128（无偏置，后接 BN）
            norm_layer(out_channels),  # 归一化层
            nn.ReLU(True)  # 就地 ReLU 激活
        )  # 全局上下文提取 Sequential 块结束

    # ----------------------- forward 参数说明 -----------------------
    # x：输入特征 (N, in_channels, H', W')；输出 (N, out_channels, H', W')（广播回原尺寸）
    def forward(self, x):  # 前向：全局池化后双线性广播回输入尺寸
        size = x.size()[2:]  # 记录输入空间尺寸 (H', W')
        pool = self.gap(x)  # 提取全局上下文 → (N, 128, 1, 1)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)  # 双线性上采样回 (N, 128, H', W')，作为全局先验
        return out  # 返回与输入同尺寸的全局上下文特征


class AttentionRefinmentModule(nn.Module):  # 注意力精炼模块 ARM（类名 Refinment 为原代码拼写，正确应为 Refinement）【注意】
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：输入通道数（c4: 512 / c3: 256）
    # out_channels：输出通道数（统一 128）
    # norm_layer：归一化层类型
    # **kwargs：透传给 _ConvBNReLU
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(AttentionRefinmentModule, self).__init__()  # 调用父类构造函数
        self.conv3x3 = _ConvBNReLU(in_channels, out_channels, 3, 1, 1, norm_layer=norm_layer)  # 3x3 卷积 stride=1 padding=1，通道 in→128，分辨率不变
        self.channel_attention = nn.Sequential(  # 通道注意力分支：GAP → 1x1 卷积 → Sigmoid
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → (N, 128, 1, 1)
            _ConvBNReLU(out_channels, out_channels, 1, 1, 0, norm_layer=norm_layer),  # 1x1 卷积（128→128）+ BN + ReLU，学习各通道重要性
            nn.Sigmoid()  # Sigmoid 压缩到 (0,1) 区间作为通道权重
        )  # 通道注意力 Sequential 块结束

    # ----------------------- forward 参数说明 -----------------------
    # x：输入特征 (N, in_channels, H', W')；输出 (N, out_channels, H', W')
    def forward(self, x):  # 前向：卷积降维后用通道注意力加权精炼
        x = self.conv3x3(x)  # 卷积降维 → (N, 128, H', W')
        attention = self.channel_attention(x)  # 计算通道注意力权重 → (N, 128, 1, 1)
        x = x * attention  # 广播相乘：按通道加权特征（注意力精炼）
        return x  # 返回精炼后的特征


class ContextPath(nn.Module):  # 上下文路径：ResNet18 骨干 + 全局上下文 + 自顶向下 ARM 精炼
    # ----------------------- __init__ 参数说明 -----------------------
    # backbone：骨干类型，仅支持 'resnet18'
    # pretrained_base：是否加载骨干 ImageNet 预训练权重
    # norm_layer：归一化层类型
    # **kwargs：透传给 resnet18 构造函数与各子模块
    def __init__(self, backbone='resnet18', pretrained_base=True, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(ContextPath, self).__init__()  # 调用父类构造函数
        if backbone == 'resnet18':  # 骨干类型分支判断
            pretrained = resnet18(pretrained=pretrained_base, **kwargs)  # 构建 ResNet18（可选加载 ImageNet 预训练权重），变量名 pretrained 此处指骨干模块
        else:  # 其他骨干类型
            raise RuntimeError('unknown backbone: {}'.format(backbone))  # 抛出运行时错误：不支持的骨干
        self.conv1 = pretrained.conv1  # 取骨干 stem 的 7x7 卷积（stride=2）
        self.bn1 = pretrained.bn1  # 取骨干 stem 的 BN 层
        self.relu = pretrained.relu  # 取骨干 stem 的 ReLU
        self.maxpool = pretrained.maxpool  # 取骨干 stem 的 3x3 最大池化（stride=2）
        self.layer1 = pretrained.layer1  # stage1：输出 64 通道，1/4 分辨率
        self.layer2 = pretrained.layer2  # stage2：输出 128 通道，1/8 分辨率
        self.layer3 = pretrained.layer3  # stage3：输出 256 通道，1/16 分辨率
        self.layer4 = pretrained.layer4  # stage4：输出 512 通道，1/32 分辨率

        inter_channels = 128  # 上下文路径统一的中间通道数 128
        self.global_context = _GlobalAvgPooling(512, inter_channels, norm_layer)  # 全局上下文：对 layer4 输出（512 通道）池化降维到 128

        self.arms = nn.ModuleList(  # 两个注意力精炼模块，分别作用于最深的两个尺度特征
            [AttentionRefinmentModule(512, inter_channels, norm_layer, **kwargs),  # ARM1：作用于 c4（512 通道，1/32 分辨率）
             AttentionRefinmentModule(256, inter_channels, norm_layer, **kwargs)]  # ARM2：作用于 c3（256 通道，1/16 分辨率）
        )  # ARM 注意力精炼模块列表 ModuleList 结束
        self.refines = nn.ModuleList(  # 两个精炼卷积（3x3，128→128），作用于上采样后的特征
            [_ConvBNReLU(inter_channels, inter_channels, 3, 1, 1, norm_layer=norm_layer),  # refine1：3x3 stride=1 padding=1
             _ConvBNReLU(inter_channels, inter_channels, 3, 1, 1, norm_layer=norm_layer)]  # refine2：3x3 stride=1 padding=1
        )  # 精炼卷积模块列表 ModuleList 结束

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像 (N, 3, H, W)
    # 返回：列表 [(N,128,H/16,W/16), (N,128,H/8,W/8)]（浅→深两个精炼输出）
    def forward(self, x):  # 前向：骨干提特征 → 全局上下文 + 自顶向下 ARM 逐级融合精炼
        x = self.conv1(x)  # 7x7 卷积 stride=2 → (N, 64, H/2, W/2)
        x = self.bn1(x)  # BN，通道不变
        x = self.relu(x)  # ReLU，通道不变
        x = self.maxpool(x)  # 3x3 最大池化 stride=2 → (N, 64, H/4, W/4)
        x = self.layer1(x)  # stage1 → (N, 64, H/4, W/4)

        context_blocks = []  # 收集骨干各阶段特征（先按浅→深顺序暂存）
        context_blocks.append(x)  # 存入 c1: (N, 64, H/4, W/4)
        x = self.layer2(x)  # stage2 → (N, 128, H/8, W/8)
        context_blocks.append(x)  # 存入 c2: (N, 128, H/8, W/8)
        c3 = self.layer3(x)  # stage3 → (N, 256, H/16, W/16)
        context_blocks.append(c3)  # 存入 c3
        c4 = self.layer4(c3)  # stage4 → (N, 512, H/32, W/32)
        context_blocks.append(c4)  # 存入 c4
        context_blocks.reverse()  # 反转为深→浅顺序 [c4, c3, c2, c1]，便于自顶向下融合

        global_context = self.global_context(c4)  # 全局上下文 → (N, 128, H/32, W/32)
        last_feature = global_context  # 自顶向下传播的“上层特征”，初值为全局上下文
        context_outputs = []  # 输出收集列表
        for i, (feature, arm, refine) in enumerate(zip(context_blocks[:2], self.arms, self.refines)):  # 取最深的两个尺度 [c4, c3] 依次精炼（c2、c1 仅用于确定上采样尺寸）
            feature = arm(feature)  # ARM 注意力精炼 → (N, 128, H/32) 或 (N, 128, H/16)
            feature += last_feature  # 与上层传来的特征逐元素相加（融入全局上下文/更深层语义）
            last_feature = F.interpolate(feature, size=context_blocks[i + 1].size()[2:],  # 上采样到更浅一级特征的尺寸（i=0 时对齐 c3，i=1 时对齐 c2）
                                         mode='bilinear', align_corners=True)  # 双线性插值，align_corners=True 对齐角点
            last_feature = refine(last_feature)  # 3x3 精炼卷积，通道保持 128
            context_outputs.append(last_feature)  # 收集该尺度输出

        return context_outputs  # 返回 [(N,128,H/16,W/16), (N,128,H/8,W/8)]


class FeatureFusion(nn.Module):  # 特征融合模块 FFM：空间路与上下文路拼接后经通道注意力自适应加权
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：拼接后的输入通道数（128+128=256）
    # out_channels：融合输出通道数（256）
    # reduction：注意力瓶颈压缩比（BiSeNet 中为 4：256→64→256）
    # norm_layer：归一化层类型
    # **kwargs：透传给 _ConvBNReLU
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(FeatureFusion, self).__init__()  # 调用父类构造函数
        self.conv1x1 = _ConvBNReLU(in_channels, out_channels, 1, 1, 0, norm_layer=norm_layer, **kwargs)  # 1x1 卷积 256→256（拼接后降回原通道数）
        self.channel_attention = nn.Sequential(  # 通道注意力：GAP → 降维 → 升维 → Sigmoid
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 → (N, 256, 1, 1)
            _ConvBNReLU(out_channels, out_channels // reduction, 1, 1, 0, norm_layer=norm_layer),  # 1x1 卷积降维 256→64
            _ConvBNReLU(out_channels // reduction, out_channels, 1, 1, 0, norm_layer=norm_layer),  # 1x1 卷积升维 64→256
            nn.Sigmoid()  # Sigmoid 得到 (0,1) 通道权重
        )  # 通道注意力 Sequential 块结束

    # ----------------------- forward 参数说明 -----------------------
    # x1：空间路径特征 (N, 128, H/8, W/8)；x2：上下文路径特征 (N, 128, H/8, W/8)
    # 返回：融合特征 (N, 256, H/8, W/8)
    def forward(self, x1, x2):  # 前向：拼接 → 1x1 融合 → 通道注意力加权
        fusion = torch.cat([x1, x2], dim=1)  # 通道维拼接空间路与上下文路 → (N, 256, H/8, W/8)
        out = self.conv1x1(fusion)  # 1x1 卷积融合 → (N, 256, H/8, W/8)
        attention = self.channel_attention(out)  # 计算通道注意力权重 → (N, 256, 1, 1)
        out = out + out * attention  # 加权融合：等价于 out*(1+attention)，自适应增强重要通道
        return out  # 返回融合后的特征


# ----------------------- get_bisenet 参数说明 -----------------------
# dataset：数据集名称（决定类别数），如 'citys'/'pascal_voc'/'ade20k' 等
# backbone：骨干类型，仅 'resnet18'
# pretrained：是否加载本模型在目标数据集上训练好的分割权重
# root：预训练权重文件查找根目录
# pretrained_base：是否加载骨干 ImageNet 预训练权重
# **kwargs：透传给 BiSeNet；pretrained=True 时须包含 local_rank 用于指定加载设备
def get_bisenet(dataset='citys', backbone='resnet18', pretrained=False, root='~/.torch/models',  # 工厂函数：按数据集构建 BiSeNet 并可选加载权重
                pretrained_base=True, **kwargs):  # 续行：默认加载骨干 ImageNet 预训练权重
    acronyms = {  # 数据集名 → 权重文件名中的缩写映射
        'pascal_voc': 'pascal_voc',  # Pascal VOC 2012
        'pascal_aug': 'pascal_aug',  # VOC 增广集（SBD）
        'ade20k': 'ade',  # ADE20K 场景解析数据集
        'coco': 'coco',  # COCO Stuff
        'citys': 'citys',  # Cityscapes 城市街景（19 类）
    }  # 缩写映射表结束
    from ..data.dataloader import datasets  # 延迟导入数据集注册表（避免模块级循环导入）
    model = BiSeNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建模型
    if pretrained:  # 若需加载本模型的分割预训练权重
        from .model_store import get_model_file  # 延迟导入权重文件路径解析工具
        device = torch.device(kwargs['local_rank'])  # 【注意】可疑写法：kwargs 未包含 local_rank 时会 KeyError；且以 int 直接构造 torch.device 的行为依赖 PyTorch 版本（更稳妥写法为 torch.device('cuda', rank)）
        model.load_state_dict(torch.load(get_model_file('bisenet_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 拼出权重文件名（如 bisenet_resnet18_citys.pth）并读取
                              map_location=device))  # 将权重映射到指定设备（多卡训练按 rank 加载）
    return model  # 返回构建/加载完成的模型


def get_bisenet_resnet18_citys(**kwargs):  # 便捷构造函数：ResNet18 骨干 + Cityscapes 数据集（19 类）
    return get_bisenet('citys', 'resnet18', **kwargs)  # 委托给 get_bisenet


if __name__ == '__main__':  # 直接运行本文件时的小型自测（不触发训练与权重下载）
    img = torch.randn(2, 3, 224, 224)  # 构造随机输入 (2, 3, 224, 224)
    model = BiSeNet(19, backbone='resnet18')  # 构建 19 类（Cityscapes）BiSeNet
    print(model.exclusive)  # 打印自有模块名单（【注意】img 与 model 均未被前向使用，仅作展示）
