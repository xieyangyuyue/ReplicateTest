# ============================================================================
# 【文件说明】danet.py —— 双注意力网络 DANet（Dual Attention Network）分割模型：
#               基于 SegBaseModel 骨干（ResNet 系列、膨胀卷积、输出 1/8 分辨率特征），
#               分割头 _DAHead 并行使用"位置注意力 PAM + 通道注意力 CAM"双分支。
# 【核心思想】1) 位置注意力（PAM）：用三个 1×1 卷积从输入特征生成
#                  B（降到 C/8，充当 Query）、C（降到 C/8，充当 Key）、
#                  D（保持 C 通道，充当 Value）；特征展平为 (N, C', H*W) 后
#                  bmm 得到 (N, H*W, H*W) 像素间相似度矩阵，在最后一维 softmax，
#                  再把注意力权重加权回 Value，为每个位置聚合远程上下文，
#                  经可学习系数 α 残差融合回原特征；
#               2) 通道注意力（CAM）：不引入卷积，Q/K/V 均为特征自身：
#                  (N, C, H*W) × (N, H*W, C) → (N, C, C) 通道相似度矩阵，
#                  做"行最大值减自身"的反向处理后 softmax，再加权聚合各通道，
#                  经可学习系数 β 残差融合回原特征；
#               3) 双分支输出逐元素相加（sum 融合）后经 Dropout + 1×1 卷积得到
#                  主 logits；两个分支各带一个独立辅助分类头（aux loss）。
# 【论文】Jun Fu, Jing Liu, et al. "Dual Attention Network for Scene Segmentation."
#         CVPR 2019
# ============================================================================
"""Dual Attention Network"""  # 模块级英文 docstring（原样保留）
import torch  # PyTorch 主库：张量运算、nn.Parameter、torch.load 等
import torch.nn as nn  # 神经网络模块：Conv2d / BatchNorm2d / Sequential / Softmax 等
import torch.nn.functional as F  # 函数式接口：interpolate 等无状态操作

from .segbase import SegBaseModel  # 分割模型基类：统一封装骨干网络构建与 base_forward

__all__ = ['DANet', 'get_danet', 'get_danet_resnet50_citys',  # 模块对外导出的公开符号列表（模型类 + 便捷构造函数）
           'get_danet_resnet101_citys', 'get_danet_resnet152_citys']  # 其余三个便捷构造函数名


# 【注意】DANet 类 docstring 的标题写的是 "Pyramid Scene Parsing Network"（PSPNet），
# 系从 PSPNet 代码复制后未修改的笔误，本类实为双注意力网络 DANet
class DANet(SegBaseModel):  # DANet 主模型：继承 SegBaseModel（内部构建膨胀 ResNet 骨干）
    # 类 docstring：DANet 模型说明与参数（【注意】标题沿袭原代码自 PSPNet 复制的笔误，原英文保留）
    r"""Pyramid Scene Parsing Network

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.
    Reference:
        Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,and Hanqing Lu.
        "Dual Attention Network for Scene Segmentation." *CVPR*, 2019
    """  # DANet 类英文 docstring 结束（原样保留）
    # 【注意】docstring 中 norm_layer 默认值写作 mxnet.gluon.nn.BatchNorm，为 MXNet Gluon 版
    # 移植遗留的描述（废弃 API 痕迹），本仓库实际使用 PyTorch 的 nn.BatchNorm2d
    # ---- __init__ 参数说明 ----
    # nclass: 类别数（决定输出通道数）
    # backbone: 骨干网络类型，默认 'resnet50'（另支持 resnet101 / resnet152）
    # aux: 是否启用辅助损失（PAM / CAM 分支各带一个辅助分类头）
    # pretrained_base: 是否加载骨干网络的 ImageNet 预训练权重
    # **kwargs: 其余参数透传给基类与头部（如 norm_layer、norm_kwargs）
    def __init__(self, nclass, backbone='resnet50', aux=True, pretrained_base=True, **kwargs):  # 构造函数
        super(DANet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)  # 调用基类构造：构建骨干网络、设置 aux 等属性
        self.head = _DAHead(2048, nclass, aux, **kwargs)  # 双注意力分割头；2048 为 ResNet50/101/152 stage4 输出通道数

        self.__setattr__('exclusive', ['head'])  # 把 head 登记为"独占"参数组，训练时可对其单独设置学习率/权重衰减

    # ---- forward 参数说明 ----
    # x: 输入图像张量 (N, 3, H, W)
    def forward(self, x):  # 前向传播：骨干提特征 → 双注意力头 → 上采样回原尺寸
        size = x.size()[2:]  # 记录输入空间尺寸 (H, W)，供最终上采样对齐
        # 骨干前向：返回 (c1, c2, c3, c4)；此处只用 c4，c3 解包后未使用（DANet 的辅助头基于 c4 分支特征）
        _, _, c3, c4 = self.base_forward(x)  # c4 形状 (N, 2048, H/8, W/8)
        outputs = []  # 输出收集列表：主输出 + 可选辅助输出
        x = self.head(c4)  # 双注意力头部处理 c4，返回 (主 logits, PAM 辅助 logits, CAM 辅助 logits)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)  # 主 logits 双线性上采样 → (N, nclass, H, W)
        outputs.append(x0)  # 主输出加入列表

        if self.aux:  # 若启用辅助损失
            x1 = F.interpolate(x[1], size, mode='bilinear', align_corners=True)  # PAM 分支辅助 logits 上采样 → (N, nclass, H, W)
            x2 = F.interpolate(x[2], size, mode='bilinear', align_corners=True)  # CAM 分支辅助 logits 上采样 → (N, nclass, H, W)
            outputs.append(x1)  # PAM 辅助输出加入列表
            outputs.append(x2)  # CAM 辅助输出加入列表
        return outputs  # 返回 [主输出, (可选)PAM 辅助输出, (可选)CAM 辅助输出]


# 位置注意力模块（Position Attention Module, PAM）
# 职责：建模任意两个空间位置之间的依赖关系，为每个位置聚合远程上下文（non-local 思想）
class _PositionAttentionModule(nn.Module):  # 位置注意力模块：继承 nn.Module
    """ Position attention module"""  # 原英文 docstring（保留）

    # ---- __init__ 参数说明 ----
    # in_channels: 输入通道数 C（DANet 中为 inter_channels=512）
    # **kwargs: 预留扩展参数（本模块未使用）
    def __init__(self, in_channels, **kwargs):  # 构造函数
        super(_PositionAttentionModule, self).__init__()  # 调用父类构造
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)  # Query 生成器：1×1 卷积降维 C→C/8（512→64），得到分支 B
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)  # Key 生成器：1×1 卷积降维 C→C/8，得到分支 C
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)  # Value 生成器：1×1 卷积保持 C 通道，得到分支 D
        self.alpha = nn.Parameter(torch.zeros(1))  # 残差融合系数 α，零初始化：训练初期注意力分支不影响原特征
        self.softmax = nn.Softmax(dim=-1)  # softmax 层：在最后一维（Key 位置维）归一化

    # ---- forward 参数说明 ----
    # x: 输入特征 (N, C, H, W)
    def forward(self, x):  # 前向：生成 Q/K/V → 相似度矩阵 → softmax → 加权聚合 → 残差融合
        batch_size, _, height, width = x.size()  # 解包批大小 N 与空间尺寸 H、W
        # Query：卷积后展平再转置，(N, C/8, H, W) → (N, C/8, H*W) → (N, H*W, C/8)
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # 分支 B（Query），(N, H*W, C/8)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)  # 分支 C（Key）：(N, C/8, H, W) → (N, C/8, H*W)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))  # 相似度矩阵：(N, H*W, C/8) × (N, C/8, H*W) → (N, H*W, H*W)，再在最后一维 softmax（每个查询位置对所有键位置的权重）
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)  # 分支 D（Value）：(N, C, H, W) → (N, C, H*W)
        # 注意力加权：(N, C, H*W) × (N, H*W, H*W) → (N, C, H*W)，随后 reshape 回四维
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)  # 加权聚合 Value 并还原空间维度 → (N, C, H, W)，以便与 x 残差相加
        out = self.alpha * feat_e + x  # 残差融合：α·注意力输出 + 原特征（α 初始为 0，由训练学习）

        return out  # 返回位置注意力增强后的特征（形状应与输入 x 一致）


# 通道注意力模块（Channel Attention Module, CAM）
# 职责：建模通道间依赖，用全局通道关系重标定每个通道的特征（Q/K/V 均为特征自身）
class _ChannelAttentionModule(nn.Module):  # 通道注意力模块：继承 nn.Module
    """Channel attention module"""  # 原英文 docstring（保留）

    # ---- __init__ 参数说明 ----
    # **kwargs: 预留扩展参数（本模块未使用）
    def __init__(self, **kwargs):  # 构造函数：无可学习卷积，仅融合系数 β 与 softmax
        super(_ChannelAttentionModule, self).__init__()  # 调用父类构造
        self.beta = nn.Parameter(torch.zeros(1))  # 残差融合系数 β，零初始化
        self.softmax = nn.Softmax(dim=-1)  # softmax 层：在最后一维（通道维）归一化

    # ---- forward 参数说明 ----
    # x: 输入特征 (N, C, H, W)
    def forward(self, x):  # 前向：特征自转置相乘 → 反向处理 → softmax → 加权聚合 → 残差融合
        batch_size, _, height, width = x.size()  # 解包 (N, C, H, W)
        feat_a = x.view(batch_size, -1, height * width)  # 特征展平：(N, C, H, W) → (N, C, H*W)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)  # 展平后转置 → (N, H*W, C)
        attention = torch.bmm(feat_a, feat_a_transpose)  # 通道相似度矩阵：(N, C, H*W) × (N, H*W, C) → (N, C, C)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention  # 反向注意力：逐行最大值广播后减自身，(N, C, C)
        attention = self.softmax(attention_new)  # 对反向矩阵在最后一维（通道维）softmax

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)  # 加权聚合：(N, C, C) × (N, C, H*W) → (N, C, H*W)，再还原四维 (N, C, H, W)
        out = self.beta * feat_e + x  # 残差融合：β·注意力输出 + 原特征

        return out  # 返回通道注意力增强后的特征 (N, C, H, W)


# 双注意力头部（Dual Attention Head）
# 职责：骨干特征 2048→512 降维后并行过 PAM / CAM 两分支，各自再卷积一次，
#       逐元素相加融合输出主 logits，并带两个分支独立的辅助分类头
class _DAHead(nn.Module):  # DANet 的分割头：继承 nn.Module
    # ---- __init__ 参数说明 ----
    # in_channels: 输入通道数（ResNet stage4 输出 2048）
    # nclass: 类别数
    # aux: 是否构造两个辅助分类头
    # norm_layer: 归一化层类型，默认 nn.BatchNorm2d（可替换为 SyncBN 等）
    # norm_kwargs: 归一化层额外参数（None 时按空字典展开）
    # **kwargs: 透传给 PAM / CAM 的预留参数
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):  # 构造函数
        super(_DAHead, self).__init__()  # 调用父类构造
        self.aux = aux  # 保存辅助损失开关
        inter_channels = in_channels // 4  # 中间通道数：2048 // 4 = 512，双分支的公共工作通道数
        # 位置分支前置卷积块 conv_p1：3×3 卷积降维（2048→512）+ BN + ReLU
        self.conv_p1 = nn.Sequential(  # 顺序容器
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),  # 3×3 卷积，padding=1 保持分辨率，无 bias（后接 BN）
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),  # 归一化层；norm_kwargs 为 None 时展开为空字典
            nn.ReLU(True)  # 就地 ReLU 激活
        )  # conv_p1 定义结束
        # 通道分支前置卷积块 conv_c1：结构与 conv_p1 相同，参数独立
        self.conv_c1 = nn.Sequential(  # 顺序容器
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),  # 3×3 卷积 2048→512
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),  # 归一化层
            nn.ReLU(True)  # 就地 ReLU 激活
        )  # conv_c1 定义结束
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)  # 位置注意力模块（512 通道输入，内部 Q/K 降到 64）
        self.cam = _ChannelAttentionModule(**kwargs)  # 通道注意力模块（除融合系数 β 外无可学习参数）
        # 位置分支后置卷积块 conv_p2：3×3 卷积（512→512）+ BN + ReLU
        self.conv_p2 = nn.Sequential(  # 顺序容器
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),  # 3×3 卷积保持通道数
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),  # 归一化层
            nn.ReLU(True)  # 就地 ReLU 激活
        )  # conv_p2 定义结束
        # 通道分支后置卷积块 conv_c2：结构与 conv_p2 相同
        self.conv_c2 = nn.Sequential(  # 顺序容器
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),  # 3×3 卷积保持通道数
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),  # 归一化层
            nn.ReLU(True)  # 就地 ReLU 激活
        )  # conv_c2 定义结束
        # 融合特征的主分类头：Dropout + 1×1 卷积
        self.out = nn.Sequential(  # 顺序容器
            nn.Dropout(0.1),  # Dropout，丢弃概率 0.1
            nn.Conv2d(inter_channels, nclass, 1)  # 1×1 卷积分类器：512 → nclass
        )  # out 定义结束
        if aux:  # 若启用辅助损失：为两个分支各建一个独立分类头
            # PAM 分支辅助分类头 conv_p3
            self.conv_p3 = nn.Sequential(  # 顺序容器
                nn.Dropout(0.1),  # Dropout 0.1
                nn.Conv2d(inter_channels, nclass, 1)  # 1×1 卷积：512 → nclass
            )  # conv_p3 定义结束
            # CAM 分支辅助分类头 conv_c3
            self.conv_c3 = nn.Sequential(  # 顺序容器
                nn.Dropout(0.1),  # Dropout 0.1
                nn.Conv2d(inter_channels, nclass, 1)  # 1×1 卷积：512 → nclass
            )  # conv_c3 定义结束

    # ---- forward 参数说明 ----
    # x: 骨干输出特征 (N, 2048, H, W)
    def forward(self, x):  # 前向：双分支并行 → 逐元素相加融合 → 输出（含辅助）
        feat_p = self.conv_p1(x)  # 位置分支降维卷积：(N, 2048, H, W) → (N, 512, H, W)
        feat_p = self.pam(feat_p)  # 位置注意力增强，通道与空间形状不变
        feat_p = self.conv_p2(feat_p)  # 位置分支后置卷积：(N, 512, H, W)

        feat_c = self.conv_c1(x)  # 通道分支降维卷积：(N, 2048, H, W) → (N, 512, H, W)
        feat_c = self.cam(feat_c)  # 通道注意力增强，形状不变
        feat_c = self.conv_c2(feat_c)  # 通道分支后置卷积：(N, 512, H, W)

        feat_fusion = feat_p + feat_c  # 双分支逐元素相加融合（论文的 sum 融合策略），(N, 512, H, W)

        outputs = []  # 输出收集列表
        fusion_out = self.out(feat_fusion)  # 融合特征 → 主分类 logits，(N, nclass, H, W)
        outputs.append(fusion_out)  # 主输出入列表
        if self.aux:  # 辅助输出
            p_out = self.conv_p3(feat_p)  # PAM 分支辅助 logits，(N, nclass, H, W)
            c_out = self.conv_c3(feat_c)  # CAM 分支辅助 logits，(N, nclass, H, W)
            outputs.append(p_out)  # PAM 辅助输出入列表
            outputs.append(c_out)  # CAM 辅助输出入列表

        return tuple(outputs)  # 返回 (主输出, PAM 辅助, CAM 辅助) 或仅 (主输出,)


# 便捷构造函数：按数据集 / 骨干 / 预训练选项构建 DANet
# ---- 参数说明 ----
# dataset: 数据集名（决定类别数），默认 'citys'
# backbone: 骨干类型，默认 'resnet50'
# pretrained: 是否加载完整模型预训练权重（True 或权重版本号字符串）
# root: 预训练权重文件存放根目录
# pretrained_base: 是否加载骨干的 ImageNet 预训练权重
# **kwargs: 透传给 DANet 构造函数
# 【注意】docstring 中写 "dataset : str, default pascal_voc"，与函数签名默认值 'citys' 不一致
def get_danet(dataset='citys', backbone='resnet50', pretrained=False,  # 构造入口：数据集 / 骨干 / 完整权重开关
              root='~/.torch/models', pretrained_base=True, **kwargs):  # 续行：权重目录、骨干预训练开关、透传参数
    # 函数 docstring：get_danet 参数说明与用法示例（原英文保留）
    r"""Dual Attention Network

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    pretrained_base : bool or str, default True
        This will load pretrained backbone network, that was trained on ImageNet.
    Examples
    --------
    >>> model = get_danet(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """  # get_danet 英文 docstring 结束（原样保留）
    acronyms = {  # 数据集名 → 预训练权重文件名中的缩写映射
        'pascal_voc': 'pascal_voc',  # Pascal VOC
        'pascal_aug': 'pascal_aug',  # Pascal VOC 增强版
        'ade20k': 'ade',  # ADE20K → 缩写 ade
        'coco': 'coco',  # COCO
        'citys': 'citys',  # Cityscapes
    }  # 缩写映射结束
    from ..data.dataloader import datasets  # 局部导入数据集注册表（避免循环导入），用于查询类别数
    model = DANet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建 DANet
    if pretrained:  # 若需加载完整模型预训练权重
        from .model_store import get_model_file  # 局部导入权重文件路径解析工具
        device = torch.device(kwargs['local_rank'])  # 【注意】直接取 kwargs['local_rank']，未传该键时会 KeyError；用 int 构造 device 的写法也较脆弱
        model.load_state_dict(torch.load(get_model_file('danet_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 拼出权重文件名 'danet_<backbone>_<数据集缩写>' 并加载（【注意】torch.load 未指定 weights_only，新版 PyTorch 会告警）
                              map_location=device))  # 将权重张量映射到指定设备
    return model  # 返回构建好的模型


def get_danet_resnet50_citys(**kwargs):  # 便捷构造：Cityscapes + ResNet50 版 DANet
    return get_danet('citys', 'resnet50', **kwargs)  # 委托 get_danet


def get_danet_resnet101_citys(**kwargs):  # 便捷构造：Cityscapes + ResNet101 版 DANet
    return get_danet('citys', 'resnet101', **kwargs)  # 委托 get_danet


def get_danet_resnet152_citys(**kwargs):  # 便捷构造：Cityscapes + ResNet152 版 DANet
    return get_danet('citys', 'resnet152', **kwargs)  # 委托 get_danet


if __name__ == '__main__':  # 直接运行本文件时的简易自测入口
    img = torch.randn(2, 3, 480, 480)  # 构造随机输入 (2, 3, 480, 480)
    model = get_danet_resnet50_citys()  # 构建 Cityscapes + ResNet50 版 DANet
    outputs = model(img)  # 前向推理，验证输出形状
