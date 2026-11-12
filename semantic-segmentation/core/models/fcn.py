# ============================================================================
# 【文件说明】fcn.py —— 经典 FCN 系列（FCN32s / FCN16s / FCN8s）的 PyTorch 实现，
#              基于 VGG16 骨干网络，并附带各数据集（VOC/ADE20K/COCO/Cityscapes/SBU）
#              的通用工厂函数与三个 Pascal VOC 预设工厂函数。
#              工程位置：core/models/fcn.py，由 model_zoo.py 汇总注册，供脚本按名调用。
# 【核心思想】全卷积网络 FCN：将分类网络的全连接层替换为卷积层，输出稠密得分图实现像素级预测；
#            FCN32s 把 1/32 分辨率的 pool5 得分一步上采样 32 倍，边界粗糙；
#            FCN16s / FCN8s 引入跳跃连接（skip connection）：深层语义得分逐级 2 倍上采样后，
#            与浅层 pool4 / pool3 经 1x1 卷积得到的得分逐元素相加融合，兼顾语义与空间细节，
#            最后上采样回原图尺寸；上采样用双线性插值 F.interpolate（原论文为学习的转置卷积）。
# 【注意】与原论文差异：VGG 的 fc6/fc7 全连接被替换为 _FCNHead 卷积头（见 FCN32s docstring）。
# ============================================================================
import os  # 【注意】未使用的导入：本文件实际并未直接使用 os 模块
import torch  # PyTorch 主模块：用于 torch.device 指定设备与 torch.load 加载权重
import torch.nn as nn  # 神经网络模块：提供 nn.Module、nn.Sequential、Conv2d、BN 等层
import torch.nn.functional as F  # 函数式接口：用于 F.interpolate 双线性上采样

from .base_models.vgg import vgg16  # 导入本仓库实现的 VGG16 骨干网络构造函数

__all__ = ['get_fcn32s', 'get_fcn16s', 'get_fcn8s',  # 模块公开接口：三个通用工厂函数
           'get_fcn32s_vgg16_voc', 'get_fcn16s_vgg16_voc', 'get_fcn8s_vgg16_voc']  # 以及三个 VOC 预设工厂函数


class FCN32s(nn.Module):  # FCN32s 模型：pool5 得分一步上采样 32 倍的最简全卷积网络，继承 nn.Module
    """There are some difference from original fcn"""  # 原注释保留：与原始 FCN 论文实现存在差异（fc6/fc7 → _FCNHead）

    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数（含背景），决定输出通道数
    # backbone：骨干网络类型，当前仅支持 'vgg16'
    # aux：是否启用辅助损失分支（与主头共用 pool5 特征）
    # pretrained_base：是否加载 VGG16 的 ImageNet 预训练权重
    # norm_layer：归一化层类型，默认 nn.BatchNorm2d（可换同步 BN）
    # **kwargs：其余参数透传给 _FCNHead
    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base=True,  # 构造：VGG 骨干 + 主/辅分割头
                 norm_layer=nn.BatchNorm2d, **kwargs):  # norm_layer 参数单独续行，便于扩展（如 SyncBN）
        super(FCN32s, self).__init__()  # 调用 nn.Module 构造函数，完成模块注册
        self.aux = aux  # 保存辅助分支开关，供 forward 判断
        if backbone == 'vgg16':  # 骨干选择分支：目前唯一支持 vgg16
            self.pretrained = vgg16(pretrained=pretrained_base).features  # 取 VGG16 卷积特征段作骨干，输出 pool5 → (N, 512, H/32, W/32)
        else:  # 其他骨干名称
            raise RuntimeError('unknown backbone: {}'.format(backbone))  # 抛运行时错误：未知骨干类型
        self.head = _FCNHead(512, nclass, norm_layer)  # 主分割头：pool5 的 512 通道 → nclass 通道得分图
        if aux:  # 启用辅助分支时
            self.auxlayer = _FCNHead(512, nclass, norm_layer)  # 辅助分割头：结构同主头，同样接 pool5（仅训练期提供额外监督）

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])  # 登记"专属层"名单（新增可训练层），训练脚本据此单独设置学习率

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像张量 (N, 3, H, W)
    # 返回：元组 (主输出,) 或 (主输出, 辅助输出)，均为 (N, nclass, H, W)
    def forward(self, x):  # 前向传播：骨干 → 分割头 → 双线性上采样回原图尺寸
        size = x.size()[2:]  # 记录输入空间尺寸 (H, W)，供最终上采样对齐
        pool5 = self.pretrained(x)  # VGG16 特征提取（含 5 次下采样）→ (N, 512, H/32, W/32)

        outputs = []  # 输出收集列表：主输出在前、辅助输出在后
        out = self.head(pool5)  # 主头生成类别得分 → (N, nclass, H/32, W/32)
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)  # 32 倍双线性上采样 → (N, nclass, H, W)
        outputs.append(out)  # 主输出加入列表

        if self.aux:  # 若启用辅助分支
            auxout = self.auxlayer(pool5)  # 辅助头得分 → (N, nclass, H/32, W/32)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)  # 上采样回原尺寸 → (N, nclass, H, W)
            outputs.append(auxout)  # 辅助输出加入列表

        return tuple(outputs)  # 返回 (主输出,) 或 (主输出, 辅助输出) 元组


class FCN16s(nn.Module):  # FCN16s 模型：融合 pool4 跳跃连接、最终上采样 16 倍的全卷积网络
    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数；backbone：仅支持 'vgg16'；aux：辅助分支开关
    # pretrained_base：是否加载 ImageNet 预训练；norm_layer：归一化层类型；**kwargs：透传给 _FCNHead
    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base=True, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造：拆分 VGG 为 pool4/pool5 两段并建分割头
        super(FCN16s, self).__init__()  # 调用父类构造函数
        self.aux = aux  # 保存辅助分支开关
        if backbone == 'vgg16':  # 骨干选择分支
            self.pretrained = vgg16(pretrained=pretrained_base).features  # 完整 VGG16 特征段；【注意】与下方 pool4/pool5 共享同一组子模块，state_dict 中参数会以两套名字重复出现
        else:  # 未知骨干
            raise RuntimeError('unknown backbone, not vgg16: {}'.format(backbone))  # 抛运行时错误
        self.pool4 = nn.Sequential(*self.pretrained[:24])  # VGG 前 24 层（到 pool4 池化为止）→ 输出 (N, 512, H/16, W/16)
        self.pool5 = nn.Sequential(*self.pretrained[24:])  # VGG 第 24 层起（pool5 段）→ (N, 512, H/32, W/32)
        self.head = _FCNHead(512, nclass, norm_layer)  # 主分割头：pool5 的 512 通道 → nclass 通道
        self.score_pool4 = nn.Conv2d(512, nclass, 1)  # pool4 跳跃分支得分卷积：1x1，512 → nclass，通道对齐以便相加
        if aux:  # 启用辅助分支时
            self.auxlayer = _FCNHead(512, nclass, norm_layer)  # 辅助分割头：接在 pool5 特征上

        self.__setattr__('exclusive', ['head', 'score_pool4', 'auxlayer'] if aux else ['head', 'score_pool4'])  # 登记专属层名单：主头、pool4 评分卷积（及辅助头）

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像张量 (N, 3, H, W)；返回 (主输出,) 或 (主输出, 辅助输出)
    def forward(self, x):  # 前向：分段提特征 → 深层得分 2 倍上采样与 pool4 得分融合 → 上采样回原图
        pool4 = self.pool4(x)  # 浅层特征 → (N, 512, H/16, W/16)
        pool5 = self.pool5(pool4)  # 深层特征 → (N, 512, H/32, W/32)

        outputs = []  # 输出收集列表
        score_fr = self.head(pool5)  # 主头得分 → (N, nclass, H/32, W/32)

        score_pool4 = self.score_pool4(pool4)  # pool4 跳跃分支得分 → (N, nclass, H/16, W/16)

        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode='bilinear', align_corners=True)  # 深层得分 2 倍上采样 → (N, nclass, H/16, W/16)
        fuse_pool4 = upscore2 + score_pool4  # 跳跃连接：逐元素相加融合语义与细节 → (N, nclass, H/16, W/16)

        out = F.interpolate(fuse_pool4, x.size()[2:], mode='bilinear', align_corners=True)  # 融合结果 16 倍上采样 → (N, nclass, H, W)
        outputs.append(out)  # 主输出加入列表

        if self.aux:  # 若启用辅助分支
            auxout = self.auxlayer(pool5)  # 辅助头得分 → (N, nclass, H/32, W/32)
            auxout = F.interpolate(auxout, x.size()[2:], mode='bilinear', align_corners=True)  # 上采样回原尺寸 → (N, nclass, H, W)
            outputs.append(auxout)  # 辅助输出加入列表

        return tuple(outputs)  # 返回输出元组


class FCN8s(nn.Module):  # FCN8s 模型：融合 pool3/pool4 两级跳跃连接、最终上采样 8 倍的全卷积网络
    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数；backbone：仅支持 'vgg16'；aux：辅助分支开关
    # pretrained_base：是否加载 ImageNet 预训练；norm_layer：归一化层类型；**kwargs：透传给 _FCNHead
    def __init__(self, nclass, backbone='vgg16', aux=False, pretrained_base=True, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造：拆分 VGG 为 pool3/pool4/pool5 三段并建分割头
        super(FCN8s, self).__init__()  # 调用父类构造函数
        self.aux = aux  # 保存辅助分支开关
        if backbone == 'vgg16':  # 骨干选择分支
            self.pretrained = vgg16(pretrained=pretrained_base).features  # 完整 VGG16 特征段；【注意】与 pool3/pool4/pool5 共享同一组子模块，state_dict 中参数会重复出现
        else:  # 未知骨干
            raise RuntimeError('unknown backbone: {}'.format(backbone))  # 抛运行时错误
        self.pool3 = nn.Sequential(*self.pretrained[:17])  # VGG 前 17 层（到 pool3 池化为止）→ (N, 256, H/8, W/8)
        self.pool4 = nn.Sequential(*self.pretrained[17:24])  # VGG 第 17~23 层（pool4 段）→ (N, 512, H/16, W/16)
        self.pool5 = nn.Sequential(*self.pretrained[24:])  # VGG 第 24 层起（pool5 段）→ (N, 512, H/32, W/32)
        self.head = _FCNHead(512, nclass, norm_layer)  # 主分割头：pool5 的 512 通道 → nclass 通道
        self.score_pool3 = nn.Conv2d(256, nclass, 1)  # pool3 跳跃分支得分卷积：1x1，256 → nclass（pool3 输出为 256 通道）
        self.score_pool4 = nn.Conv2d(512, nclass, 1)  # pool4 跳跃分支得分卷积：1x1，512 → nclass
        if aux:  # 启用辅助分支时
            self.auxlayer = _FCNHead(512, nclass, norm_layer)  # 辅助分割头：接在 pool5 特征上

        self.__setattr__('exclusive',  # 登记专属层名单（列表较长，续行书写）
                         ['head', 'score_pool3', 'score_pool4', 'auxlayer'] if aux else ['head', 'score_pool3',  # aux 时包含辅助头；否则仅主头与 pool3 评分卷积
                                                                                         'score_pool4'])  # 以及 pool4 评分卷积

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像张量 (N, 3, H, W)；返回 (主输出,) 或 (主输出, 辅助输出)
    def forward(self, x):  # 前向：三段提特征 → 两级跳跃融合 → 上采样回原图
        pool3 = self.pool3(x)  # 浅层特征 → (N, 256, H/8, W/8)
        pool4 = self.pool4(pool3)  # 中层特征 → (N, 512, H/16, W/16)
        pool5 = self.pool5(pool4)  # 深层特征 → (N, 512, H/32, W/32)

        outputs = []  # 输出收集列表
        score_fr = self.head(pool5)  # 主头得分 → (N, nclass, H/32, W/32)

        score_pool4 = self.score_pool4(pool4)  # pool4 跳跃分支得分 → (N, nclass, H/16, W/16)
        score_pool3 = self.score_pool3(pool3)  # pool3 跳跃分支得分 → (N, nclass, H/8, W/8)

        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode='bilinear', align_corners=True)  # 深层得分 2 倍上采样 → (N, nclass, H/16, W/16)
        fuse_pool4 = upscore2 + score_pool4  # 第一级跳跃融合 → (N, nclass, H/16, W/16)

        upscore_pool4 = F.interpolate(fuse_pool4, score_pool3.size()[2:], mode='bilinear', align_corners=True)  # 融合得分再 2 倍上采样 → (N, nclass, H/8, W/8)
        fuse_pool3 = upscore_pool4 + score_pool3  # 第二级跳跃融合 → (N, nclass, H/8, W/8)

        out = F.interpolate(fuse_pool3, x.size()[2:], mode='bilinear', align_corners=True)  # 8 倍上采样回原尺寸 → (N, nclass, H, W)
        outputs.append(out)  # 主输出加入列表

        if self.aux:  # 若启用辅助分支
            auxout = self.auxlayer(pool5)  # 辅助头得分 → (N, nclass, H/32, W/32)
            auxout = F.interpolate(auxout, x.size()[2:], mode='bilinear', align_corners=True)  # 上采样回原尺寸 → (N, nclass, H, W)
            outputs.append(auxout)  # 辅助输出加入列表

        return tuple(outputs)  # 返回输出元组


class _FCNHead(nn.Module):  # FCN 分割头：替代原论文 VGG 的 fc6/fc7 全连接（改为卷积）+ score 层
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：输入通道数（VGG pool5 为 512）
    # channels：输出通道数（= nclass）
    # norm_layer：归一化层类型，默认 nn.BatchNorm2d
    # **kwargs：兼容上层透传的其他参数（此处不使用）
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造：3x3 卷积降维 + BN + ReLU + Dropout + 1x1 投影
        super(_FCNHead, self).__init__()  # 调用父类构造函数
        inter_channels = in_channels // 4  # 瓶颈中间通道数 = 输入/4（512→128），起到降维减参作用
        self.block = nn.Sequential(  # 顺序组合的分割头块
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),  # 3x3 卷积 padding=1 保持分辨率，in_channels→inter_channels，无偏置（后接 BN）
            norm_layer(inter_channels),  # 归一化层（作用在 inter_channels 通道上）
            nn.ReLU(inplace=True),  # 就地 ReLU 激活，节省内存
            nn.Dropout(0.1),  # Dropout p=0.1，防止过拟合（分割头常用 0.1）
            nn.Conv2d(inter_channels, channels, 1)  # 1x1 卷积投影到类别数 → 输出 (N, nclass, H', W') 得分图
        )  # 分割头 Sequential 块结束

    # ----------------------- forward 参数说明 -----------------------
    # x：输入特征 (N, in_channels, H', W')；输出 (N, channels, H', W')
    def forward(self, x):  # 前向：直接过 block，通道 in_channels→channels，空间分辨率不变
        return self.block(x)  # 返回分割得分图 (N, channels, H', W')


# ----------------------- get_fcn32s 参数说明 -----------------------
# dataset：数据集名称（决定类别数），如 'pascal_voc' / 'ade20k' 等
# backbone：骨干类型，默认 'vgg16'；pretrained：是否加载本仓库训练好的分割权重
# root：权重缓存目录，默认 '~/.torch/models'；pretrained_base：是否加载骨干 ImageNet 预训练
# **kwargs：透传给 FCN32s（如 aux、norm_layer、local_rank 等）
def get_fcn32s(dataset='pascal_voc', backbone='vgg16', pretrained=False, root='~/.torch/models',  # FCN32s 通用工厂函数
               pretrained_base=True, **kwargs):  # 续行：默认加载骨干 ImageNet 预训练权重
    acronyms = {  # 数据集名 → 权重文件命名缩写映射表
        'pascal_voc': 'pascal_voc',  # Pascal VOC（原生标注）
        'pascal_aug': 'pascal_aug',  # Pascal VOC 增强标注（SBD）
        'ade20k': 'ade',  # ADE20K 缩写为 ade
        'coco': 'coco',  # COCO
        'citys': 'citys',  # Cityscapes
        'sbu': 'sbu',  # SBU
    }  # 数据集缩写映射表结束
    from ..data.dataloader import datasets  # 局部导入数据集注册表（避免模块加载期的循环依赖）
    model = FCN32s(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数实例化 FCN32s
    if pretrained:  # 需要加载分割预训练权重时
        from .model_store import get_model_file  # 局部导入权重文件定位工具
        device = torch.device(kwargs['local_rank'])  # 【注意】可疑写法：直接取 kwargs['local_rank']，未传该键会 KeyError；int 会被当作 CUDA 设备序号，纯 CPU 环境会出错
        model.load_state_dict(torch.load(get_model_file('fcn32s_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 定位权重文件（文件名如 fcn32s_vgg16_pascal_voc.pth）并加载
                              map_location=device))  # 权重张量映射到指定设备后载入模型
    return model  # 返回构建（并可选加载权重）的模型


# ----------------------- get_fcn16s 参数说明 -----------------------
# 参数含义同 get_fcn32s：dataset/backbone/pretrained/root/pretrained_base/**kwargs
def get_fcn16s(dataset='pascal_voc', backbone='vgg16', pretrained=False, root='~/.torch/models',  # FCN16s 通用工厂函数
               pretrained_base=True, **kwargs):  # 续行：默认加载骨干 ImageNet 预训练权重
    acronyms = {  # 数据集名 → 权重文件命名缩写映射表
        'pascal_voc': 'pascal_voc',  # Pascal VOC（原生标注）
        'pascal_aug': 'pascal_aug',  # Pascal VOC 增强标注（SBD）
        'ade20k': 'ade',  # ADE20K 缩写为 ade
        'coco': 'coco',  # COCO
        'citys': 'citys',  # Cityscapes
        'sbu': 'sbu',  # SBU
    }  # 数据集缩写映射表结束
    from ..data.dataloader import datasets  # 局部导入数据集注册表
    model = FCN16s(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数实例化 FCN16s
    if pretrained:  # 需要加载分割预训练权重时
        from .model_store import get_model_file  # 局部导入权重文件定位工具
        device = torch.device(kwargs['local_rank'])  # 【注意】可疑写法：同 get_fcn32s，依赖 kwargs 中的 local_rank
        model.load_state_dict(torch.load(get_model_file('fcn16s_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 定位权重文件（如 fcn16s_vgg16_pascal_voc.pth）并加载
                              map_location=device))  # 映射到指定设备
    return model  # 返回模型


# ----------------------- get_fcn8s 参数说明 -----------------------
# 参数含义同 get_fcn32s：dataset/backbone/pretrained/root/pretrained_base/**kwargs
def get_fcn8s(dataset='pascal_voc', backbone='vgg16', pretrained=False, root='~/.torch/models',  # FCN8s 通用工厂函数
              pretrained_base=True, **kwargs):  # 续行：默认加载骨干 ImageNet 预训练权重
    acronyms = {  # 数据集名 → 权重文件命名缩写映射表
        'pascal_voc': 'pascal_voc',  # Pascal VOC（原生标注）
        'pascal_aug': 'pascal_aug',  # Pascal VOC 增强标注（SBD）
        'ade20k': 'ade',  # ADE20K 缩写为 ade
        'coco': 'coco',  # COCO
        'citys': 'citys',  # Cityscapes
        'sbu': 'sbu',  # SBU
    }  # 数据集缩写映射表结束
    from ..data.dataloader import datasets  # 局部导入数据集注册表
    model = FCN8s(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数实例化 FCN8s
    if pretrained:  # 需要加载分割预训练权重时
        from .model_store import get_model_file  # 局部导入权重文件定位工具
        device = torch.device(kwargs['local_rank'])  # 【注意】可疑写法：同 get_fcn32s，依赖 kwargs 中的 local_rank
        model.load_state_dict(torch.load(get_model_file('fcn8s_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 定位权重文件（如 fcn8s_vgg16_pascal_voc.pth）并加载
                              map_location=device))  # 映射到指定设备
    return model  # 返回模型


# ----------------------- 函数参数说明 -----------------------
# **kwargs：透传给 get_fcn32s（可含 pretrained、local_rank 等）
def get_fcn32s_vgg16_voc(**kwargs):  # VOC 预设工厂：FCN32s + VGG16 + Pascal VOC
    return get_fcn32s('pascal_voc', 'vgg16', **kwargs)  # 固定数据集与骨干，调用通用工厂


# ----------------------- 函数参数说明 -----------------------
# **kwargs：透传给 get_fcn16s
def get_fcn16s_vgg16_voc(**kwargs):  # VOC 预设工厂：FCN16s + VGG16 + Pascal VOC
    return get_fcn16s('pascal_voc', 'vgg16', **kwargs)  # 固定数据集与骨干，调用通用工厂


# ----------------------- 函数参数说明 -----------------------
# **kwargs：透传给 get_fcn8s
def get_fcn8s_vgg16_voc(**kwargs):  # VOC 预设工厂：FCN8s + VGG16 + Pascal VOC
    return get_fcn8s('pascal_voc', 'vgg16', **kwargs)  # 固定数据集与骨干，调用通用工厂


if __name__ == '__main__':  # 直接运行本文件时的简易自测入口（python fcn.py）
    model = FCN16s(21)  # 以 Pascal VOC 的 21 类构建 FCN16s 模型
    print(model)  # 打印模型结构（nn.Module 默认 repr）
