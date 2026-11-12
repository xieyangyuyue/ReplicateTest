# ============================================================================
# 【文件说明】espnet.py —— ESPNetv2（ESPNet 第二代）语义分割模型的 PyTorch 实现。
#              以 EESPNet 轻量编码器（高效空间金字塔 EESP 模块）提特征，
#              解码端自顶向下：最深层 1/16 特征压缩上采样后与 1/8 特征拼接，
#              经 EESP + 变体 PSP 模块聚合上下文并投影到类别数，
#              再逐级与 1/4、1/2 浅层特征拼接投影，最终上采样到原图尺寸。
# 【核心思想】论文《ESPNetv2: A Light-weight, Power Efficient, and General Purpose
#            Convolutional Neural Network》(Sachin Mehta et al., arXiv:1811.11431)。
#            核心 EESP 模块：1x1 分组卷积把通道分成 k 份，各份分别用不同空洞率的
#            3x3 深度卷积处理并逐级累加，再 1x1 分组卷积融合——以极少参数获得
#            大感受野（空间金字塔）；解码端反复用“拼接 + 投影”实现逐级求精。
# ============================================================================
"ESPNetv2: A Light-weight, Power Efficient, and General Purpose for Semantic Segmentation"  # 原文件首行字符串保留：ESPNetv2 论文标题（即模块 docstring）
import torch  # PyTorch 主库：张量运算与权重加载
import torch.nn as nn  # 神经网络模块：nn.Module、卷积/BN/PReLU/池化等网络层
import torch.nn.functional as F  # 函数式接口：双线性上采样 F.interpolate、均值池化 F.avg_pool2d 等

from core.models.base_models import eespnet, EESP  # 导入 EESPNet 编码器构造函数及 EESP 模块（core/models/base_models/eespnet.py）
from core.nn import _ConvBNPReLU, _BNPReLU  # 导入基础模块：“卷积+BN+PReLU”与“BN+PReLU”（core/nn/basic.py）


class ESPNetV2(nn.Module):  # ESPNetv2 分割网络：EESPNet 编码器 + 自顶向下逐级拼接融合解码
    # 类 docstring：ESPNetv2 模型说明与参数（原英文保留）
    r"""ESPNetV2

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Sachin Mehta, et al. "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network."
        arXiv preprint arXiv:1811.11431 (2018).
    """  # 上方英文 docstring 原样保留（【注意】其中 backbone 参数说明系模板复制，本模型实际不使用外部 ResNet 骨干）

    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数（输出通道数）
    # backbone / jpu：接口占位参数，本模型未使用【注意】
    # aux：是否输出辅助分支（取 1/8 分辨率的中间分割图）
    # pretrained_base：是否加载编码器预训练权重
    # **kwargs：透传给 eespnet 与各子模块（如 norm_layer）
    def __init__(self, nclass, backbone='', aux=False, jpu=False, pretrained_base=False, **kwargs):  # 构造函数
        super(ESPNetV2, self).__init__()  # 调用父类构造函数
        self.pretrained = eespnet(pretrained=pretrained_base, **kwargs)  # EESPNet 编码器；【注意】eespnet 内部对 pretrained=True 会直接 raise ValueError("Don't support pretrained")，故本模型实际无法加载骨干预训练权重
        self.proj_L4_C = _ConvBNPReLU(256, 128, 1, **kwargs)  # 1x1 卷积 256→128：压缩最深层（1/16 分辨率）特征通道
        self.pspMod = nn.Sequential(  # PSP 变体组合模块：先 EESP 提取大感受野特征，再 _PSPModule 聚合多尺度上下文
            EESP(256, 128, stride=1, k=4, r_lim=7, **kwargs),  # EESP：k=4 份金字塔分支、卷积核上限 r_lim=7，256→128，分辨率不变
            _PSPModule(128, 128, **kwargs))  # PSP 聚合：128→128，四级渐进下采样上下文拼接
        self.project_l3 = nn.Sequential(  # 1/8 分辨率处的分割投影（不含激活，便于辅助分支取激活前的 logits）
            nn.Dropout2d(0.1),  # 2D Dropout p=0.1；【注意】nn.Dropout2d 在 PyTorch≥1.11 已弃用（官方建议改用 nn.Dropout）
            nn.Conv2d(128, nclass, 1, bias=False))  # 1x1 卷积 128→nclass，无偏置
        self.act_l3 = _BNPReLU(nclass, **kwargs)  # BN+PReLU：【注意】作用在 nclass 维“得分图”上（对类别通道做归一化激活，较少见的写法）
        self.project_l2 = _ConvBNPReLU(64 + nclass, nclass, 1, **kwargs)  # 1x1 卷积：编码器 l2 特征(64) 与上采样预测(nclass) 拼接后投影回 nclass
        self.project_l1 = nn.Sequential(  # 1/2 分辨率处的最终投影
            nn.Dropout2d(0.1),  # 2D Dropout p=0.1
            nn.Conv2d(32 + nclass, nclass, 1, bias=False))  # 1x1 卷积：l1 特征(32) + 上采样预测(nclass) → nclass，无偏置

        self.aux = aux  # 保存辅助输出开关

        self.__setattr__('exclusive', ['proj_L4_C', 'pspMod', 'project_l3', 'act_l3', 'project_l2', 'project_l1'])  # 登记除编码器外的自有模块名单（编码器参数另列，供训练脚本区分参数组）

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像 (N, 3, H, W)；返回元组 (主输出 (N,nclass,H,W) [, 辅助输出 (N,nclass,H,W)])
    def forward(self, x):  # 前向：编码器四层特征 → 自顶向下逐级融合投影 → 上采样到原图尺寸
        size = x.size()[2:]  # 记录输入空间尺寸 (H, W)
        out_l1, out_l2, out_l3, out_l4 = self.pretrained(x, seg=True)  # 编码器四层特征：l1 (N,32,H/2)、l2 (N,64,H/4)、l3 (N,128,H/8)、l4 (N,256,H/16)（seg=True 只返回特征不做分类）
        out_l4_proj = self.proj_L4_C(out_l4)  # 最深层 1x1 压缩 → (N, 128, H/16, W/16)
        up_l4_to_l3 = F.interpolate(out_l4_proj, scale_factor=2, mode='bilinear', align_corners=True)  # ×2 上采样到 1/8 分辨率 → (N, 128, H/8, W/8)
        merged_l3_upl4 = self.pspMod(torch.cat([out_l3, up_l4_to_l3], 1))  # l3 特征(128) 与上采样的 l4(128) 拼接(256 通道) → EESP+PSP → (N, 128, H/8, W/8)
        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)  # 投影到类别数（激活前 logits）→ (N, nclass, H/8, W/8)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)  # BN+PReLU 激活 → (N, nclass, H/8, W/8)
        out_up_l3 = F.interpolate(proj_merge_l3, scale_factor=2, mode='bilinear', align_corners=True)  # ×2 上采样到 1/4 分辨率 → (N, nclass, H/4, W/4)
        merge_l2 = self.project_l2(torch.cat([out_l2, out_up_l3], 1))  # l2 特征(64) 与 1/8 分辨率预测拼接 → 1x1 投影 → (N, nclass, H/4, W/4)
        out_up_l2 = F.interpolate(merge_l2, scale_factor=2, mode='bilinear', align_corners=True)  # ×2 上采样到 1/2 分辨率 → (N, nclass, H/2, W/2)
        merge_l1 = self.project_l1(torch.cat([out_l1, out_up_l2], 1))  # l1 特征(32) 与 1/4 分辨率预测拼接 → 1x1 投影 → (N, nclass, H/2, W/2)

        outputs = list()  # 输出收集列表
        merge1_l1 = F.interpolate(merge_l1, scale_factor=2, mode='bilinear', align_corners=True)  # ×2 上采样到原尺寸 → (N, nclass, H, W)，即主输出
        outputs.append(merge1_l1)  # 收集主输出
        if self.aux:  # 启用辅助输出时
            # different from paper
            auxout = F.interpolate(proj_merge_l3_bef_act, size, mode='bilinear', align_corners=True)  # 取激活前的 1/8 分辨率预测，上采样到原尺寸 → (N, nclass, H, W)（原英文注释保留：与论文做法不同）
            outputs.append(auxout)  # 收集辅助输出

        return tuple(outputs)  # 返回 (主输出[, 辅助输出]) 元组


# different from PSPNet
class _PSPModule(nn.Module):  # PSP 变体模块（原英文注释保留：与 PSPNet 不同）——用逐级 stride=2 均值池化代替 PSPNet 的自适应多尺寸池化
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：输入通道数（128）
    # out_channels：投影输出通道数（默认 1024，本模型传 128）
    # sizes：金字塔级数元组（默认 4 级，仅决定分支个数，不决定池化输出尺寸）
    # **kwargs：透传给 _ConvBNPReLU
    def __init__(self, in_channels, out_channels=1024, sizes=(1, 2, 4, 8), **kwargs):  # 构造函数
        super(_PSPModule, self).__init__()  # 调用父类构造函数
        self.stages = nn.ModuleList(  # 各级分支的卷积列表
            [nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False) for _ in sizes])  # 每级一个 3x3 深度卷积（stride=1 padding=1，groups=in_channels 逐通道、深度可分离，无偏置），共 len(sizes)=4 个
        self.project = _ConvBNPReLU(in_channels * (len(sizes) + 1), out_channels, 1, 1, **kwargs)  # 1x1 卷积：原始特征 + 4 级池化特征拼接（5*128=640 通道）→ out_channels

    # ----------------------- forward 参数说明 -----------------------
    # x：(N, in_channels, H', W')；输出 (N, out_channels, H', W')
    def forward(self, x):  # 前向：逐级均值池化构成金字塔，各级卷积后上采样回原尺寸再拼接投影
        size = x.size()[2:]  # 记录输入空间尺寸 (H', W')
        feats = [x]  # 特征列表：首元素为原始特征（相当于保留 1x1 全局分支）
        for stage in self.stages:  # 逐级处理（每级在上一级输出基础上再减半）
            x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)  # 3x3 均值池化 stride=2 → 依次得到 1/2、1/4、1/8、1/16 尺度
            upsampled = F.interpolate(stage(x), size, mode='bilinear', align_corners=True)  # 该级深度卷积后双线性上采样回原尺寸（多尺度对齐）
            feats.append(upsampled)  # 收集该级特征
        return self.project(torch.cat(feats, dim=1))  # 5 组特征通道拼接后 1x1 投影 → (N, out_channels, H', W')


# ----------------------- get_espnet 参数说明 -----------------------
# dataset：数据集名称（决定类别数）
# backbone：占位参数（本模型未使用）
# pretrained：是否加载本模型的分割预训练权重
# root：权重文件查找根目录
# pretrained_base：是否加载编码器预训练权重（【注意】传 True 会触发 eespnet 内部 ValueError）
# **kwargs：透传；pretrained=True 时须包含 local_rank 用于指定加载设备
def get_espnet(dataset='pascal_voc', backbone='', pretrained=False, root='~/.torch/models',  # 工厂函数：按数据集构建 ESPNetV2 并可选加载权重
               pretrained_base=False, **kwargs):  # 续行：默认不加载编码器预训练权重
    acronyms = {  # 数据集名 → 权重文件名缩写映射
        'pascal_voc': 'pascal_voc',  # Pascal VOC 2012
        'pascal_aug': 'pascal_aug',  # VOC 增广集（SBD）
        'ade20k': 'ade',  # ADE20K 场景解析数据集
        'coco': 'coco',  # COCO Stuff
        'citys': 'citys',  # Cityscapes 城市街景（19 类）
    }  # 缩写映射表结束
    from core.data.dataloader import datasets  # 延迟导入数据集注册表（绝对导入写法）
    model = ESPNetV2(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建模型
    if pretrained:  # 若需加载分割预训练权重
        from .model_store import get_model_file  # 延迟导入权重文件路径解析工具
        device = torch.device(kwargs['local_rank'])  # 【注意】可疑写法：kwargs 未包含 local_rank 时会 KeyError；且以 int 直接构造 torch.device 的行为依赖 PyTorch 版本（更稳妥写法为 torch.device('cuda', rank)）
        model.load_state_dict(torch.load(get_model_file('espnet_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 拼出权重文件名（【注意】backbone 为空串，故实际形如 espnet__citys）
                              map_location=device))  # 将权重映射到指定设备
    return model  # 返回构建/加载完成的模型


def get_espnet_citys(**kwargs):  # 便捷构造函数：Cityscapes 数据集（19 类）ESPNetV2
    return get_espnet('citys', **kwargs)  # 委托给 get_espnet


if __name__ == '__main__':  # 直接运行本文件时的小型自测（不触发训练）
    model = get_espnet_citys()  # 构建 Cityscapes 版 ESPNetV2（pretrained 默认 False，仅随机初始化）
