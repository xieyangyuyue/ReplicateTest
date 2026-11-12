# ============================================================================
# 【文件说明】lednet.py —— 实现 LEDNet（Lightweight Encoder-Decoder Network）
#              轻量级实时语义分割网络，工程位置 core/models/lednet.py，
#              包含 LEDNet 主网络、下采样模块、SSnbt 分裂-混洗非瓶颈块、
#              APN 注意力金字塔解码器及 get_lednet 工厂函数。
# 【核心思想】编码器由"下采样模块（Downsampling）+ 分裂-混洗非瓶颈块（SSnbt）"堆叠：
#            SSnbt 把通道一分为二，分别走两条非对称卷积支路（后段带空洞卷积），
#            拼接后做残差并执行 ShuffleNet 式通道混洗，参数极少却能获得大感受野；
#            解码器为注意力金字塔网络（APN）：先逐级下采样形成特征金字塔，
#            再自顶向下上采样累加融合，最后与 1x1 分类图相乘（注意力门控）、
#            与全局平均池化特征相加（注入图像级上下文）。
# 【论文名称】Yu Wang, et al. "LEDNet: A Lightweight Encoder-Decoder Network for
#            Real-Time Semantic Segmentation." arXiv preprint arXiv:1905.02423 (2019).
# ============================================================================
"""LEDNet: A Lightweight Encoder-Decoder Network for Real-time Semantic Segmentation"""  # 模块级英文 docstring，原样保留
import torch  # PyTorch 主库：张量拼接/重排（cat、transpose）、权重加载 torch.load
import torch.nn as nn  # 神经网络模块库：Conv2d、BatchNorm2d、ReLU、Sequential 等
import torch.nn.functional as F  # 函数式接口：F.interpolate 双线性上/下采样

from core.nn import _ConvBNReLU  # 从 core/nn 导入"卷积+BN+ReLU"基础单元（APN 解码器使用）【注意】绝对导入，与 icnet 的相对导入风格不同

__all__ = ['LEDNet', 'get_lednet', 'get_lednet_citys']  # 模块公开接口：限定星号导入时暴露的符号
class LEDNet(nn.Module):  # LEDNet 主网络类：轻量编码器 + APN 注意力金字塔解码器
    # 类 docstring：LEDNet 模型说明与参数（原英文保留）
    r"""LEDNet

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
        Yu Wang, et al. "LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation."
        arXiv preprint arXiv:1905.02423 (2019).
    """  # 上方为原英文类 docstring（含参数说明与文献引用），原样保留，不做任何改动

    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数（决定解码器输出通道数）
    # backbone：骨干网类型（LEDNet 无骨干网，仅为统一接口保留）
    # aux：辅助损失开关（仅为统一接口保留，本类未实现）
    # jpu：JPU 开关（仅为统一接口保留，本类未使用）
    # pretrained_base：是否加载骨干预训练权重（无骨干网，未使用）
    # **kwargs：透传给 SSnbt 等子模块（如 norm_layer）
    def __init__(self, nclass, backbone='', aux=False, jpu=False, pretrained_base=True, **kwargs):  # 构造函数【注意】backbone/aux/jpu/pretrained_base 四个参数在类内均未使用
        super(LEDNet, self).__init__()  # 调用父类 nn.Module 构造函数
        self.encoder = nn.Sequential(  # 编码器：3 次下采样至 1/8 分辨率（通道 3->32->64->128），后接一串 SSnbt 空洞块
            Downsampling(3, 32),  # 下采样模块：3->32 通道，分辨率 1/1 -> 1/2
            SSnbt(32, **kwargs), SSnbt(32, **kwargs), SSnbt(32, **kwargs),  # 三个空洞率 1 的 SSnbt，形状不变
            Downsampling(32, 64),  # 下采样模块：32->64 通道，1/2 -> 1/4
            SSnbt(64, **kwargs), SSnbt(64, **kwargs),  # 两个空洞率 1 的 SSnbt
            Downsampling(64, 128),  # 下采样模块：64->128 通道，1/4 -> 1/8（编码器最终分辨率）
            SSnbt(128, **kwargs),  # 空洞率 1
            SSnbt(128, 2, **kwargs),  # 空洞率 2
            SSnbt(128, 5, **kwargs),  # 空洞率 5
            SSnbt(128, 9, **kwargs),  # 空洞率 9
            SSnbt(128, 2, **kwargs),  # 空洞率 2（第二组重复递增）
            SSnbt(128, 5, **kwargs),  # 空洞率 5
            SSnbt(128, 9, **kwargs),  # 空洞率 9
            SSnbt(128, 17, **kwargs),  # 空洞率 17：两组 2/5/9 + 17 的递增空洞率设计逐步扩大感受野
        )  # 编码器 encoder Sequential 结束
        self.decoder = APNModule(128, nclass)  # APN 解码器：输入 128 通道，输出 nclass 通道（分辨率保持 1/8）

        self.__setattr__('exclusive', ['encoder', 'decoder'])  # 训练时"独占新参数"的子模块列表（LEDNet 全部为新参数）

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像 (N, 3, H, W)
    # 返回：单元素元组，主输出形状 (N, nclass, H, W)
    def forward(self, x):  # 前向：编码 -> APN 解码 -> 双线性上采样回原分辨率
        size = x.size()[2:]  # 记录输入高宽 (H, W)，供最后恢复分辨率
        x = self.encoder(x)  # 编码器：(N,3,H,W) -> (N,128,H/8,W/8)
        x = self.decoder(x)  # APN 解码器：(N,128,H/8) -> (N,nclass,H/8)
        outputs = list()  # 输出收集列表
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 双线性上采样回输入尺寸：(N,nclass,H,W)，align_corners=True 对齐角点像素
        outputs.append(x)  # 收集主输出

        return tuple(outputs)  # 以元组返回，与其他模型的多输出接口统一


class Downsampling(nn.Module):  # LEDNet 下采样模块：双分支卷积 + 步长 1 池化修正，分辨率减半
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels / out_channels：输入/输出通道数（两个分支各输出 out_channels//2）
    def __init__(self, in_channels, out_channels, **kwargs):  # 构造函数
        super(Downsampling, self).__init__()  # 父类构造
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 3, 2, 2, bias=False)  # 分支 1：3x3 卷积，stride=2，padding=2，输出 out/2 通道
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, 3, 2, 2, bias=False)  # 分支 2：参数形态相同的另一 3x3/s2 卷积（双分支并行增加特征多样性）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)  # 2x2 池化、stride=1：裁掉 padding=2 带来的多余 1 像素，保证输出尺寸恰为输入一半

    def forward(self, x):  # 前向：两分支分别"卷积+池化"后拼接
        x1 = self.conv1(x)  # 分支 1 卷积：(N,in,H,W) -> (N,out/2,ceil(H/2)+1,W/…)（因 padding=2 多出 1 像素）
        x1 = self.pool(x1)  # 分支 1 池化修正：-> (N,out/2,H/2,W/2)（偶数尺寸时精确一半）

        x2 = self.conv2(x)  # 分支 2 卷积：同分支 1，约减半
        x2 = self.pool(x2)  # 分支 2 池化修正：-> (N,out/2,H/2,W/2)

        return torch.cat([x1, x2], dim=1)  # 沿通道维拼接：(N,out,H/2,W/2)


class SSnbt(nn.Module):  # Split-Shuffle non-bottleneck（分裂-混洗非瓶颈）块：LEDNet 编码器基本单元
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：输入通道数（内部按一半通道处理）
    # dilation：支路后段空洞卷积的空洞率（默认 1）
    # norm_layer：归一化层类型，默认 nn.BatchNorm2d
    def __init__(self, in_channels, dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(SSnbt, self).__init__()  # 父类构造
        inter_channels = in_channels // 2  # 每条支路只处理一半通道（分裂思想，参数量减半）
        self.branch1 = nn.Sequential(  # 支路 1：先 3x1 后 1x3 的非对称卷积栈，两段串联、后段带空洞
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(1, 0), bias=False),  # 竖向 3x1 卷积，padding=(1,0)，尺寸不变
            nn.ReLU(True),  # 就地 ReLU
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, 1), bias=False),  # 横向 1x3 卷积，与上一步合计等效 3x3 感受野
            norm_layer(inter_channels),  # BN
            nn.ReLU(True),  # 激活
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(dilation, 0), dilation=(dilation, 1),  # 竖向 3x1 空洞卷积：竖向空洞率 dilation、横向 1，尺寸不变
                      bias=False),  # 无 bias（后接 BN）
            nn.ReLU(True),  # 激活
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, dilation), dilation=(1, dilation),  # 横向 1x3 空洞卷积：横向空洞率 dilation，两步合计等效空洞 3x3
                      bias=False),  # 无 bias
            norm_layer(inter_channels),  # BN
            nn.ReLU(True))  # 激活（Sequential 收尾，形状不变）

        self.branch2 = nn.Sequential(  # 支路 2：先 1x3 后 3x1（卷积顺序与支路 1 相反，捕获互补的方向模式）
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, 1), bias=False),  # 横向 1x3 卷积，尺寸不变
            nn.ReLU(True),  # 激活
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(1, 0), bias=False),  # 竖向 3x1 卷积
            norm_layer(inter_channels),  # BN
            nn.ReLU(True),  # 激活
            nn.Conv2d(inter_channels, inter_channels, (1, 3), padding=(0, dilation), dilation=(1, dilation),  # 横向 1x3 空洞卷积：横向空洞率 dilation
                      bias=False),  # 无 bias
            nn.ReLU(True),  # 激活
            nn.Conv2d(inter_channels, inter_channels, (3, 1), padding=(dilation, 0), dilation=(dilation, 1),  # 竖向 3x1 空洞卷积：竖向空洞率 dilation
                      bias=False),  # 无 bias
            norm_layer(inter_channels),  # BN
            nn.ReLU(True))  # 激活（形状不变）

        self.relu = nn.ReLU(True)  # 残差相加后的激活函数

    @staticmethod  # 静态方法：不依赖实例状态
    def channel_shuffle(x, groups):  # 通道混洗：把各组通道交错重排，促进两半通道间信息交流（ShuffleNet 技巧）
        n, c, h, w = x.size()  # 取批大小 n、通道数 c、高 h、宽 w

        channels_per_group = c // groups  # 每组包含的通道数
        x = x.view(n, groups, channels_per_group, h, w)  # reshape：把通道维拆成 (组数, 每组通道数)
        x = torch.transpose(x, 1, 2).contiguous()  # 交换"组"与"组内通道"两个维度并做内存连续化
        x = x.view(n, -1, h, w)  # 展平回 (n, c, h, w)，完成通道交错重排

        return x  # 返回混洗后的张量

    def forward(self, x):  # 前向：分裂 -> 双支路 -> 拼接残差 -> 通道混洗
        # channels split
        x1, x2 = x.split(x.size(1) // 2, 1)  # 沿通道维对半分裂：x1、x2 各 (N, c/2, H, W)（原英文注释上方保留）

        x1 = self.branch1(x1)  # 支路 1 处理前一半通道，形状不变
        x2 = self.branch2(x2)  # 支路 2 处理后一半通道，形状不变

        out = torch.cat([x1, x2], dim=1)  # 沿通道维拼接回 (N, c, H, W)
        out = self.relu(out + x)  # 残差相加（跳跃连接）后 ReLU
        out = self.channel_shuffle(out, groups=2)  # 按 2 组做通道混洗，让两半通道信息互通

        return out  # 返回 (N, c, H, W)


class APNModule(nn.Module):  # 注意力金字塔网络（Attention Pyramid Network）解码器
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：输入通道数（LEDNet 中为 128）
    # nclass：分割类别数（各 level 分支的输出通道数）
    # norm_layer：归一化层类型，默认 nn.BatchNorm2d
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(APNModule, self).__init__()  # 父类构造
        self.conv1 = _ConvBNReLU(in_channels, in_channels, 3, 2, 1, norm_layer=norm_layer)  # 3x3/s2/p1 卷积：金字塔一级下采样（尺寸减半）
        self.conv2 = _ConvBNReLU(in_channels, in_channels, 5, 2, 2, norm_layer=norm_layer)  # 5x5/s2/p2 卷积：金字塔二级下采样（累计 1/4）
        self.conv3 = _ConvBNReLU(in_channels, in_channels, 7, 2, 3, norm_layer=norm_layer)  # 7x7/s2/p3 卷积：金字塔三级下采样（累计 1/8，更大的核补充上下文）
        self.level1 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer)  # 1x1：最粗尺度分支 -> nclass 通道
        self.level2 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer)  # 1x1：中间尺度分支 -> nclass 通道
        self.level3 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer)  # 1x1：细尺度分支 -> nclass 通道
        self.level4 = _ConvBNReLU(in_channels, nclass, 1, norm_layer=norm_layer)  # 1x1：与编码器原图特征相乘，充当注意力门控
        self.level5 = nn.Sequential(  # 全局分支：自适应平均池化 + 1x1 卷积，产生图像级全局上下文
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化到 1x1（任意输入尺寸均可）
            _ConvBNReLU(in_channels, nclass, 1))  # 1x1 卷积：128 -> nclass 通道，输出 (N,nclass,1,1)

    def forward(self, x):  # 前向：自底向上建金字塔，再自顶向下累加融合，最后注意力加权（x 为编码器输出 (N,128,w,h)）
        w, h = x.size()[2:]  # 取编码器输出高宽 w、h（即输入图的 1/8）
        branch3 = self.conv1(x)  # 一级下采样：(N,128,ceil(w/2),ceil(h/2))（约 1/16 原图）
        branch2 = self.conv2(branch3)  # 二级下采样：(N,128,ceil(w/4),ceil(h/4))（约 1/32 原图）
        branch1 = self.conv3(branch2)  # 三级下采样：(N,128,ceil(w/8),ceil(h/8))（约 1/64 原图）

        out = self.level1(branch1)  # 最粗尺度分类：-> (N,nclass,w/8,h/8)
        out = F.interpolate(out, ((w + 3) // 4, (h + 3) // 4), mode='bilinear', align_corners=True)  # 上采样对齐 branch2 尺寸：(w+3)//4 即 ceil(w/4)，恰好匹配两级 s2 卷积的向上取整输出
        out = self.level2(branch2) + out  # 与中间尺度分类图逐元素相加（金字塔自顶向下融合）
        out = F.interpolate(out, ((w + 1) // 2, (h + 1) // 2), mode='bilinear', align_corners=True)  # 上采样对齐 branch3 尺寸：(w+1)//2 即 ceil(w/2)
        out = self.level3(branch3) + out  # 再与细尺度分类图相加
        out = F.interpolate(out, (w, h), mode='bilinear', align_corners=True)  # 上采样回编码器输出尺寸 (w,h)
        out = self.level4(x) * out  # 注意力门控：编码器特征的 1x1 分类图与融合图逐元素相乘
        out = self.level5(x) + out  # 加上全局池化特征 (N,nclass,1,1)，广播到 (w,h)，注入图像级上下文
        return out  # 返回 (N,nclass,w,h)，仍为 1/8 原图分辨率


# ----------------------- get_lednet 参数说明 -----------------------
# dataset：数据集名称（默认 'citys' 即 Cityscapes）
# backbone：骨干网名称（LEDNet 不使用，仅保留参数）
# pretrained：是否加载本仓库提供的模型权重文件
# root：权重文件的下载/缓存目录
# pretrained_base：是否加载骨干预训练权重（无骨干网，未使用）
# **kwargs：透传给模型构造（如 local_rank、norm_layer）
def get_lednet(dataset='citys', backbone='', pretrained=False, root='~/.torch/models',
               pretrained_base=True, **kwargs):  # LEDNet 工厂函数（参数跨两行书写）
    acronyms = {  # 数据集名 -> 预训练权重文件名中的缩写映射表
        'pascal_voc': 'pascal_voc',  # Pascal VOC 数据集缩写
        'pascal_aug': 'pascal_aug',  # Pascal VOC 增强标注集缩写
        'ade20k': 'ade',  # ADE20K 场景解析数据集缩写
        'coco': 'coco',  # COCO 数据集缩写
        'citys': 'citys',  # Cityscapes 街景数据集缩写
    }  # acronyms 字典结束
    from ..data.dataloader import datasets  # 延迟导入数据集注册表【注意】此处为相对导入，enet.py 用绝对导入 core.data.dataloader，风格不一致但功能等价
    model = LEDNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建 LEDNet
    if pretrained:  # 需要加载预训练权重时
        from .model_store import get_model_file  # 延迟导入权重文件定位/下载工具
        device = torch.device(kwargs['local_rank'])  # 【注意】直接取 kwargs['local_rank']，未传该键时会抛 KeyError
        model.load_state_dict(torch.load(get_model_file('lednet_%s' % (acronyms[dataset]), root=root),  # 拼出权重文件名（如 lednet_citys）并加载【注意】torch.load 未指定 weights_only，新版 PyTorch 会有告警
                              map_location=device))  # 权重张量映射到指定设备（跨两行的调用）
    return model  # 返回构建好的模型


def get_lednet_citys(**kwargs):  # Cityscapes 专用快捷工厂函数
    return get_lednet('citys', **kwargs)  # 固定数据集为 citys


if __name__ == '__main__':  # 模块自测入口：直接运行本文件时执行
    model = get_lednet_citys()  # 仅构建模型验证可实例化（与 enet 不同，此处未做前向测试）
