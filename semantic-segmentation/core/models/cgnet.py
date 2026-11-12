# ============================================================================
# 【文件说明】cgnet.py —— 上下文引导网络 CGNet（Context Guided Network）的 PyTorch 实现。
#              纯轻量级语义分割网络（不加载预训练骨干、从零训练），由 3 个 stage 构成：
#              stage1 浅层卷积 + stage2/stage3 堆叠上下文引导块 CGB，配合输入下采样注入
#              （Input Injection，把原图池化后与深层特征拼接）实现细节补充，
#              最后 1x1 卷积分类 + 双线性上采样输出。
# 【核心思想】论文《CGNet: A Light-weight Context Guided Network for Semantic
#            Segmentation》(Tianyi Wu et al., arXiv:1811.08201)。每个 CGB 并行提取
#            “局部特征”(f_loc，空洞率 1 的深度卷积) 与“周围上下文特征”(f_sur，
#            更大空洞率的深度卷积)，拼接后再经全局上下文模块 f_glo（SE 式通道注意力）
#            加权——即“局部 + 周围 + 全局”三级上下文引导；网络窄而深、全部用
#            深度可分离卷积，参数量约 0.5M，适合嵌入式端实时分割。
# ============================================================================
"""Context Guided Network for Semantic Segmentation"""  # 原英文 docstring 保留：面向语义分割的上下文引导网络
import torch  # PyTorch 主库：张量运算与权重加载
import torch.nn as nn  # 神经网络模块：nn.Module、卷积/BN/PReLU/池化等网络层
import torch.nn.functional as F  # 函数式接口：双线性上采样 F.interpolate 等

from core.nn import _ConvBNPReLU, _BNPReLU  # 导入基础模块：“卷积+BN+PReLU”与“BN+PReLU”（定义在 core/nn/basic.py）

__all__ = ['CGNet', 'get_cgnet', 'get_cgnet_citys']  # 模块公开接口：模型类 + 两个工厂函数


class CGNet(nn.Module):  # CGNet 主网络：三个 stage 顺序连接，直接从 RGB 图像开始学习分割
    # 类 docstring：CGNet 模型说明与参数（原英文保留）
    r"""CGNet

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Tianyi Wu, et al. "CGNet: A Light-weight Context Guided Network for Semantic Segmentation."
        arXiv preprint arXiv:1811.08201 (2018).
    """  # 上方英文 docstring 原样保留：参数说明与论文引用

    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数（输出通道数）
    # backbone：接口占位参数，CGNet 不使用外部骨干【注意】
    # aux：接口占位参数，本实现未包含辅助分支【注意】
    # jpu：接口占位参数，未使用【注意】
    # pretrained_base：接口占位参数，未使用（CGNet 从零训练、无骨干预训练）【注意】
    # M：stage2 中上下文引导块 CGB 的总数（默认 3 = 1 个下采样块 + M-1 个等分辨率块）
    # N：stage3 中 CGB 的总数（默认 21 = 1 个下采样块 + N-1 个等分辨率块）
    # **kwargs：透传给各子模块（如 norm_layer）
    def __init__(self, nclass, backbone='', aux=False, jpu=False, pretrained_base=True, M=3, N=21, **kwargs):  # 构造函数：搭建三个 stage 与输出头
        super(CGNet, self).__init__()  # 调用父类 nn.Module 构造函数
        # stage 1
        self.stage1_0 = _ConvBNPReLU(3, 32, 3, 2, 1, **kwargs)  # 3x3 卷积 stride=2 padding=1（普通卷积）→ (N, 32, H/2, W/2)
        self.stage1_1 = _ConvBNPReLU(32, 32, 3, 1, 1, **kwargs)  # 3x3 卷积 s=1 p=1 → (N, 32, H/2, W/2)
        self.stage1_2 = _ConvBNPReLU(32, 32, 3, 1, 1, **kwargs)  # 3x3 卷积 s=1 p=1 → (N, 32, H/2, W/2)

        self.sample1 = _InputInjection(1)  # 输入注入 1：原图 1 次 3x3 均值池化下采样 → (N, 3, H/2, W/2)
        self.sample2 = _InputInjection(2)  # 输入注入 2：原图 2 次 3x3 均值池化下采样 → (N, 3, H/4, W/4)
        self.bn_prelu1 = _BNPReLU(32 + 3, **kwargs)  # BN+PReLU：作用于 stage1 输出与注入原图拼接后的 35 通道

        # stage 2
        self.stage2_0 = ContextGuidedBlock(32 + 3, 64, dilation=2, reduction=8, down=True, residual=False, **kwargs)  # 下采样 CGB：35→64，f_sur 空洞率 2，注意力压缩比 8，stride=2 → (N, 64, H/4, W/4)，因尺寸变化不使用残差
        self.stage2 = nn.ModuleList()  # 存放 stage2 其余 M-1 个等分辨率 CGB
        for i in range(0, M - 1):  # 循环添加 M-1=2 个块
            self.stage2.append(ContextGuidedBlock(64, 64, dilation=2, reduction=8, **kwargs))  # 等分辨率 CGB：64→64，空洞率 2，带残差
        self.bn_prelu2 = _BNPReLU(128 + 3, **kwargs)  # BN+PReLU：作用于 [out1(64)+out1_0(64)+注入原图(3)] 拼接后的 131 通道

        # stage 3
        self.stage3_0 = ContextGuidedBlock(128 + 3, 128, dilation=4, reduction=16, down=True, residual=False, **kwargs)  # 下采样 CGB：131→128，空洞率 4，stride=2 → (N, 128, H/8, W/8)
        self.stage3 = nn.ModuleList()  # 存放 stage3 其余 N-1 个等分辨率 CGB
        for i in range(0, N - 1):  # 循环添加 N-1=20 个块
            self.stage3.append(ContextGuidedBlock(128, 128, dilation=4, reduction=16, **kwargs))  # 等分辨率 CGB：128→128，空洞率 4，带残差
        self.bn_prelu3 = _BNPReLU(256, **kwargs)  # BN+PReLU：作用于 [out2_0(128)+out2(128)] 拼接后的 256 通道

        self.head = nn.Sequential(  # 输出头：Dropout + 1x1 分类卷积
            nn.Dropout2d(0.1, False),  # 2D Dropout p=0.1，inplace=False；【注意】nn.Dropout2d 在 PyTorch≥1.11 已弃用（官方建议改用 nn.Dropout）
            nn.Conv2d(256, nclass, 1))  # 1x1 卷积 256→nclass → (N, nclass, H/8, W/8)

        self.__setattr__('exclusive', ['stage1_0', 'stage1_1', 'stage1_2', 'sample1', 'sample2',  # 登记模型全部自有模块名单（CGNet 无预训练骨干，所有参数均需训练）
                                       'bn_prelu1', 'stage2_0', 'stage2', 'bn_prelu2', 'stage3_0',  # 续行：stage1/stage2 相关模块
                                       'stage3', 'bn_prelu3', 'head'])  # 续行：stage3 相关模块与输出头

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像张量 (N, 3, H, W)；返回单元素元组 ((N, nclass, H, W),)
    def forward(self, x):  # 前向：三 stage 逐级提取（含原图注入与多尺度拼接）→ 分类 → 上采样
        size = x.size()[2:]  # 记录输入尺寸 (H, W)
        # stage1
        out0 = self.stage1_0(x)  # 3x3 s2 卷积 → (N, 32, H/2, W/2)
        out0 = self.stage1_1(out0)  # → (N, 32, H/2, W/2)
        out0 = self.stage1_2(out0)  # → (N, 32, H/2, W/2)

        inp1 = self.sample1(x)  # 原图 1 次均值池化下采样 → (N, 3, H/2, W/2)
        inp2 = self.sample2(x)  # 原图 2 次均值池化下采样 → (N, 3, H/4, W/4)

        # stage 2
        out0_cat = self.bn_prelu1(torch.cat([out0, inp1], dim=1))  # stage1 特征与同尺寸原图拼接（32+3=35 通道）→ BN+PReLU → (N, 35, H/2, W/2)
        out1_0 = self.stage2_0(out0_cat)  # 下采样 CGB → (N, 64, H/4, W/4)
        for i, layer in enumerate(self.stage2):  # 依次通过 M-1 个等分辨率 CGB（【注意】该 enumerate+if i==0 写法可直接用 out1=out1_0 初始化后循环简化，功能等价）
            if i == 0:  # 首块以下采样块输出为输入
                out1 = layer(out1_0)  # → (N, 64, H/4, W/4)
            else:  # 后续块链式连接
                out1 = layer(out1)  # → (N, 64, H/4, W/4)
        out1_cat = self.bn_prelu2(torch.cat([out1, out1_0, inp2], dim=1))  # [out1(64)+out1_0(64)+原图(3)] 拼接（131 通道）→ BN+PReLU → (N, 131, H/4, W/4)

        # stage 3
        out2_0 = self.stage3_0(out1_cat)  # 下采样 CGB → (N, 128, H/8, W/8)
        for i, layer in enumerate(self.stage3):  # 依次通过 N-1 个等分辨率 CGB
            if i == 0:  # 首块以下采样块输出为输入
                out2 = layer(out2_0)  # → (N, 128, H/8, W/8)
            else:  # 后续块链式连接
                out2 = layer(out2)  # → (N, 128, H/8, W/8)
        out2_cat = self.bn_prelu3(torch.cat([out2_0, out2], dim=1))  # 下采样块输出与链式输出拼接（128+128=256 通道）→ BN+PReLU → (N, 256, H/8, W/8)

        outputs = []  # 输出收集列表
        out = self.head(out2_cat)  # 分类头 → (N, nclass, H/8, W/8)
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)  # 双线性上采样回原尺寸 → (N, nclass, H, W)
        outputs.append(out)  # 收集主输出
        return tuple(outputs)  # 返回单元素元组（与其他模型输出接口保持一致）


class _ChannelWiseConv(nn.Module):  # 逐通道卷积（深度卷积）：groups=in_channels，每个通道独立卷积
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels / out_channels：输入/输出通道数（CGB 中两者相等）
    # dilation：空洞率（f_sur 分支用它扩大感受野以建模周围上下文）
    # **kwargs：兼容透传（本类中未使用）
    def __init__(self, in_channels, out_channels, dilation=1, **kwargs):  # 构造函数
        super(_ChannelWiseConv, self).__init__()  # 调用父类构造函数
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, dilation, dilation, groups=in_channels, bias=False)  # 3x3 深度卷积：stride=1，padding=dilation，空洞率=dilation，groups=in_channels（逐通道、深度可分离），无偏置

    # ----------------------- forward 参数说明 -----------------------
    # x：(N, in_channels, H', W')；输出 (N, out_channels, H', W')
    def forward(self, x):  # 前向：逐通道空洞卷积
        x = self.conv(x)  # 空间尺寸不变（padding 与空洞率相等时尺寸守恒）
        return x  # 返回卷积结果


class _FGlo(nn.Module):  # 全局上下文模块（SE 式通道注意力）：聚合全局信息生成通道权重
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：输入/输出通道数
    # reduction：FC 瓶颈压缩比（默认 16）
    # **kwargs：兼容透传（本类中未使用）
    def __init__(self, in_channels, reduction=16, **kwargs):  # 构造函数
        super(_FGlo, self).__init__()  # 调用父类构造函数
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 → (N, C, 1, 1)
        self.fc = nn.Sequential(  # 两层全连接瓶颈结构
            nn.Linear(in_channels, in_channels // reduction),  # 降维：C → C/r
            nn.ReLU(True),  # 就地 ReLU 激活
            nn.Linear(in_channels // reduction, in_channels),  # 升维：C/r → C
            nn.Sigmoid())  # Sigmoid 得到 (0,1) 通道权重

    # ----------------------- forward 参数说明 -----------------------
    # x：(N, C, H', W')；输出同形状（按通道加权后）
    def forward(self, x):  # 前向：GAP → FC 瓶颈 → 通道加权
        n, c, _, _ = x.size()  # 拆出批量 N 与通道数 C
        out = self.gap(x).view(n, c)  # 全局池化并展平 → (N, C)
        out = self.fc(out).view(n, c, 1, 1)  # 计算通道权重并恢复为 (N, C, 1, 1)
        return x * out  # 广播相乘，按通道加权特征


class _InputInjection(nn.Module):  # 输入注入模块：把原始图像下采样后注入深层特征，补充低层细节
    # ----------------------- __init__ 参数说明 -----------------------
    # ratio：下采样次数（每次 3x3 均值池化 stride=2，尺寸减半）
    def __init__(self, ratio):  # 构造函数
        super(_InputInjection, self).__init__()  # 调用父类构造函数
        self.pool = nn.ModuleList()  # 池化层列表
        for i in range(0, ratio):  # 循环创建 ratio 个池化层
            self.pool.append(nn.AvgPool2d(3, 2, 1))  # 3x3 均值池化，stride=2，padding=1（尺寸减半）

    # ----------------------- forward 参数说明 -----------------------
    # x：原始图像 (N, 3, H, W)；输出 (N, 3, H/2^ratio, W/2^ratio)
    def forward(self, x):  # 前向：连续 ratio 次 2 倍下采样
        for pool in self.pool:  # 逐个应用池化层
            x = pool(x)  # 空间尺寸减半
        return x  # 返回下采样后的图像（通道数仍为 3）


class _ConcatInjection(nn.Module):  # 拼接注入模块：特征拼接 + BN + PReLU
    # 【注意】本类在本文件中定义但从未被使用（CGNet 前向中直接用 bn_prelu1/2/3 完成同样功能），属于遗留/冗余代码
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：拼接后的总通道数
    # norm_layer：归一化层类型
    # **kwargs：兼容透传（本类中未使用）
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(_ConcatInjection, self).__init__()  # 调用父类构造函数
        self.bn = norm_layer(in_channels)  # BN 层
        self.prelu = nn.PReLU(in_channels)  # 逐通道 PReLU 激活

    # ----------------------- forward 参数说明 -----------------------
    # x1 / x2：待拼接的两个特征（通道维拼接，空间尺寸需一致）
    def forward(self, x1, x2):  # 前向：拼接 → BN → PReLU
        out = torch.cat([x1, x2], dim=1)  # 通道维拼接
        out = self.bn(out)  # 归一化
        out = self.prelu(out)  # 激活
        return out  # 返回拼接激活后的特征


class ContextGuidedBlock(nn.Module):  # 上下文引导块 CGB：局部特征 + 周围上下文 + 全局上下文三路引导
    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels / out_channels：输入/输出通道数
    # dilation：f_sur 周围上下文分支的空洞率（stage2 用 2，stage3 用 4）
    # reduction：f_glo 全局注意力的压缩比
    # down：True 表示块内先 stride=2 下采样（各 stage 首块）；False 表示等分辨率
    # residual：是否使用残差连接（下采样块因尺寸/通道变化不使用）
    # norm_layer：归一化层类型
    # **kwargs：透传给子模块
    def __init__(self, in_channels, out_channels, dilation=2, reduction=16, down=False,  # 构造函数首行
                 residual=True, norm_layer=nn.BatchNorm2d, **kwargs):  # 续行：残差开关与归一化层参数
        super(ContextGuidedBlock, self).__init__()  # 调用父类构造函数
        inter_channels = out_channels // 2 if not down else out_channels  # 中间通道数：等分辨率块取一半（更轻量），下采样块取全量
        if down:  # 下采样分支（各 stage 首块）
            self.conv = _ConvBNPReLU(in_channels, inter_channels, 3, 2, 1, norm_layer=norm_layer, **kwargs)  # 3x3 卷积 stride=2 padding=1，通道 in→out，空间尺寸减半
            self.reduce = nn.Conv2d(inter_channels * 2, out_channels, 1, bias=False)  # 1x1 卷积：两路拼接（2*out 通道）降回 out 通道，无偏置
        else:  # 等分辨率分支
            self.conv = _ConvBNPReLU(in_channels, inter_channels, 1, 1, 0, norm_layer=norm_layer, **kwargs)  # 1x1 卷积降通道 in→out/2（省参数，后续由深度卷积处理）
        self.f_loc = _ChannelWiseConv(inter_channels, inter_channels, **kwargs)  # 局部特征分支：3x3 深度卷积，空洞率 1
        self.f_sur = _ChannelWiseConv(inter_channels, inter_channels, dilation, **kwargs)  # 周围上下文分支：3x3 深度卷积，空洞率 dilation（更大感受野）
        self.bn = norm_layer(inter_channels * 2)  # 两路拼接后（2*inter 通道）的 BN
        self.prelu = nn.PReLU(inter_channels * 2)  # 逐通道 PReLU 激活
        self.f_glo = _FGlo(out_channels, reduction, **kwargs)  # 全局上下文模块：SE 式通道注意力
        self.down = down  # 保存下采样标志，forward 中使用
        self.residual = residual  # 保存残差开关，forward 中使用

    # ----------------------- forward 参数说明 -----------------------
    # x：(N, in_channels, H', W')
    # 输出：down=True 时 (N, out_channels, H'/2, W'/2)；down=False 时 (N, out_channels, H', W')
    def forward(self, x):  # 前向：降维 → 局部/周围两路并行 → 拼接 → 全局注意力加权 → 可选残差
        out = self.conv(x)  # 降维卷积 → (N, inter, H', W')（等分辨率）或 (N, out, H'/2, W'/2)（下采样）
        loc = self.f_loc(out)  # 局部特征（空洞率 1 深度卷积）→ (N, inter, ·, ·)
        sur = self.f_sur(out)  # 周围上下文特征（空洞率 dilation 深度卷积）→ (N, inter, ·, ·)

        joi_feat = torch.cat([loc, sur], dim=1)  # 局部与周围两路特征通道拼接 → (N, 2*inter, ·, ·)
        joi_feat = self.prelu(self.bn(joi_feat))  # BN + PReLU 融合激活
        if self.down:  # 下采样块需把 2*out 通道降回 out
            joi_feat = self.reduce(joi_feat)  # 1x1 卷积 → (N, out, H'/2, W'/2)

        out = self.f_glo(joi_feat)  # 全局上下文通道注意力加权 → (N, out, ·, ·)
        if self.residual:  # 等分辨率块启用残差
            out = out + x  # 逐元素残差相加（要求 in_channels == out_channels 且空间尺寸一致）

        return out  # 返回 CGB 输出


# ----------------------- get_cgnet 参数说明 -----------------------
# dataset：数据集名称（决定类别数）
# backbone：占位参数（CGNet 不使用外部骨干）
# pretrained：是否加载本模型的分割预训练权重
# root：权重文件查找根目录
# pretrained_base：占位参数（CGNet 无骨干预训练权重可加载）
# **kwargs：透传给 CGNet；pretrained=True 时须包含 local_rank 用于指定加载设备
def get_cgnet(dataset='citys', backbone='', pretrained=False, root='~/.torch/models', pretrained_base=True, **kwargs):  # 工厂函数：按数据集构建 CGNet 并可选加载权重
    acronyms = {  # 数据集名 → 权重文件名缩写映射
        'pascal_voc': 'pascal_voc',  # Pascal VOC 2012
        'pascal_aug': 'pascal_aug',  # VOC 增广集（SBD）
        'ade20k': 'ade',  # ADE20K 场景解析数据集
        'coco': 'coco',  # COCO Stuff
        'citys': 'citys',  # Cityscapes 城市街景（19 类）
    }  # 缩写映射表结束
    from core.data.dataloader import datasets  # 延迟导入数据集注册表（此处用绝对导入，与 bisenet/dfanet 的相对导入风格不一致，但两者均可运行）
    model = CGNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建模型
    if pretrained:  # 若需加载分割预训练权重
        from .model_store import get_model_file  # 延迟导入权重文件路径解析工具
        device = torch.device(kwargs['local_rank'])  # 【注意】可疑写法：kwargs 未包含 local_rank 时会 KeyError；且以 int 直接构造 torch.device 的行为依赖 PyTorch 版本（更稳妥写法为 torch.device('cuda', rank)）
        model.load_state_dict(torch.load(get_model_file('cgnet_%s' % (acronyms[dataset]), root=root),  # 拼出权重文件名（如 cgnet_citys.pth）并读取
                              map_location=device))  # 将权重映射到指定设备
    return model  # 返回构建/加载完成的模型


def get_cgnet_citys(**kwargs):  # 便捷构造函数：Cityscapes 数据集（19 类）CGNet
    return get_cgnet('citys', '', **kwargs)  # 委托给 get_cgnet（骨干传空字符串）


if __name__ == '__main__':  # 直接运行本文件时的小型自测（不触发训练）
    model = get_cgnet_citys()  # 构建 Cityscapes 版 CGNet（pretrained 默认 False，仅随机初始化、不下载权重）
    print(model)  # 打印网络结构
