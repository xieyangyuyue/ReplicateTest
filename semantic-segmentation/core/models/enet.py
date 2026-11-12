# ============================================================================
# 【文件说明】enet.py —— 实现 ENet（Efficient Neural Network）实时语义分割网络，
#              工程位置 core/models/enet.py，包含 ENet 主网络、四类基础模块
#              （InitialBlock / Bottleneck / UpsamplingBottleneck）及 get_enet 工厂函数。
# 【核心思想】专为实时分割设计的轻量编码-解码网络（参数量约 0.37M）：
#            1) 初始块将 3x3/s2 卷积与最大池化并联后拼接，早期下采样大幅削减计算量；
#            2) 编码器瓶颈块 = 1x1 降维 -> 3x3（可带空洞 dilation / 非对称 5x1+1x5 分解）
#               -> 1x1 升维的残差单元，中部 stage 用空洞率 2/4/8/16 在不降分辨率的前提下
#               逐步扩大感受野；
#            3) 解码器 UpsamplingBottleneck 借助编码器记录的最大池化索引做 MaxUnpool
#               上采样恢复细节，末尾 2x2 转置卷积回到原图分辨率。
# 【论文名称】A. Paszke, A. Chaurasia, S. Kim, E. Culurciello. "ENet: A Deep Neural
#            Network Architecture for Real-Time Semantic Segmentation."
#            arXiv preprint arXiv:1606.02147 (2016).
# ============================================================================
"""Efficient Neural Network"""  # 模块级英文 docstring，原样保留
import torch  # PyTorch 主库：张量拼接 torch.cat、权重加载 torch.load、设备管理
import torch.nn as nn  # 神经网络模块库：nn.Module、Conv2d、BatchNorm2d、PReLU、MaxUnpool2d 等层

__all__ = ['ENet', 'get_enet', 'get_enet_citys']  # 模块公开接口：限定星号导入时对外暴露的符号


class ENet(nn.Module):  # ENet 主网络类：编码-解码一体的轻量实时分割模型
    """Efficient Neural Network"""  # 原英文类 docstring，原样保留

    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数（决定最终输出通道数）
    # backbone：骨干网类型（ENet 自带结构、无需骨干网，仅为统一接口保留）
    # aux：辅助损失开关（仅为统一接口保留，本类未实现辅助分支）
    # jpu：JPU 模块开关（仅为统一接口保留，本类未使用）
    # pretrained_base：是否加载骨干预训练权重（ENet 无骨干网，本类未使用）
    # **kwargs：其余参数透传给子模块（如 norm_layer 归一化层类型）
    def __init__(self, nclass, backbone='', aux=False, jpu=False, pretrained_base=None, **kwargs):  # 构造函数【注意】backbone/aux/jpu/pretrained_base 四个参数在类内均未使用，仅为兼容仓库统一训练接口
        super(ENet, self).__init__()  # 调用父类 nn.Module 构造函数，初始化模块注册表
        # 初始块：3x3/s2 卷积（出 13 通道）与 2x2/s2 最大池化（保留 3 通道）并联拼接 -> 16 通道，分辨率降为 1/2
        self.initial = InitialBlock(13, **kwargs)  # 13 为卷积分支输出通道数，加池化分支 3 通道共 16

        # ---- stage 1：下采样至 1/4 分辨率，通道 16 -> 64 ----
        self.bottleneck1_0 = Bottleneck(16, 16, 64, downsampling=True, **kwargs)  # 下采样瓶颈块：16->64 通道，分辨率减半，返回池化索引 max_indices1 供 stage5 反池化
        self.bottleneck1_1 = Bottleneck(64, 16, 64, **kwargs)  # 常规瓶颈块：64->16->64，形状不变 (N,64,H/4,W/4)
        self.bottleneck1_2 = Bottleneck(64, 16, 64, **kwargs)  # 常规瓶颈块，形状不变
        self.bottleneck1_3 = Bottleneck(64, 16, 64, **kwargs)  # 常规瓶颈块，形状不变
        self.bottleneck1_4 = Bottleneck(64, 16, 64, **kwargs)  # 常规瓶颈块，形状不变

        # ---- stage 2：下采样至 1/8 分辨率，通道 64 -> 128，穿插空洞/非对称卷积 ----
        self.bottleneck2_0 = Bottleneck(64, 32, 128, downsampling=True, **kwargs)  # 下采样瓶颈块：64->128，分辨率减半，返回池化索引 max_indices2 供 stage4 反池化
        self.bottleneck2_1 = Bottleneck(128, 32, 128, **kwargs)  # 常规瓶颈块，空洞率 1，形状不变
        self.bottleneck2_2 = Bottleneck(128, 32, 128, dilation=2, **kwargs)  # 空洞率 2：扩大感受野，形状不变
        self.bottleneck2_3 = Bottleneck(128, 32, 128, asymmetric=True, **kwargs)  # 非对称卷积：5x1 与 1x5 两级分解，等效 5x5 感受野、参数减半
        self.bottleneck2_4 = Bottleneck(128, 32, 128, dilation=4, **kwargs)  # 空洞率 4，形状不变
        self.bottleneck2_5 = Bottleneck(128, 32, 128, **kwargs)  # 常规瓶颈块，空洞率 1，形状不变
        self.bottleneck2_6 = Bottleneck(128, 32, 128, dilation=8, **kwargs)  # 空洞率 8，形状不变
        self.bottleneck2_7 = Bottleneck(128, 32, 128, asymmetric=True, **kwargs)  # 非对称 5x1+1x5 分解卷积，形状不变
        self.bottleneck2_8 = Bottleneck(128, 32, 128, dilation=16, **kwargs)  # 空洞率 16：stage2 最大感受野

        # ---- stage 3：分辨率保持 1/8，纯空洞卷积堆叠（结构与 stage2 类似但不下采样）----
        self.bottleneck3_1 = Bottleneck(128, 32, 128, **kwargs)  # 常规瓶颈块，空洞率 1
        self.bottleneck3_2 = Bottleneck(128, 32, 128, dilation=2, **kwargs)  # 空洞率 2
        self.bottleneck3_3 = Bottleneck(128, 32, 128, asymmetric=True, **kwargs)  # 非对称 5x1+1x5 分解卷积
        self.bottleneck3_4 = Bottleneck(128, 32, 128, dilation=4, **kwargs)  # 空洞率 4
        self.bottleneck3_5 = Bottleneck(128, 32, 128, **kwargs)  # 常规瓶颈块（空洞率 1）【注意】该模块已定义但 forward 中从未被调用，疑似漏写（见 forward stage 3）
        self.bottleneck3_6 = Bottleneck(128, 32, 128, dilation=8, **kwargs)  # 空洞率 8
        self.bottleneck3_7 = Bottleneck(128, 32, 128, asymmetric=True, **kwargs)  # 非对称分解卷积
        self.bottleneck3_8 = Bottleneck(128, 32, 128, dilation=16, **kwargs)  # 空洞率 16

        # ---- stage 4：上采样回 1/4 分辨率，通道 128 -> 64（解码器开始）----
        self.bottleneck4_0 = UpsamplingBottleneck(128, 16, 64, **kwargs)  # 上采样瓶颈块：128->64，分辨率翻倍，需传入 max_indices2
        self.bottleneck4_1 = Bottleneck(64, 16, 64, **kwargs)  # 常规瓶颈块，形状不变
        self.bottleneck4_2 = Bottleneck(64, 16, 64, **kwargs)  # 常规瓶颈块，形状不变

        # ---- stage 5：上采样回 1/2 分辨率，通道 64 -> 16 ----
        self.bottleneck5_0 = UpsamplingBottleneck(64, 4, 16, **kwargs)  # 上采样瓶颈块：64->16，分辨率翻倍，需传入 max_indices1
        self.bottleneck5_1 = Bottleneck(16, 4, 16, **kwargs)  # 常规瓶颈块，形状不变

        # 最终输出层：2x2 转置卷积（反卷积），stride=2，16 -> nclass 通道，分辨率 1/2 -> 1/1
        self.fullconv = nn.ConvTranspose2d(16, nclass, 2, 2, bias=False)  # 上采样同时完成分类，输出 logits（后无 BN/激活），无 bias

        # exclusive：训练脚本中"独占训练参数"的子模块名列表（此处列出全部自有模块，不含 InitialBlock 的命名前缀冲突）
        self.__setattr__('exclusive', ['bottleneck1_0', 'bottleneck1_1', 'bottleneck1_2', 'bottleneck1_3',  # stage1 各瓶颈块名
                                       'bottleneck1_4', 'bottleneck2_0', 'bottleneck2_1', 'bottleneck2_2',  # stage1 尾块 + stage2 前段块名
                                       'bottleneck2_3', 'bottleneck2_4', 'bottleneck2_5', 'bottleneck2_6',  # stage2 中段块名
                                       'bottleneck2_7', 'bottleneck2_8', 'bottleneck3_1', 'bottleneck3_2',  # stage2 尾块 + stage3 前段块名
                                       'bottleneck3_3', 'bottleneck3_4', 'bottleneck3_5', 'bottleneck3_6',  # stage3 中段块名
                                       'bottleneck3_7', 'bottleneck3_8', 'bottleneck4_0', 'bottleneck4_1',  # stage3 尾块 + stage4 块名
                                       'bottleneck4_2', 'bottleneck5_0', 'bottleneck5_1', 'fullconv'])  # stage4 尾块 + stage5 块名 + 输出层名

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像张量，形状 (N, 3, H, W)
    # 返回：单元素元组 (pred,)，pred 形状 (N, nclass, H, W)（与其他模型的多输出接口统一）
    def forward(self, x):  # 前向传播：编码 -> 解码 -> 原分辨率分类图
        # init
        x = self.initial(x)  # 初始块：(N,3,H,W) -> (N,16,H/2,W/2)

        # stage 1
        x, max_indices1 = self.bottleneck1_0(x)  # 下采样瓶颈：(N,16,H/2,W/2) -> (N,64,H/4,W/4)，记录池化索引 max_indices1 供 stage5 使用
        x = self.bottleneck1_1(x)  # (N,64,H/4,W/4) -> (N,64,H/4,W/4)
        x = self.bottleneck1_2(x)  # 形状不变
        x = self.bottleneck1_3(x)  # 形状不变
        x = self.bottleneck1_4(x)  # 形状不变

        # stage 2
        x, max_indices2 = self.bottleneck2_0(x)  # 下采样瓶颈：(N,64,H/4) -> (N,128,H/8)，记录池化索引 max_indices2 供 stage4 使用
        x = self.bottleneck2_1(x)  # 常规块，形状不变 (N,128,H/8)
        x = self.bottleneck2_2(x)  # 空洞率 2，形状不变
        x = self.bottleneck2_3(x)  # 非对称卷积，形状不变
        x = self.bottleneck2_4(x)  # 空洞率 4，形状不变
        x = self.bottleneck2_5(x)  # 常规块，形状不变
        x = self.bottleneck2_6(x)  # 空洞率 8，形状不变
        x = self.bottleneck2_7(x)  # 非对称卷积，形状不变
        x = self.bottleneck2_8(x)  # 空洞率 16，形状不变

        # stage 3
        x = self.bottleneck3_1(x)  # 空洞率 1，形状不变 (N,128,H/8)
        x = self.bottleneck3_2(x)  # 空洞率 2，形状不变
        x = self.bottleneck3_3(x)  # 非对称卷积，形状不变
        x = self.bottleneck3_4(x)  # 空洞率 4，形状不变
        x = self.bottleneck3_6(x)  # 空洞率 8【注意】此处直接从 3_4 跳到 3_6，bottleneck3_5 被跳过未调用
        x = self.bottleneck3_7(x)  # 非对称卷积，形状不变
        x = self.bottleneck3_8(x)  # 空洞率 16，形状不变

        # stage 4
        x = self.bottleneck4_0(x, max_indices2)  # 上采样瓶颈：用 max_indices2 反池化，(N,128,H/8) -> (N,64,H/4)
        x = self.bottleneck4_1(x)  # 形状不变 (N,64,H/4)
        x = self.bottleneck4_2(x)  # 形状不变

        # stage 5
        x = self.bottleneck5_0(x, max_indices1)  # 上采样瓶颈：用 max_indices1 反池化，(N,64,H/4) -> (N,16,H/2)
        x = self.bottleneck5_1(x)  # 形状不变 (N,16,H/2)

        # out
        x = self.fullconv(x)  # 转置卷积上采样：(N,16,H/2) -> (N,nclass,H,W)，恢复原图分辨率
        return tuple([x])  # 包装成元组返回，兼容多输出模型的统一接口


class InitialBlock(nn.Module):  # ENet 初始块：卷积与池化并联，一步完成下采样
    """ENet initial block"""  # 原英文 docstring，原样保留

    # ----------------------- __init__ 参数说明 -----------------------
    # out_channels：卷积分支的输出通道数（本文件固定传 13）
    # norm_layer：归一化层类型，默认 nn.BatchNorm2d
    # **kwargs：预留扩展参数
    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(InitialBlock, self).__init__()  # 父类构造
        self.conv = nn.Conv2d(3, out_channels, 3, 2, 1, bias=False)  # 3x3 卷积，stride=2，padding=1：3 -> 13 通道，分辨率减半
        self.maxpool = nn.MaxPool2d(2, 2)  # 2x2 最大池化，stride=2：3 通道原始特征同步减半，保留低层纹理信息
        self.bn = norm_layer(out_channels + 3)  # BN 通道数为拼接后的 13+3=16
        self.act = nn.PReLU()  # PReLU 激活（斜率可学习），ENet 论文标配

    def forward(self, x):  # 输入 (N,3,H,W)
        x_conv = self.conv(x)  # 卷积分支：(N,3,H,W) -> (N,13,H/2,W/2)
        x_pool = self.maxpool(x)  # 池化分支：(N,3,H,W) -> (N,3,H/2,W/2)
        x = torch.cat([x_conv, x_pool], dim=1)  # 沿通道维拼接：(N,16,H/2,W/2)
        x = self.bn(x)  # 批归一化，形状不变
        x = self.act(x)  # PReLU 激活，形状不变
        return x  # 返回 (N,16,H/2,W/2)


class Bottleneck(nn.Module):  # ENet 瓶颈块：常规/空洞/非对称/下采样四种形态的统一残差单元
    """Bottlenecks include regular, asymmetric, downsampling, dilated"""  # 原英文 docstring，原样保留

    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：输入通道数
    # inter_channels：中间（1x1 降维后）通道数
    # out_channels：输出通道数
    # dilation：3x3 主卷积的空洞率（默认 1）
    # asymmetric：是否用 5x1 + 1x5 非对称分解卷积替代 3x3 主卷积
    # downsampling：是否为下采样块（额外输出最大池化索引）
    # norm_layer：归一化层类型，默认 nn.BatchNorm2d
    # **kwargs：预留扩展参数
    def __init__(self, in_channels, inter_channels, out_channels, dilation=1, asymmetric=False,  # 构造函数第一行（参数跨两行书写）
                 downsampling=False, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数第二行
        super(Bottleneck, self).__init__()  # 父类构造
        self.downsamping = downsampling  # 记录下采样标志【注意】属性名拼写为 downsamping（少一个 l），与单词 downsampling 不符但功能正常
        if downsampling:  # 下采样块需要额外的恒等映射分支（池化 + 1x1 投影对齐通道）
            self.maxpool = nn.MaxPool2d(2, 2, return_indices=True)  # 2x2 最大池化 stride=2，返回索引供解码器 MaxUnpool 使用
            self.conv_down = nn.Sequential(  # 恒等分支投影子模块
                nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1x1 卷积：in -> out 通道
                norm_layer(out_channels)  # BN
            )  # conv_down 恒等分支投影子模块结束

        self.conv1 = nn.Sequential(  # 主分支第一步：1x1 降维
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),  # 1x1 卷积：in -> inter 通道
            norm_layer(inter_channels),  # BN
            nn.PReLU()  # 激活
        )  # conv1 降维子模块结束

        if downsampling:  # 下采样形态：主分支用 2x2/s2 卷积降分辨率，与池化后的恒等分支尺寸对齐
            self.conv2 = nn.Sequential(  # 主分支第二步（下采样版）
                nn.Conv2d(inter_channels, inter_channels, 2, stride=2, bias=False),  # 2x2 卷积 stride=2：分辨率减半
                norm_layer(inter_channels),  # BN
                nn.PReLU()  # 激活
            )  # 下采样版 conv2 子模块结束
        else:  # 非下采样形态：分辨率保持不变
            if asymmetric:  # 非对称形态：5x5 感受野分解为 5x1 + 1x5，参数量由 25C^2 降为 10C^2
                self.conv2 = nn.Sequential(  # 主分支第二步（非对称版）
                    nn.Conv2d(inter_channels, inter_channels, (5, 1), padding=(2, 0), bias=False),  # 竖向 5x1 卷积，padding=(2,0) 保证尺寸不变
                    nn.Conv2d(inter_channels, inter_channels, (1, 5), padding=(0, 2), bias=False),  # 横向 1x5 卷积，两级串联等效 5x5 卷积
                    norm_layer(inter_channels),  # BN
                    nn.PReLU()  # 激活
                )  # 非对称版 conv2 子模块结束
            else:  # 常规/空洞形态：单个 3x3 主卷积
                self.conv2 = nn.Sequential(  # 主分支第二步（常规/空洞版）
                    nn.Conv2d(inter_channels, inter_channels, 3, dilation=dilation, padding=dilation, bias=False),  # 3x3 卷积，空洞率 dilation、padding 同值，保证尺寸不变
                    norm_layer(inter_channels),  # BN
                    nn.PReLU()  # 激活
                )  # 常规/空洞版 conv2 子模块结束
        self.conv3 = nn.Sequential(  # 主分支第三步：1x1 升维回输出通道
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),  # 1x1 卷积：inter -> out 通道
            norm_layer(out_channels),  # BN
            nn.Dropout2d(0.1)  # 空间随机失活，丢弃率 0.1（ENet 论文推荐的轻量正则化）
        )  # conv3 升维子模块结束
        self.act = nn.PReLU()  # 残差相加后的激活函数

    def forward(self, x):  # 前向：主分支 + 恒等分支残差相加
        identity = x  # 保存输入作为残差分支
        if self.downsamping:  # 下采样块：恒等分支先池化（记录索引）再 1x1 投影对齐通道
            identity, max_indices = self.maxpool(identity)  # 池化并记录最大值索引，分辨率减半
            identity = self.conv_down(identity)  # 1x1 投影到 out_channels

        out = self.conv1(x)  # 主分支 1x1 降维：(N,in,H,W) -> (N,inter,H,W)
        out = self.conv2(out)  # 主分支卷积：下采样块分辨率减半；其余块尺寸不变
        out = self.conv3(out)  # 主分支 1x1 升维 + Dropout：(N,inter,·) -> (N,out,·)
        out = self.act(out + identity)  # 残差相加后 PReLU 激活（通道数与分辨率均已对齐）

        if self.downsamping:  # 下采样块需把池化索引传给解码器
            return out, max_indices  # 返回 (特征, 最大池化索引) 元组
        else:  # 普通块
            return out  # 直接返回特征


class UpsamplingBottleneck(nn.Module):  # ENet 解码器上采样瓶颈块：反池化分支 + 卷积主分支
    """upsampling Block"""  # 原英文 docstring，原样保留

    # ----------------------- __init__ 参数说明 -----------------------
    # in_channels：输入通道数
    # inter_channels：中间通道数
    # out_channels：输出通道数
    # norm_layer：归一化层类型，默认 nn.BatchNorm2d
    # **kwargs：预留扩展参数
    def __init__(self, in_channels, inter_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(UpsamplingBottleneck, self).__init__()  # 父类构造
        self.conv = nn.Sequential(  # 恒等（上采样）分支：1x1 投影后反池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1x1 卷积：in -> out 通道
            norm_layer(out_channels)  # BN
        )  # 恒等分支 conv 子模块结束
        self.upsampling = nn.MaxUnpool2d(2)  # 2x2 反最大池化：按编码器记录的索引把最大值放回原位置，分辨率翻倍

        self.block = nn.Sequential(  # 主分支：降维 -> 转置卷积上采样 -> 升维
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),  # 1x1 降维：in -> inter
            norm_layer(inter_channels),  # BN
            nn.PReLU(),  # 激活
            nn.ConvTranspose2d(inter_channels, inter_channels, 2, 2, bias=False),  # 2x2 转置卷积 stride=2：分辨率翻倍
            norm_layer(inter_channels),  # BN
            nn.PReLU(),  # 激活
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),  # 1x1 升维：inter -> out
            norm_layer(out_channels),  # BN
            nn.Dropout2d(0.1)  # 空间 Dropout，丢弃率 0.1
        )  # 主分支 block 子模块结束
        self.act = nn.PReLU()  # 两分支相加后的激活

    # ----------------------- forward 参数说明 -----------------------
    # x：输入特征 (N, in_channels, H, W)
    # max_indices：编码器对应下采样块记录的最大池化索引
    # 返回：(N, out_channels, 2H, 2W)
    def forward(self, x, max_indices):  # 前向：两分支分别上采样后残差融合
        out_up = self.conv(x)  # 恒等分支 1x1 投影：(N,in,H,W) -> (N,out,H,W)
        out_up = self.upsampling(out_up, max_indices)  # 反池化：-> (N,out,2H,2W)

        out_ext = self.block(x)  # 主分支（含转置卷积上采样）：-> (N,out,2H,2W)
        out = self.act(out_up + out_ext)  # 两分支相加后 PReLU 激活
        return out  # 返回 (N,out,2H,2W)


# ----------------------- get_enet 参数说明 -----------------------
# dataset：数据集名称（默认 'citys' 即 Cityscapes）
# backbone：骨干网名称（ENet 不使用，仅保留参数）
# pretrained：是否加载本仓库提供的模型权重文件
# root：权重文件的下载/缓存目录
# pretrained_base：是否加载骨干预训练权重（ENet 无骨干网，未使用）
# **kwargs：透传给模型构造（如 local_rank、norm_layer）
def get_enet(dataset='citys', backbone='', pretrained=False, root='~/.torch/models', pretrained_base=True, **kwargs):  # ENet 工厂函数：按数据集类别数构建模型并可选加载权重
    acronyms = {  # 数据集名 -> 预训练权重文件名中的缩写映射表
        'pascal_voc': 'pascal_voc',  # Pascal VOC 数据集缩写
        'pascal_aug': 'pascal_aug',  # Pascal VOC 增强标注集缩写
        'ade20k': 'ade',  # ADE20K 场景解析数据集缩写
        'coco': 'coco',  # COCO 数据集缩写
        'citys': 'citys',  # Cityscapes 街景数据集缩写
    }  # acronyms 字典结束
    from core.data.dataloader import datasets  # 延迟导入数据集注册表（避免循环导入与重量级依赖）【注意】此处为绝对导入，lednet/icnet 用相对导入，风格不一致但功能等价
    model = ENet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建 ENet
    if pretrained:  # 需要加载预训练权重时
        from .model_store import get_model_file  # 延迟导入权重文件定位/下载工具
        device = torch.device(kwargs['local_rank'])  # 【注意】直接取 kwargs['local_rank']，未传该键时会抛 KeyError；此处用整数设备号构建 device
        model.load_state_dict(torch.load(get_model_file('enet_%s' % (acronyms[dataset]), root=root),  # 拼出权重文件名（如 enet_citys）并加载【注意】torch.load 未指定 weights_only，新版 PyTorch 会有告警
                              map_location=device))  # 权重张量映射到指定设备（跨两行的调用）
    return model  # 返回构建好的模型


def get_enet_citys(**kwargs):  # Cityscapes 专用快捷工厂函数
    return get_enet('citys', '', **kwargs)  # 固定数据集为 citys、骨干为空字符串


if __name__ == '__main__':  # 模块自测入口：直接运行本文件时执行
    img = torch.randn(1, 3, 512, 512)  # 构造随机输入 (1,3,512,512)
    model = get_enet_citys()  # 构建 Cityscapes 配置的 ENet
    output = model(img)  # 前向传播一次，验证网络可用
