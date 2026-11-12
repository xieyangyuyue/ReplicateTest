# ============================================================================
# 【文件说明】xception.py —— Xception 系列骨干网络（Aligned 修改版），包含：
#             SeparableConv2d（深度可分离卷积）、Block（带对齐残差连接的基本块）、
#             Xception65 / Xception71（DeepLabV3+ 常用骨干），以及供 DFANet 使用的
#             轻量骨干 XceptionA（Enc 编码级联 + FCAttention 全局通道注意力）。
# 【核心思想】Xception 以“深度可分离卷积”（depthwise 3x3 卷积逐通道空间滤波 +
#             pointwise 1x1 卷积跨通道组合）取代标准卷积，解耦空间相关性与通道相关性。
#             整体分三段：Entry flow（下采样并把通道升至 728）→ Middle flow（16 个
#             同构 Block 反复精炼特征，通道不变）→ Exit flow（升至 2048 收尾）。
#             “Aligned”指为复用 ImageNet 预训练权重而与 TF 版对齐的结构：skip 分支
#             用 1x1 卷积 + BN 对齐通道数与分辨率。工程位置：core/models/base_models/，
#             作为编码器被 DeepLabV3+、DFANet 等分割模型调用。
# ============================================================================

import torch  # PyTorch 主库：张量运算、加载预训练权重
import torch.nn as nn  # 神经网络模块库：Conv2d/BatchNorm2d/Sequential/ReLU 等
import torch.nn.functional as F  # 函数式接口：F.pad 用于深度可分离卷积的手动填充

__all__ = ['Enc', 'FCAttention', 'Xception65', 'Xception71', 'get_xception', 'get_xception_71', 'get_xception_a']  # 模块导出符号表（控制 from ... import * 的行为）


class SeparableConv2d(nn.Module):  # 深度可分离卷积模块：depthwise 3x3 卷积 → BN → pointwise 1x1 卷积
    # 参数说明：in_channels/out_channels——输入/输出通道数；kernel_size——空间核尺寸（默认 3）
    #           stride——步长（默认 1）；dilation——空洞率（默认 1）；bias——偏置开关（默认 False，因后接 BN）
    #           norm_layer——归一化层构造函数（如 nn.BatchNorm2d）
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False, norm_layer=None):  # 构造 depthwise 卷积、中间 BN 与 pointwise 卷积
        super(SeparableConv2d, self).__init__()  # 调用父类 nn.Module 初始化
        self.kernel_size = kernel_size  # 记录核尺寸，供 forward 中的 fix_padding 计算填充量
        self.dilation = dilation  # 记录空洞率，供 forward 中的 fix_padding 计算填充量

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 0, dilation, groups=in_channels,  # depthwise 卷积：groups=输入通道数（逐通道滤波）、dilation=空洞率、padding=0（same 填充由 fix_padding 手动完成）
                               bias=bias)  # 偏置开关（与上一行同属一条语句，默认关闭）
        self.bn = norm_layer(in_channels)  # depthwise 与 pointwise 之间的 BN（Xception 论文设计），通道数不变
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)  # pointwise 1x1 卷积：跨通道线性组合，通道 in→out

    def forward(self, x):  # 前向传播：手动填充 → depthwise → BN → pointwise
        x = self.fix_padding(x, self.kernel_size, self.dilation)  # 手动补零实现 same 填充（兼容空洞卷积），空间尺寸不变
        x = self.conv1(x)  # depthwise 卷积：stride=1 时 (N,C_in,H,W)→(N,C_in,H,W)，仅逐通道空间滤波
        x = self.bn(x)  # BN：形状不变
        x = self.pointwise(x)  # pointwise 1x1 卷积：(N,C_in,H,W)→(N,C_out,H,W)

        return x  # 返回输出张量

    def fix_padding(self, x, kernel_size, dilation):  # 计算并施加 same 填充（总填充量为奇数时首尾不对称）
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)  # 空洞卷积的有效核尺寸
        pad_total = kernel_size_effective - 1  # 需要的总填充量
        pad_beg = pad_total // 2  # 左/上侧填充量
        pad_end = pad_total - pad_beg  # 右/下侧填充量（与起始侧相差至多 1）
        padded_inputs = F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))  # 按（左, 右, 上, 下）顺序补零
        return padded_inputs  # 返回填充后的张量


class Block(nn.Module):  # Xception 基本残差块：若干 SeparableConv2d 串联 + 对齐的 skip 连接
    # 参数说明：in_channels/out_channels——输入/输出通道数；reps——块内 SeparableConv2d 重复次数
    #           stride——下采样步长；dilation——空洞率；norm_layer——归一化层构造函数
    #           start_with_relu——是否以 ReLU 开头；grow_first——是否第一层就升到 out_channels
    #           is_last——是否为所在 flow 的最后一个块（True 时额外追加一层卷积）
    def __init__(self, in_channels, out_channels, reps, stride=1, dilation=1, norm_layer=None,  # 构造函数（参数含义见上方注释）
                 start_with_relu=True, grow_first=True, is_last=False):  # 三个结构开关控制块内层的排列顺序
        super(Block, self).__init__()  # 调用父类初始化
        if out_channels != in_channels or stride != 1:  # 通道数变化或需下采样时，skip 分支需要投影对齐
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)  # skip：1x1 卷积同时完成通道对齐与 stride 下采样（Aligned 版关键写法）
            self.skipbn = norm_layer(out_channels)  # skip 分支的 BN
        else:  # 通道不变且不下采样
            self.skip = None  # skip 退化为恒等映射
        self.relu = nn.ReLU(True)  # ReLU 激活（位置参数 True 即 inplace=True，原地操作省显存）
        rep = list()  # 主分支的层列表
        filters = in_channels  # 主分支当前通道数
        if grow_first:  # “先升维”模式：第一层就把通道升到 out_channels
            if start_with_relu:  # 需要以 ReLU 开头时
                rep.append(self.relu)  # 先追加 ReLU
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, dilation, norm_layer=norm_layer))  # 3x3 深度可分离卷积（stride=1、空洞 dilation），通道 in→out
            rep.append(norm_layer(out_channels))  # BN(out_channels)
            filters = out_channels  # 更新当前通道数
        for i in range(reps - 1):  # 中间重复 reps-1 层，通道保持不变
            if grow_first or start_with_relu:  # 两种模式下中间层前都要激活
                rep.append(self.relu)  # 追加 ReLU
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, norm_layer=norm_layer))  # 3x3 深度可分离卷积，通道不变
            rep.append(norm_layer(filters))  # BN(filters)
        if not grow_first:  # “后升维”模式：最后一层才把通道升到 out_channels
            rep.append(self.relu)  # 追加 ReLU
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, dilation, norm_layer=norm_layer))  # 3x3 深度可分离卷积，通道 in→out
        if stride != 1:  # 需要下采样：主分支末端用带 stride 的深度可分离卷积完成下采样
            rep.append(self.relu)  # 追加 ReLU
            rep.append(SeparableConv2d(out_channels, out_channels, 3, stride, norm_layer=norm_layer))  # 3x3 深度可分离卷积 + stride 下采样；【注意】此处未传 dilation（用默认 1），且与原版 Aligned Xception 用 maxpool 下采样的写法不同
            rep.append(norm_layer(out_channels))  # BN(out_channels)
        elif is_last:  # 不下采样但为流末端块：额外再叠一层卷积增强特征
            rep.append(self.relu)  # 追加 ReLU
            rep.append(SeparableConv2d(out_channels, out_channels, 3, 1, dilation, norm_layer=norm_layer))  # 3x3 深度可分离卷积（带空洞），通道不变
            rep.append(norm_layer(out_channels))  # BN(out_channels)
        self.rep = nn.Sequential(*rep)  # 主分支打包为 Sequential

    def forward(self, x):  # 前向传播：主分支 + skip 分支相加
        out = self.rep(x)  # 主分支输出 (N, out_channels, H/stride, W/stride)
        if self.skip is not None:  # 存在投影 skip
            skip = self.skipbn(self.skip(x))  # skip：1x1 卷积 + BN，与主分支的通道/空间尺寸对齐
        else:  # 否则
            skip = x  # 恒等 skip
        out = out + skip  # 残差相加（逐元素，两分支形状一致）
        return out  # 返回相加结果


class Xception65(nn.Module):  # 65 层 Aligned Xception（DeepLabV3+ 的默认骨干）
    # 下方为原始英文 docstring，原样保留
    """Modified Aligned Xception
    """  # docstring 结束（内容未改动）

    # 参数说明：num_classes——分类头类别数（分割任务中一般不用分类头）
    #           output_stride——输出特征图相对输入的下采样倍率（32/16/8，越小分辨率越高）
    #           norm_layer——归一化层构造函数（分割训练中常换成 SyncBN 等）
    def __init__(self, num_classes=1000, output_stride=32, norm_layer=nn.BatchNorm2d):  # 初始化：按 output_stride 分配 stride 与空洞率
        super(Xception65, self).__init__()  # 调用父类初始化
        if output_stride == 32:  # 输出步长 32：全程正常下采样，不引入空洞
            entry_block3_stride = 2  # Entry 末端 block3 步长 2
            exit_block20_stride = 2  # Exit 首块 block20 步长 2
            middle_block_dilation = 1  # Middle 各块空洞率 1
            exit_block_dilations = (1, 1)  # (block20 空洞率, Exit 尾部卷积空洞率)
        elif output_stride == 16:  # 输出步长 16：最后一级下采样改为空洞卷积补偿感受野
            entry_block3_stride = 2  # block3 仍下采样（至此 /16）
            exit_block20_stride = 1  # block20 不再下采样
            middle_block_dilation = 1  # Middle 空洞率 1
            exit_block_dilations = (1, 2)  # block20 空洞 1，conv3~conv5 空洞 2
        elif output_stride == 8:  # 输出步长 8：更早停止下采样
            entry_block3_stride = 1  # block3 不下采样（保持 /8）
            exit_block20_stride = 1  # block20 不下采样
            middle_block_dilation = 2  # Middle 空洞率 2
            exit_block_dilations = (2, 4)  # block20 空洞 2，conv3~conv5 空洞 4
        else:  # 其他 output_stride 取值不支持
            raise NotImplementedError  # 抛出未实现异常
        # Entry flow
        # 入口流：标准卷积 stem + 3 个 Block，快速下采样并扩张通道
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)  # 3x3 标准卷积，stride=2、padding=1，通道 3→32
        self.bn1 = norm_layer(32)  # BN(32)
        self.relu = nn.ReLU(True)  # ReLU（inplace）

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)  # 3x3 标准卷积，stride=1，通道 32→64
        self.bn2 = norm_layer(64)  # BN(64)

        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False)  # Block1：64→128，2 次重复，stride=2，不以 ReLU 开头（stem 已激活过）
        self.block2 = Block(128, 256, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False, grow_first=True)  # Block2：128→256，stride=2，先升维模式
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, norm_layer=norm_layer,  # Block3：256→728，步长随 output_stride 变化
                            start_with_relu=True, grow_first=True, is_last=True)  # 以 ReLU 开头、先升维、Entry 末端块（is_last 追加一层卷积）

        # Middle flow
        # 中间流：16 个同构 Block（编号 4~19），通道保持 728，仅反复精炼特征
        midflow = list()  # 中间流层列表
        for i in range(4, 20):  # 循环 16 次（对应 block4~block19）；【注意】循环变量 i 未被使用
            midflow.append(Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, norm_layer=norm_layer,  # 每块：728→728，3 次重复，不下采样，空洞率随 output_stride
                                 start_with_relu=True, grow_first=True))  # 以 ReLU 开头、先升维（通道实际不变）
        self.midflow = nn.Sequential(*midflow)  # 打包为 Sequential

        # Exit flow
        # 出口流：728→1024→1536→1536→2048，其输出即分割模型取用的骨干特征
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride, dilation=exit_block_dilations[0],  # Block20：728→1024，步长/空洞率随 output_stride
                             norm_layer=norm_layer, start_with_relu=True, grow_first=False, is_last=True)  # 以 ReLU 开头、后升维、追加一层卷积
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, dilation=exit_block_dilations[1], norm_layer=norm_layer)  # 深度可分离卷积 1024→1536（空洞率 exit_block_dilations[1]）
        self.bn3 = norm_layer(1536)  # BN(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], norm_layer=norm_layer)  # 深度可分离卷积 1536→1536（空洞卷积）
        self.bn4 = norm_layer(1536)  # BN(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, dilation=exit_block_dilations[1], norm_layer=norm_layer)  # 深度可分离卷积 1536→2048（骨干最终通道 2048）
        self.bn5 = norm_layer(2048)  # BN(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化到 1x1（分类头专用；分割任务只取其前的特征）
        self.fc = nn.Linear(2048, num_classes)  # 全连接分类层 2048→num_classes

    def forward(self, x):  # 前向传播：输入 (N,3,H,W)，以下形状以 output_stride=32 为例
        # Entry flow
        # 入口流前向
        x = self.conv1(x)  # 3x3 卷积 stride=2：(N,3,H,W)→(N,32,H/2,W/2)
        x = self.bn1(x)  # BN：形状不变
        x = self.relu(x)  # ReLU：形状不变

        x = self.conv2(x)  # 3x3 卷积 stride=1：(N,32,H/2,W/2)→(N,64,H/2,W/2)
        x = self.bn2(x)  # BN：形状不变
        x = self.relu(x)  # ReLU：形状不变

        x = self.block1(x)  # Block1（stride=2）：(N,64,H/2,W/2)→(N,128,H/4,W/4)
        x = self.relu(x)  # ReLU：形状不变（Entry 中的额外激活）
        # c1 = x
        # （上行为原代码注释：预留的浅层特征 c1，分割 decoder 可在此取用）
        x = self.block2(x)  # Block2（stride=2）：(N,128,H/4,W/4)→(N,256,H/8,W/8)
        # c2 = x
        # （上行为原代码注释：预留的浅层特征 c2）
        x = self.block3(x)  # Block3：(N,256,H/8,W/8)→(N,728,H/16,W/16)（OS=8 时步长 1，保持 H/8）

        # Middle flow
        # 中间流前向：16 个 Block 反复精炼特征
        x = self.midflow(x)  # 形状不变：(N,728,H/16,W/16)（OS=8 配置时空洞率为 2）
        # c3 = x
        # （上行为原代码注释：预留的深层特征 c3）

        # Exit flow
        # 出口流前向
        x = self.block20(x)  # Block20：(N,728,H/16,W/16)→(N,1024,H/32,W/32)（OS=16/8 时不下采样）
        x = self.relu(x)  # ReLU：形状不变
        x = self.conv3(x)  # 深度可分离卷积：(N,1024,·,·)→(N,1536,·,·)
        x = self.bn3(x)  # BN：形状不变
        x = self.relu(x)  # ReLU：形状不变

        x = self.conv4(x)  # 深度可分离卷积：(N,1536,·,·)→(N,1536,·,·)
        x = self.bn4(x)  # BN：形状不变
        x = self.relu(x)  # ReLU：形状不变

        x = self.conv5(x)  # 深度可分离卷积：(N,1536,·,·)→(N,2048,·,·)
        x = self.bn5(x)  # BN：形状不变
        x = self.relu(x)  # ReLU：形状不变

        x = self.avgpool(x)  # 全局平均池化：(N,2048,H/32,W/32)→(N,2048,1,1)
        x = x.view(x.size(0), -1)  # 展平：(N,2048,1,1)→(N,2048)
        x = self.fc(x)  # 全连接分类：(N,2048)→(N,num_classes)

        return x  # 返回分类 logits (N,num_classes)


class Xception71(nn.Module):  # 71 层 Aligned Xception（Entry flow 更深的变体）
    # 下方为原始英文 docstring，原样保留
    """Modified Aligned Xception
    """  # docstring 结束（内容未改动）

    # 参数说明：num_classes——分类头类别数；output_stride——期望的输出下采样倍率（32/16/8）；norm_layer——归一化层构造函数
    # 【注意】block2 由两个 stride=2 的 Block 串联（比 Xception65 多降一次采样），
    #         实际输出分辨率只有 output_stride 参数的一半（如参数 32 时特征图为输入的 1/64），疑为未对齐的可疑写法
    def __init__(self, num_classes=1000, output_stride=32, norm_layer=nn.BatchNorm2d):  # 初始化（整体结构与 Xception65 基本一致）
        super(Xception71, self).__init__()  # 调用父类初始化
        if output_stride == 32:  # 输出步长 32 配置：无空洞
            entry_block3_stride = 2  # block3 步长 2
            exit_block20_stride = 2  # block20 步长 2
            middle_block_dilation = 1  # Middle 空洞率 1
            exit_block_dilations = (1, 1)  # Exit 空洞率 (1, 1)
        elif output_stride == 16:  # 输出步长 16 配置
            entry_block3_stride = 2  # block3 步长 2
            exit_block20_stride = 1  # block20 不下采样
            middle_block_dilation = 1  # Middle 空洞率 1
            exit_block_dilations = (1, 2)  # Exit 空洞率 (1, 2)
        elif output_stride == 8:  # 输出步长 8 配置
            entry_block3_stride = 1  # block3 不下采样
            exit_block20_stride = 1  # block20 不下采样
            middle_block_dilation = 2  # Middle 空洞率 2
            exit_block_dilations = (2, 4)  # Exit 空洞率 (2, 4)
        else:  # 不支持的取值
            raise NotImplementedError  # 抛出未实现异常
        # Entry flow
        # 入口流（与 Xception65 的差异：block2 拆成两个下采样 Block，Entry 更深）
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)  # 3x3 标准卷积 stride=2，通道 3→32
        self.bn1 = norm_layer(32)  # BN(32)
        self.relu = nn.ReLU(True)  # ReLU（inplace）

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)  # 3x3 标准卷积 stride=1，通道 32→64
        self.bn2 = norm_layer(64)  # BN(64)

        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False)  # Block1：64→128，stride=2，不以 ReLU 开头
        self.block2 = nn.Sequential(  # block2 为两个 Block 的串联（Xception71 与 65 的主要结构差异）
            Block(128, 256, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False, grow_first=True),  # 第一个 Block：128→256，stride=2 下采样
            Block(256, 728, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False, grow_first=True))  # 第二个 Block：256→728，再 stride=2 下采样（Entry 因此多降一次）
        self.block3 = Block(728, 728, reps=2, stride=entry_block3_stride, norm_layer=norm_layer,  # Block3：728→728，步长随 output_stride
                            start_with_relu=True, grow_first=True, is_last=True)  # 以 ReLU 开头、先升维、Entry 末端块

        # Middle flow
        # 中间流：16 个同构 Block，通道保持 728
        midflow = list()  # 层列表
        for i in range(4, 20):  # 16 个块（block4~block19）；【注意】循环变量 i 未被使用
            midflow.append(Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, norm_layer=norm_layer,  # 728→728，3 次重复，不下采样，空洞率随 output_stride
                                 start_with_relu=True, grow_first=True))  # 以 ReLU 开头、先升维
        self.midflow = nn.Sequential(*midflow)  # 打包为 Sequential

        # Exit flow
        # 出口流：与 Xception65 完全相同
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride, dilation=exit_block_dilations[0],  # Block20：728→1024，步长/空洞率随 output_stride
                             norm_layer=norm_layer, start_with_relu=True, grow_first=False, is_last=True)  # 以 ReLU 开头、后升维、追加一层卷积
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, dilation=exit_block_dilations[1], norm_layer=norm_layer)  # 深度可分离卷积 1024→1536
        self.bn3 = norm_layer(1536)  # BN(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], norm_layer=norm_layer)  # 深度可分离卷积 1536→1536（空洞卷积）
        self.bn4 = norm_layer(1536)  # BN(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, dilation=exit_block_dilations[1], norm_layer=norm_layer)  # 深度可分离卷积 1536→2048
        self.bn5 = norm_layer(2048)  # BN(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Linear(2048, num_classes)  # 分类全连接 2048→num_classes

    def forward(self, x):  # 前向传播：输入 (N,3,H,W)，以下形状以 output_stride=32 配置为例
        # Entry flow
        # 入口流前向
        x = self.conv1(x)  # 3x3 卷积 stride=2：(N,3,H,W)→(N,32,H/2,W/2)
        x = self.bn1(x)  # BN：形状不变
        x = self.relu(x)  # ReLU：形状不变

        x = self.conv2(x)  # 3x3 卷积 stride=1：(N,32,H/2,W/2)→(N,64,H/2,W/2)
        x = self.bn2(x)  # BN：形状不变
        x = self.relu(x)  # ReLU：形状不变

        x = self.block1(x)  # Block1（stride=2）：(N,64,H/2,W/2)→(N,128,H/4,W/4)
        x = self.relu(x)  # ReLU：形状不变
        # c1 = x
        # （上行为原代码注释：预留的浅层特征 c1）
        x = self.block2(x)  # 两个 stride=2 的 Block 串联：(N,128,H/4,W/4)→(N,256,H/8,W/8)→(N,728,H/16,W/16)
        # c2 = x
        # （上行为原代码注释：预留的浅层特征 c2）
        x = self.block3(x)  # Block3：(N,728,H/16,W/16)→(N,728,H/32,W/32)（OS=8 配置时步长 1，保持 H/16）

        # Middle flow
        # 中间流前向
        x = self.midflow(x)  # 形状不变：(N,728,H/32,W/32)（OS=8 配置时空洞率为 2）
        # c3 = x
        # （上行为原代码注释：预留的深层特征 c3）

        # Exit flow
        # 出口流前向（实际分辨率比 output_stride 参数再低一倍，见类定义处的【注意】）
        x = self.block20(x)  # Block20：(N,728,H/32,W/32)→(N,1024,H/64,W/64)（OS=16/8 配置时不下采样）
        x = self.relu(x)  # ReLU：形状不变
        x = self.conv3(x)  # 深度可分离卷积：(N,1024,·,·)→(N,1536,·,·)
        x = self.bn3(x)  # BN：形状不变
        x = self.relu(x)  # ReLU：形状不变

        x = self.conv4(x)  # 深度可分离卷积：(N,1536,·,·)→(N,1536,·,·)
        x = self.bn4(x)  # BN：形状不变
        x = self.relu(x)  # ReLU：形状不变

        x = self.conv5(x)  # 深度可分离卷积：(N,1536,·,·)→(N,2048,·,·)
        x = self.bn5(x)  # BN：形状不变
        x = self.relu(x)  # ReLU：形状不变

        x = self.avgpool(x)  # 全局平均池化：(N,2048,·,·)→(N,2048,1,1)
        x = x.view(x.size(0), -1)  # 展平：(N,2048,1,1)→(N,2048)
        x = self.fc(x)  # 分类：(N,2048)→(N,num_classes)

        return x  # 返回分类 logits (N,num_classes)【Xception71 前向结束】


# -------------------------------------------------
#                   For DFANet
# -------------------------------------------------
# （上方三行横线区为原代码注释，原样保留）以下为 DFANet 使用的轻量 Xception 组件
class BlockA(nn.Module):  # DFANet 的轻量 Xception 基本块（主分支带 1/4 瓶颈降维 + 残差 skip）
    # 参数说明：in_channels/out_channels——输入/输出通道数；stride——步长；dilation——空洞率
    #           norm_layer——归一化层构造函数；start_with_relu——是否以 ReLU 开头
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, norm_layer=None, start_with_relu=True):  # 初始化
        super(BlockA, self).__init__()  # 调用父类初始化
        if out_channels != in_channels or stride != 1:  # 通道变化或需下采样时
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)  # skip：1x1 卷积投影（带 stride 下采样）
            self.skipbn = norm_layer(out_channels)  # skip 分支的 BN
        else:  # 无需对齐时
            self.skip = None  # 恒等 skip
        self.relu = nn.ReLU(True)  # ReLU（inplace）
        rep = list()  # 主分支层列表
        inter_channels = out_channels // 4  # 瓶颈通道数 = 输出通道的 1/4（DFANet 的轻量化设计）

        if start_with_relu:  # 以 ReLU 开头
            rep.append(self.relu)  # 追加 ReLU
        rep.append(SeparableConv2d(in_channels, inter_channels, 3, 1, dilation, norm_layer=norm_layer))  # 深度可分离卷积：in→inter（stride=1）
        rep.append(norm_layer(inter_channels))  # BN(inter)

        rep.append(self.relu)  # ReLU
        rep.append(SeparableConv2d(inter_channels, inter_channels, 3, 1, dilation, norm_layer=norm_layer))  # 深度可分离卷积：inter→inter
        rep.append(norm_layer(inter_channels))  # BN(inter)

        if stride != 1:  # 需要下采样
            rep.append(self.relu)  # ReLU
            rep.append(SeparableConv2d(inter_channels, out_channels, 3, stride, norm_layer=norm_layer))  # 深度可分离卷积：inter→out，stride 下采样（【注意】未传 dilation，用默认 1）
            rep.append(norm_layer(out_channels))  # BN(out)
        else:  # 不下采样
            rep.append(self.relu)  # ReLU
            rep.append(SeparableConv2d(inter_channels, out_channels, 3, 1, norm_layer=norm_layer))  # 深度可分离卷积：inter→out（【注意】同样未传 dilation）
            rep.append(norm_layer(out_channels))  # BN(out)
        self.rep = nn.Sequential(*rep)  # 打包主分支

    def forward(self, x):  # 前向：主分支 + skip
        out = self.rep(x)  # 主分支输出
        if self.skip is not None:  # 有投影 skip
            skip = self.skipbn(self.skip(x))  # 1x1 卷积 + BN
        else:  # 否则
            skip = x  # 恒等 skip
        out = out + skip  # 残差相加
        return out  # 返回结果


class Enc(nn.Module):  # DFANet 的编码 stage：1 个 stride=2 的 BlockA + (blocks-1) 个 stride=1 的 BlockA
    # 参数说明：in_channels/out_channels——输入/输出通道数；blocks——BlockA 个数；norm_layer——归一化层构造函数
    def __init__(self, in_channels, out_channels, blocks, norm_layer=None):  # 初始化
        super(Enc, self).__init__()  # 调用父类初始化
        block = list()  # 层列表
        block.append(BlockA(in_channels, out_channels, 2, norm_layer=norm_layer))  # 首个 BlockA：stride=2 下采样，通道 in→out
        for i in range(blocks - 1):  # 其余 blocks-1 个
            block.append(BlockA(out_channels, out_channels, 1, norm_layer=norm_layer))  # stride=1，通道保持 out
        self.block = nn.Sequential(*block)  # 打包为 Sequential

    def forward(self, x):  # 前向传播
        return self.block(x)  # 直接过 Sequential：(N,C_in,H,W)→(N,C_out,H/2,W/2)


class FCAttention(nn.Module):  # DFANet 的全连接注意力模块（通道注意力，用全局向量调制特征图）
    # 参数说明：in_channels——输入通道数；norm_layer——归一化层构造函数
    def __init__(self, in_channels, norm_layer=None):  # 初始化
        super(FCAttention, self).__init__()  # 调用父类初始化
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化到 1x1（收集全局上下文）
        self.fc = nn.Linear(in_channels, 1000)  # 全连接：in→1000（【注意】隐层维度硬编码为 1000）
        self.conv = nn.Sequential(  # 注意力分支尾部的 1x1 卷积组
            nn.Conv2d(1000, in_channels, 1, bias=False),  # 1x1 卷积：1000→in_channels
            norm_layer(in_channels),  # BN(in_channels)
            nn.ReLU(True))  # ReLU

    def forward(self, x):  # 前向：输入 (N,C,H,W)
        n, c, _, _ = x.size()  # 取 batch 数 n 与通道数 c（忽略 H/W）
        att = self.avgpool(x).view(n, c)  # 全局池化并展平：(N,C,H,W)→(N,C,1,1)→(N,C)
        att = self.fc(att).view(n, 1000, 1, 1)  # 全连接升维并还原为 4D：(N,C)→(N,1000)→(N,1000,1,1)
        att = self.conv(att)  # 1x1 卷积组：(N,1000,1,1)→(N,C,1,1)
        return x * att.expand_as(x)  # 广播逐通道加权：注意力扩展到 (N,C,H,W) 后与原特征相乘


class XceptionA(nn.Module):  # DFANet 使用的轻量骨干 Xception-A（三级 Enc 编码 + FCAttention）
    # 参数说明：num_classes——分类头类别数；norm_layer——归一化层构造函数
    def __init__(self, num_classes=1000, norm_layer=nn.BatchNorm2d):  # 初始化（无 output_stride 参数，最终特征固定为输入的 1/16）
        super(XceptionA, self).__init__()  # 调用父类初始化
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1, bias=False),  # stem：3x3 标准卷积，3→8，stride=2
                                   norm_layer(8),  # BN(8)
                                   nn.ReLU(True))  # ReLU

        self.enc2 = Enc(8, 48, 4, norm_layer=norm_layer)  # 编码 stage2：8→48，4 个 BlockA，空间降至 1/4
        self.enc3 = Enc(48, 96, 6, norm_layer=norm_layer)  # 编码 stage3：48→96，6 个 BlockA，空间降至 1/8
        self.enc4 = Enc(96, 192, 4, norm_layer=norm_layer)  # 编码 stage4：96→192，4 个 BlockA，空间降至 1/16

        self.fca = FCAttention(192, norm_layer=norm_layer)  # 对 192 通道特征做全局通道注意力
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化（分类头用）
        self.fc = nn.Linear(192, num_classes)  # 分类全连接 192→num_classes

    def forward(self, x):  # 前向：输入 (N,3,H,W)
        x = self.conv1(x)  # stem：(N,3,H,W)→(N,8,H/2,W/2)

        x = self.enc2(x)  # stage2：(N,8,H/2,W/2)→(N,48,H/4,W/4)
        x = self.enc3(x)  # stage3：(N,48,H/4,W/4)→(N,96,H/8,W/8)
        x = self.enc4(x)  # stage4：(N,96,H/8,W/8)→(N,192,H/16,W/16)
        x = self.fca(x)  # 通道注意力加权：形状不变 (N,192,H/16,W/16)

        x = self.avgpool(x)  # 全局池化：(N,192,H/16,W/16)→(N,192,1,1)
        x = x.view(x.size(0), -1)  # 展平：(N,192,1,1)→(N,192)
        x = self.fc(x)  # 分类：(N,192)→(N,num_classes)

        return x  # 返回 logits


# Constructor
# 工厂函数：构建各 Xception 变体，并按需加载 ImageNet 预训练权重
def get_xception(pretrained=False, root='~/.torch/models', **kwargs):  # 构建 Xception65；pretrained——是否加载预训练权重；root——权重缓存目录；**kwargs——透传给模型构造函数
    model = Xception65(**kwargs)  # 实例化 Xception65
    if pretrained:  # 需要加载预训练权重
        from ..model_store import get_model_file  # 局部导入：解析（必要时下载）本地权重文件路径
        model.load_state_dict(torch.load(get_model_file('xception', root=root)))  # 加载 xception 权重；【注意】torch.load 未指定 map_location，且新版 PyTorch（≥2.6）默认 weights_only=True 可能导致加载失败
    return model  # 返回模型


def get_xception_71(pretrained=False, root='~/.torch/models', **kwargs):  # 构建 Xception71（参数含义同 get_xception）
    model = Xception71(**kwargs)  # 实例化 Xception71
    if pretrained:  # 需要加载预训练权重
        from ..model_store import get_model_file  # 局部导入权重路径工具
        model.load_state_dict(torch.load(get_model_file('xception71', root=root)))  # 加载 xception71 权重（【注意】同上 torch.load 隐患）
    return model  # 返回模型


def get_xception_a(pretrained=False, root='~/.torch/models', **kwargs):  # 构建 DFANet 用的轻量 XceptionA（参数含义同 get_xception）
    model = XceptionA(**kwargs)  # 实例化 XceptionA
    if pretrained:  # 需要加载预训练权重
        from ..model_store import get_model_file  # 局部导入权重路径工具
        model.load_state_dict(torch.load(get_model_file('xception_a', root=root)))  # 加载 xception_a 权重（【注意】同上 torch.load 隐患）
    return model  # 返回模型


if __name__ == '__main__':  # 直接运行本文件时的 smoke test（只构建模型，不训练）
    model = get_xception_a()  # 构建轻量 XceptionA 验证网络结构（不加载预训练权重）
