# ============================================================================
# 【文件说明】hrnet.py —— HRNet（High-Resolution Network，高分辨率网络）骨干实现
#     包含 BasicBlock / Bottleneck 两种残差块、HighResolutionModule（HRNet 核心：
#     多分支并行卷积 + 多尺度融合的基本单元）以及顶层 HighResolutionNet
#     （带 ImageNet 分类头的完整网络）。
# 【核心思想】与传统"编码器逐级下采样、解码器再上采样恢复"的串行结构不同，
#     HRNet 从 stage2 起并行维护多条分辨率分支：最高分辨率分支（1/4）全程
#     保持不下降，每个 stage 新增一条更低的分辨率分支；stage 内各分支独立卷积，
#     stage 末尾经 fuse_layers 融合——低分辨率→高分辨率用"1×1 卷积 + BN +
#     最近邻上采样（放大 2^(j-i) 倍）"对齐，高分辨率→低分辨率用若干个
#     "步长 2 的 3×3 卷积 + BN"串联下采样对齐，随后逐元素相加
#     （fuse_method='sum'）再 ReLU。由此高分辨率表征始终保留并持续被强语义
#     增强，特别适合分割等密集预测任务。
# 【工程位置】core/models/base_models/ 下的骨干网络文件之一；注意本文件是
#     带分类头的独立实现（工程内分割用 HRNet 在 core/models/hrnet.py 中另有
#     完整实现、并未复用本文件），且本文件存在若干历史遗留 bug（transition2/3
#     被 transition1 覆盖、F 未导入、调试 print 残留），详见行内【注意】标注。
# ============================================================================
import torch  # PyTorch 主库：张量与随机数（__main__ 自测构造输入用）
import torch.nn as nn  # 神经网络子模块：Conv2d/BatchNorm2d/ReLU/Sequential/ModuleList/Upsample 等


# 基础残差块：两层 3×3 卷积 + 跳跃连接；HRNet 各分辨率分支的构件（expansion=1，输出通道不扩张）
class BasicBlock(nn.Module):
    expansion = 1  # 通道扩张倍数：块输出通道数 = planes × expansion（BasicBlock 不扩张）

    # 参数：inplanes 输入通道数；planes 块内卷积通道数；stride 步长（2 时空间减半）；
    # downsample 恒等分支的下采样投影（None 表示直连）；norm_layer 归一化层构造器
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()  # 初始化 nn.Module 基类（完成子模块注册）
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1, bias=False)  # 第一个 3×3 卷积（stride 可为 2），后接 BN 故关闭 bias
        self.bn1 = norm_layer(planes)  # conv1 之后的归一化层
        self.relu = nn.ReLU(True)  # 原位 ReLU 激活
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)  # 第二个 3×3 卷积，步长固定 1（空间不变）
        self.bn2 = norm_layer(planes)  # conv2 之后的归一化层
        self.downsample = downsample  # 保存跳跃连接投影分支（可能为 None）
        self.stride = stride  # 记录步长（仅作属性保存，forward 中未使用）

    # 前向传播：输入 (N, inplanes, H, W) → 输出 (N, planes*expansion, H/stride, W/stride)
    def forward(self, x):
        identity = x  # 暂存输入，作为残差相加的恒等分支

        out = self.conv1(x)  # (N, inplanes, H, W) → (N, planes, H/stride, W/stride)
        out = self.bn1(out)  # 归一化，形状不变
        out = self.relu(out)  # 激活，形状不变

        out = self.conv2(out)  # (N, planes, H', W') → (N, planes, H', W')，空间不变
        out = self.bn2(out)  # 归一化，形状不变

        if self.downsample is not None:  # stride≠1 或通道不匹配时，恒等分支需投影对齐
            identity = self.downsample(x)  # (N, inplanes, H, W) → (N, planes*expansion, H/stride, W/stride)

        out += identity  # 残差相加：主分支与对齐后的恒等分支逐元素相加
        out = self.relu(out)  # 相加后再激活，形状不变

        return out  # 返回残差块输出


# 瓶颈残差块：1×1 降维 → 3×3 卷积 → 1×1 升维（×4）+ 跳跃连接；用于 HRNet 的 layer1（stem 后的第一个 stage）
class Bottleneck(nn.Module):
    expansion = 4  # 通道扩张倍数：块最终输出通道数 = planes × 4

    # 参数：inplanes 输入通道；planes 瓶颈中间通道；stride 步长；downsample 投影分支；norm_layer 归一化层
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()  # 初始化 nn.Module 基类
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)  # 1×1 卷积降维：(N, inplanes, H, W) → (N, planes, H, W)
        self.bn1 = norm_layer(planes)  # conv1 后归一化
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)  # 3×3 卷积（stride 下采样、padding=1）：→ (N, planes, H/stride, W/stride)
        self.bn2 = norm_layer(planes)  # conv2 后归一化
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)  # 1×1 卷积升维 4 倍：→ (N, planes*4, H', W')
        self.bn3 = norm_layer(planes * self.expansion)  # conv3 后归一化（通道数 planes*4）
        self.relu = nn.ReLU(True)  # 原位 ReLU 激活
        self.downsample = downsample  # 跳跃连接投影分支（可能为 None）
        self.stride = stride  # 记录步长（仅属性保存）

    # 前向传播：输入 (N, inplanes, H, W) → 输出 (N, planes*4, H/stride, W/stride)
    def forward(self, x):
        identity = x  # 暂存输入作为恒等分支

        out = self.conv1(x)  # 1×1 降维：(N, inplanes, H, W) → (N, planes, H, W)
        out = self.bn1(out)  # 归一化，形状不变
        out = self.relu(out)  # 激活，形状不变

        out = self.conv2(out)  # 3×3 卷积：→ (N, planes, H/stride, W/stride)
        out = self.bn2(out)  # 归一化，形状不变
        out = self.relu(out)  # 激活，形状不变

        out = self.conv3(out)  # 1×1 升维：→ (N, planes*4, H', W')
        out = self.bn3(out)  # 归一化，形状不变

        if self.downsample is not None:  # 需要时对恒等分支投影
            identity = self.downsample(x)  # (N, inplanes, H, W) → (N, planes*4, H/stride, W/stride)

        out += identity  # 残差相加
        out = self.relu(out)  # 相加后激活，形状不变

        return out  # 返回瓶颈块输出


# HRNet 核心模块：同一 stage 内并行 num_branches 条分辨率分支卷积，末端经 fuse_layers 做多尺度融合
class HighResolutionModule(nn.Module):
    # 参数：num_branches 分支数；blocks 残差块类型；num_blocks 各分支块数列表；num_inchannels 各分支输入通道列表；
    # num_channels 各分支基础通道列表；fuse_method 融合方式（'sum'）；multi_scale_output 是否输出所有分辨率；norm_layer 归一化层
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels,
                 fuse_method, multi_scale_output=True, norm_layer=nn.BatchNorm2d):  # 构造函数参数续行：融合方式 / 多尺度输出开关 / 归一化层
        super(HighResolutionModule, self).__init__()  # 初始化 nn.Module 基类
        assert num_branches == len(num_blocks)  # 校验：分支数与块数列表长度一致
        assert num_branches == len(num_channels)  # 校验：分支数与通道配置列表长度一致
        assert num_branches == len(num_inchannels)  # 校验：分支数与输入通道列表长度一致

        self.num_inchannels = num_inchannels  # 各分支输入通道（_make_one_branch 中会被原地更新为输出通道）
        self.fuse_method = fuse_method  # 融合方式（本实现仅使用 'sum'）
        self.num_branches = num_branches  # 分支数量
        self.multi_scale_output = multi_scale_output  # 是否输出全部分辨率（False 时仅输出最高分辨率分支）

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels, norm_layer=norm_layer)  # 构建并行分支（ModuleList）
        self.fuse_layers = self._make_fuse_layers(norm_layer)  # 构建跨分支融合层（上/下采样 + 通道适配）
        self.relu = nn.ReLU(True)  # 融合求和之后使用的原位激活

    # 构建单条分支：branch_index 分支序号；block 块类型；num_blocks 各分支块数；num_channels 各分支通道；
    # stride 首块步长（分支内保持分辨率不变，本实现恒为 1）；norm_layer 归一化层
    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1, norm_layer=nn.BatchNorm2d):  # 参数续行：首块步长（分支内恒为 1）与归一化层
        downsample = None  # 投影分支初始为 None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:  # 首块输入通道与目标输出不一致（或需下采样）时
            downsample = nn.Sequential(  # 构造恒等分支投影：1×1 卷积 + BN
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,  # 1×1 卷积对齐通道数（并按 stride 下采样）
                          1, stride, bias=False),  # Conv2d 参数续行：kernel=1、步长、无 bias，调用闭合
                norm_layer(num_channels[branch_index] * block.expansion))  # 投影后归一化

        layers = list()  # 收集该分支的所有残差块
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index],  # 首个块：带 stride 与投影分支
                            stride, downsample, norm_layer=norm_layer))  # 首块参数续行：步长、投影捷径、归一化层；块构造与 append 闭合
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion  # 原地更新该分支输出通道，供后续块使用
        for i in range(1, num_blocks[branch_index]):  # 其余块：stride=1、无投影、通道稳定
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], norm_layer=norm_layer))  # 追加标准残差块

        return nn.Sequential(*layers)  # 串联为该分支网络

    # 构建全部并行分支：分支 i 的分辨率为输入的 1/2^i（分支间互不下采样）
    def _make_branches(self, num_branches, block, num_blocks, num_channels, norm_layer=nn.BatchNorm2d):
        branches = list()  # 分支收集列表
        for i in range(num_branches):  # 分辨率从高到低逐条构建
            branches.append(  # 收集第 i 条分支（_make_one_branch 调用开始，续行）
                self._make_one_branch(i, block, num_blocks, num_channels, norm_layer=norm_layer))  # 构建第 i 条分支并加入

        return nn.ModuleList(branches)  # ModuleList 包装，便于按索引调用

    # 构建融合层：fuse_layers[i][j] 负责把来源分支 j 的特征适配（上/下采样 + 通道对齐）到目标分支 i
    def _make_fuse_layers(self, norm_layer=nn.BatchNorm2d):
        if self.num_branches == 1:  # 单分支时无需融合
            return None  # 直接返回 None

        num_branches = self.num_branches  # 局部引用：分支数
        num_inchannels = self.num_inchannels  # 局部引用：各分支输出通道
        fuse_layers = []  # 外层列表：按目标分支 i 索引
        for i in range(num_branches if self.multi_scale_output else 1):  # multi_scale_output=False 时只构建 i=0（最高分辨率）的融合结果
            fuse_layer = list()  # 内层列表：按来源分支 j 索引
            for j in range(num_branches):  # 遍历全部来源分支
                if j > i:  # 来源 j 分辨率低于目标 i：需上采样对齐（低分辨率→高分辨率）
                    fuse_layer.append(nn.Sequential(  # 1×1 卷积 + BN + 最近邻上采样
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, bias=False),  # 1×1 卷积：通道对齐到目标分支 i
                        norm_layer(num_inchannels[i]),  # 归一化
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))  # 最近邻上采样 2^(j-i) 倍：空间对齐到分支 i
                elif j == i:  # 来源与目标为同一分支：无需变换
                    fuse_layer.append(None)  # 占位 None（forward 中直连）
                else:  # 来源 j 分辨率高于目标 i：需下采样对齐（高分辨率→低分辨率）
                    conv3x3s = list()  # 收集串联的下采样卷积
                    for k in range(i - j):  # 分辨率相差 (i-j) 级 → 串联 (i-j) 个步长 2 的 3×3 卷积
                        if k == i - j - 1:  # 最后一级
                            num_outchannels_conv3x3 = num_inchannels[i]  # 输出通道对齐到目标分支 i
                            conv3x3s.append(nn.Sequential(  # 最后一级下采样：append 步长 2 的 3×3 卷积容器（续行）
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),  # 步长 2 的 3×3 卷积（padding=1），空间减半
                                norm_layer(num_outchannels_conv3x3)))  # 最后一级不带 ReLU（相加后再统一激活）
                        else:  # 中间级
                            num_outchannels_conv3x3 = num_inchannels[j]  # 通道保持来源分支 j 的通道数
                            conv3x3s.append(nn.Sequential(  # 中间级下采样：append 步长 2 的 3×3 卷积容器（续行）
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),  # 步长 2 的 3×3 卷积，空间减半
                                norm_layer(num_outchannels_conv3x3),  # 归一化
                                nn.ReLU(False)))  # 中间级带非原位 ReLU
                    fuse_layer.append(nn.Sequential(*conv3x3s))  # 串联为"高→低分辨率"的下采样适配路径
            fuse_layers.append(nn.ModuleList(fuse_layer))  # 汇入目标分支 i 的一组来源适配层

        return nn.ModuleList(fuse_layers)  # 返回二维 ModuleList：fuse_layers[i][j]

    # 查询接口：返回各分支输出通道数列表（供下一个 stage 构建 transition 时使用）
    def get_num_inchannels(self):
        return self.num_inchannels  # 已被 _make_one_branch 原地更新为输出通道

    # 前向传播：x 为各分支特征列表 [x_0(1/4), x_1(1/8), ...]；返回同结构的多尺度融合特征列表
    def forward(self, x):
        if self.num_branches == 1:  # 单分支特例
            return [self.branches[0](x[0])]  # 仅过分支卷积，包成列表返回

        for i in range(self.num_branches):  # 各分支先独立卷积（并行、分辨率互不变换）
            x[i] = self.branches[i](x[i])  # x[i]: (N, C_i, H_i, W_i)，仅首块可能改变通道

        x_fuse = list()  # 收集各目标分支的融合结果
        for i in range(len(self.fuse_layers)):  # 遍历目标分支 i
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])  # 初值取来源分支 0：目标 0 直连，其余经下采样适配
            for j in range(1, self.num_branches):  # 累加其余来源分支 j=1..n-1
                if i == j:  # 来源即目标分支
                    y = y + x[j]  # 同形状直连相加
                else:  # 来源≠目标分支
                    y = y + self.fuse_layers[i][j](x[j])  # 经融合层对齐到 (N, C_i, H_i, W_i) 后相加
            x_fuse.append(self.relu(y))  # 求和后过 ReLU，得到该分辨率的融合输出

        return x_fuse  # 返回多分辨率融合特征（高分辨率表征全程保持并持续增强）


# 顶层网络：deep stem → layer1 → 逐级加分支的 stage2/3/4（分支数 2→3→4，每级前有 transition 层）→ 分类头
class HighResolutionNet(nn.Module):
    # 参数：blocks 各 stage 残差块类型列表；num_channels 各 stage 各分支基础通道配置；num_modules 各 stage 的模块数；
    # num_branches 各 stage 分支数；num_blocks 各 stage 各分支块数；fuse_method 各 stage 融合方式；
    # norm_layer 归一化层；**kwargs 兼容上层传入的额外参数
    def __init__(self, blocks, num_channels, num_modules, num_branches, num_blocks,
                 fuse_method, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数参数续行：融合方式列表、归一化层、兼容额外关键字参数
        super(HighResolutionNet, self).__init__()  # 初始化 nn.Module 基类
        self.num_branches = num_branches  # 保存各 stage 分支数列表（forward 遍历 transition 时使用）

        # deep stem
        self.conv1 = nn.Sequential(  # 深层茎：两个"3×3 卷积(stride=2)+BN+ReLU"，整体下采样 4 倍
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),  # (N,3,H,W) → (N,64,H/2,W/2)
            norm_layer(64),  # 归一化
            nn.ReLU(True),  # 原位激活
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),  # (N,64,H/2,W/2) → (N,64,H/4,W/4)
            norm_layer(64),  # 归一化
            nn.ReLU(True))  # 原位激活

        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4, norm_layer=norm_layer)  # stage1：4 个 Bottleneck，(N,64,H/4,W/4) → (N,256,H/4,W/4)

        # stage 2
        num_channel, block = num_channels[0], blocks[0]  # 取 stage2 的通道配置与块类型
        channels = [channel * block.expansion for channel in num_channel]  # 乘扩张倍数得到各分支实际通道
        self.transition1 = self._make_transition_layer([256], channels, norm_layer)  # 由 stage1 输出（256 通道）生成 stage2 两分支（新分支经 3×3 stride2 下采样产生）
        self.stage2, pre_stage_channels = self._make_stage(num_modules[0], num_branches[0],  # 构建 stage2（2 分支）并返回其输出通道
                                                           num_blocks[0], channels, block,  # _make_stage 参数续行：stage2 各分支块数、通道、块类型
                                                           fuse_method[0], channels,  # _make_stage 参数续行：stage2 融合方式、输入通道（与输出一致）
                                                           norm_layer=norm_layer)  # _make_stage 参数续行：stage2 归一化层；调用闭合

        # stage 3
        num_channel, block = num_channels[1], blocks[1]  # 取 stage3 的通道配置与块类型
        channels = [channel * block.expansion for channel in num_channel]  # 各分支实际通道
        # 【注意】严重 bug：此处应为 self.transition2，却重复赋给 self.transition1，
        # 会覆盖 stage2 的过渡层，且 forward 中引用的 self.transition2 因未定义将抛 AttributeError
        self.transition1 = self._make_transition_layer(pre_stage_channels, channels, norm_layer)  # 由 stage2 输出生成 stage3 三分支
        self.stage3, pre_stage_channels = self._make_stage(num_modules[1], num_branches[1],  # 构建 stage3（3 分支）
                                                           num_blocks[1], channels, block,  # _make_stage 参数续行：stage3 各分支块数、通道、块类型
                                                           fuse_method[1], channels,  # _make_stage 参数续行：stage3 融合方式、输入通道
                                                           norm_layer=norm_layer)  # _make_stage 参数续行：stage3 归一化层；调用闭合

        # stage 4
        num_channel, block = num_channels[2], blocks[2]  # 取 stage4 的通道配置与块类型
        channels = [channel * block.expansion for channel in num_channel]  # 各分支实际通道
        # 【注意】严重 bug：此处应为 self.transition3，同样误赋给 self.transition1（问题同上一处）
        self.transition1 = self._make_transition_layer(pre_stage_channels, channels, norm_layer)  # 由 stage3 输出生成 stage4 四分支
        self.stage4, pre_stage_channels = self._make_stage(num_modules[2], num_branches[2],  # 构建 stage4（4 分支）
                                                           num_blocks[2], channels, block,  # _make_stage 参数续行：stage4 各分支块数、通道、块类型
                                                           fuse_method[2], channels,  # _make_stage 参数续行：stage4 融合方式、输入通道
                                                           norm_layer=norm_layer)  # _make_stage 参数续行：stage4 归一化层；调用闭合

        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(pre_stage_channels, norm_layer)  # 分类头：各分支升维 + 级联下采样聚合 + 1×1 升到 2048

        self.classifier = nn.Linear(2048, 1000)  # ImageNet 1000 类分类层：(N,2048) → (N,1000)

    # 构建普通残差层（用于 layer1 与分类头）：block 块类型；inplanes 输入通道；planes 中间通道；blocks 块数；stride 步长
    def _make_layer(self, block, inplanes, planes, blocks, stride=1, norm_layer=nn.BatchNorm2d):
        downsample = None  # 投影分支初始为 None
        if stride != 1 or inplanes != planes * block.expansion:  # 尺寸/通道不匹配时构建投影
            downsample = nn.Sequential(  # 恒等分支投影：1×1 卷积 + BN
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride, bias=False),  # 1×1 卷积对齐通道（并按 stride 下采样）
                norm_layer(planes * block.expansion))  # 投影后归一化

        layers = list()  # 收集残差块
        layers.append(block(inplanes, planes, stride, downsample=downsample, norm_layer=norm_layer))  # 首块（带投影）
        inplanes = planes * block.expansion  # 更新后续块的输入通道
        for _ in range(1, blocks):  # 其余块
            layers.append(block(inplanes, planes, norm_layer=norm_layer))  # 标准块（stride=1、无投影）

        return nn.Sequential(*layers)  # 串联返回

    # 构建阶段间过渡层：num_channels_pre_layer 上一 stage 各分支通道；num_channels_cur_layer 当前 stage 各分支通道
    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer, norm_layer=nn.BatchNorm2d):
        num_branches_cur = len(num_channels_cur_layer)  # 当前 stage 分支数
        num_branches_pre = len(num_channels_pre_layer)  # 上一 stage 分支数

        transition_layers = list()  # 收集每条新分支的过渡层
        for i in range(num_branches_cur):  # 遍历当前 stage 的每条分支
            if i < num_branches_pre:  # 与上一 stage 重叠的（高分辨率侧）分支
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:  # 通道数不一致时
                    transition_layers.append(nn.Sequential(  # 3×3 卷积(stride=1) + BN + ReLU 适配通道
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, padding=1, bias=False),  # 改通道，空间不变
                        norm_layer(num_channels_cur_layer[i]),  # 归一化
                        nn.ReLU(True)))  # 原位激活
                else:  # 通道一致
                    transition_layers.append(None)  # 无需过渡，直连（占位 None）
            else:  # 当前 stage 新增的（低分辨率）分支：由上一 stage 最低分辨率分支下采样生成
                conv3x3s = list()  # 收集串联的下采样卷积
                for j in range(i + 1 - num_branches_pre):  # 需要下采样的级数
                    in_channels = num_channels_pre_layer[-1]  # 每级输入都取上一 stage 最后一条（最低分辨率）分支的通道
                    out_channels = num_channels_cur_layer[i] if j == i - num_branches_pre else in_channels  # 最后一级对齐到新分支通道，中间级保持不变
                    conv3x3s.append(nn.Sequential(  # 3×3(stride=2) + BN + ReLU
                        nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),  # 空间减半
                        norm_layer(out_channels),  # 归一化
                        nn.ReLU(True)))  # 原位激活
                transition_layers.append(nn.Sequential(*conv3x3s))  # 串联为新分支的下采样过渡路径

        return nn.ModuleList(transition_layers)  # ModuleList 便于按分支索引

    # 构建一个 stage（可含多个 HighResolutionModule）：num_modules 模块数；num_branches 分支数；num_blocks 各分支块数；
    # num_channels 各分支通道；block 块类型；fuse_method 融合方式；num_inchannels 输入通道；multi_scale_output 末模块是否多尺度输出
    def _make_stage(self, num_modules, num_branches, num_blocks, num_channels, block,
                    fuse_method, num_inchannels, multi_scale_output=True, norm_layer=nn.BatchNorm2d):  # 参数续行：融合方式、输入通道、多尺度输出开关、归一化层
        modules = list()  # 收集该 stage 的所有模块
        for i in range(num_modules):  # 逐个构建
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:  # 只有最后一个模块允许关闭多尺度输出
                reset_multi_scale_output = False  # 末模块仅输出最高分辨率分支
            else:  # 其余情况（非末模块或要求多尺度输出）：保持全分辨率输出
                reset_multi_scale_output = True  # 其余模块必须输出全部分辨率（后续模块需要多尺度输入）

            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels,  # 构建高分辨率模块并加入
                                                fuse_method, reset_multi_scale_output, norm_layer=norm_layer))  # 模块参数续行：融合方式 / 多尺度开关 / 归一化层；构造与 append 闭合
            num_inchannels = modules[-1].get_num_inchannels()  # 更新为上一模块的输出通道

        return nn.Sequential(*modules), num_inchannels  # 返回串联后的 stage 及其输出通道列表

    # 构建分类头：返回 incre_modules（各分辨率升维）、downsamp_modules（相邻分辨率间的下采样）、final_layer（末端 1×1 升到 2048）
    def _make_head(self, pre_stage_channels, norm_layer=nn.BatchNorm2d):
        head_block = Bottleneck  # 分类头统一使用 Bottleneck 升维
        head_channels = [32, 64, 128, 256]  # 各分辨率升维基础通道（×4 后为 128/256/512/1024）

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = list()  # 收集各分辨率的升维模块
        for i, channels in enumerate(pre_stage_channels):  # 遍历 stage4 各分支
            incre_module = self._make_layer(head_block, channels, head_channels[i], 1)  # 每分支 1 个 Bottleneck 升维到 128/256/512/1024
            incre_modules.append(incre_module)  # 加入列表
        incre_modules = nn.ModuleList(incre_modules)  # 包装为 ModuleList

        # downsampling modules
        downsamp_modules = []  # 收集相邻分辨率间的下采样模块
        for i in range(len(pre_stage_channels) - 1):  # 4 条分支共 3 个下采样模块
            in_channels = head_channels[i] * head_block.expansion  # 输入通道（升维后）
            out_channels = head_channels[i + 1] * head_block.expansion  # 输出通道（下一级）

            downsamp_module = nn.Sequential(  # 3×3(stride=2) + BN + ReLU
                nn.Conv2d(in_channels, out_channels, 3, 2, 1),  # 下采样卷积【注意】此处 bias=True（默认值），与主干其他卷积（bias=False）风格不一致
                norm_layer(out_channels),  # 归一化
                nn.ReLU(True))  # 原位激活

            downsamp_modules.append(downsamp_module)  # 加入列表
        downsamp_modules = nn.ModuleList(downsamp_modules)  # 包装为 ModuleList

        final_layer = nn.Sequential(  # 末端 1×1 卷积：把最低分辨率分支通道升至 2048
            nn.Conv2d(head_channels[3] * head_block.expansion, 2048, 1),  # (N,1024,H/32,W/32) → (N,2048,H/32,W/32)（bias=True）
            norm_layer(2048),  # 归一化
            nn.ReLU(True))  # 原位激活

        return incre_modules, downsamp_modules, final_layer  # 返回三个组件

    # 前向传播：x (N,3,H,W) → 分类 logits (N,1000)
    def forward(self, x):
        x = self.conv1(x)  # deep stem：(N,3,H,W) → (N,64,H/4,W/4)
        x = self.layer1(x)  # stage1：(N,64,H/4,W/4) → (N,256,H/4,W/4)

        x_list = list()  # 收集 stage2 各分支输入
        for i in range(self.num_branches[0]):  # 遍历 stage2 的 2 条分支
            if self.transition1[i] is not None:  # 需要过渡层适配时
                tmp = self.transition1[i](x)  # 【注意】调试用的一次性前向：结果仅用于打印，属重复计算
                print(tmp.size())  # 【注意】遗留的调试打印语句，正式代码应删除
                x_list.append(self.transition1[i](x))  # 再次前向并保存（同一输入被计算了两遍，浪费算力）
            else:  # 无需过渡层
                x_list.append(x)  # stage1 特征直通（保持 1/4 分辨率）
        y_list = self.stage2(x_list)  # stage2 融合输出：[(N,C,H/4,W/4), (N,C',H/8,W/8)]

        x_list = []  # 收集 stage3 各分支输入
        for i in range(self.num_branches[1]):  # 遍历 stage3 的 3 条分支
            if self.transition2[i] is not None:  # 【注意】bug：self.transition2 从未定义（__init__ 中被误命名为 transition1），运行到此将抛 AttributeError
                x_list.append(self.transition2[i](y_list[-1]))  # 由上一 stage 最低分辨率分支（y_list[-1]）经过渡层生成新分支
            else:  # 与上一 stage 重叠的分支
                x_list.append(y_list[i])  # 直通 stage2 对应分支输出
        y_list = self.stage3(x_list)  # stage3 融合输出：3 个分辨率 [1/4, 1/8, 1/16] 的特征列表

        x_list = []  # 收集 stage4 各分支输入
        for i in range(self.num_branches[2]):  # 遍历 stage4 的 4 条分支
            if self.transition3[i] is not None:  # 【注意】bug：self.transition3 同样未定义（与 transition2 问题相同）
                x_list.append(self.transition3[i](y_list[-1]))  # 由 stage3 最低分辨率分支下采样生成新分支
            else:  # 与上一 stage 重叠的分支
                x_list.append(y_list[i])  # 直通 stage3 对应分支输出
        y_list = self.stage4(x_list)  # stage4 融合输出：[1/4, 1/8, 1/16, 1/32] 四个分辨率的特征列表

        # Classification Head
        y = self.incre_modules[0](y_list[0])  # 最高分辨率分支升维：(N,C0,H/4,W/4) → (N,128,H/4,W/4)
        for i in range(len(self.downsamp_modules)):  # 自上而下逐级融合更低分辨率分支
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)  # 下一分支升维结果 + 上一级结果下采样后相加（空间逐级减半、通道逐级加倍）

        y = self.final_layer(y)  # (N,1024,H/32,W/32) → (N,2048,H/32,W/32)

        y = F.avg_pool2d(y, kernel_size=y.size()  # 全局平均池化到 1×1【注意】F（torch.nn.functional）未导入，运行到此将 NameError
[2:]).view(y.size(0), -1)  # kernel_size 取空间尺寸 y.size()[2:]；池化后展平为 (N,2048)

        y = self.classifier(y)  # 全连接分类：(N,2048) → (N,1000)

        return y  # 返回分类 logits


# ===== 模块级默认网络配置（自定义宽度版 HRNet：BasicBlock、stage 分支数 2/3/4；注意 stage2 通道 [256,256] 比官方 HRNet18 的 [32,64] 更宽）=====
blocks = [BasicBlock, BasicBlock, BasicBlock]  # 各 stage 使用的残差块类型
num_modules = [1, 1, 1]  # 各 stage 的 HighResolutionModule 数量
num_branches = [2, 3, 4]  # 各 stage 分支数（逐级 2→3→4）
num_blocks = [[4, 4], [4, 4, 4], [4, 4, 4, 4]]  # 各 stage 每条分支的残差块数
num_channels = [[256, 256], [32, 64, 128], [32, 64, 128, 256]]  # 各 stage 每条分支的基础通道数（×expansion 后为实际通道）
fuse_method = ['sum', 'sum', 'sum']  # 各 stage 的融合方式：逐元素相加

if __name__ == '__main__':  # 直接运行本文件时做前向自测
    img = torch.randn(1, 3, 256, 256)  # 构造随机输入 (1,3,256,256)
    model = HighResolutionNet(blocks, num_channels, num_modules, num_branches, num_blocks, fuse_method)  # 用上述默认配置实例化 HRNet
    output = model(img)  # 前向传播（因上述 transition2/transition3、F 未导入等 bug，实际运行到此会报错）
