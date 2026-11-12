# ============================================================================
# 【文件说明】eespnet.py —— EESPNet（EESP: Extremely Efficient Spatial Pyramid，
#     极端高效空间金字塔）轻量级骨干网络：包含 EESP 单元、DownSampler 下采样
#     模块与 EESPNet 主网络三部分，并给出 eespnet() 工厂函数。
# 【核心思想】EESP 是 ESP 模块的"深度可分离 + 分组"升级版：先用分组 1×1 卷积
#     把特征分成 k 组（每分支通道 n=out/k），每组接一个膨胀率递增（核 3/5/7/9、
#     上限 r_lim）的 3×3 空洞深度卷积，各分支输出逐级累加（层级相连利于梯度流）
#     后拼接，再经 BN+PReLU 与分组 1×1 卷积融合——用极少参数获得多尺度感受野。
#     下采样由 DownSampler 完成（AvgPool 直通通道 + EESP 新增通道拼接），并可
#     融合原始输入图像的增强分支（reinf）；forward(seg=True) 时返回 4 个 level
#     的特征图列表，供分割头做多层融合。
# 【工程位置】core/models/base_models/ 下的骨干网络之一，上层轻量分割模型
#     通过 eespnet() 工厂函数实例化（预训练权重不支持）。
# ============================================================================
import math  # 数学工具：本文件用 math.ceil 把缩放后的通道数向上取整到 k 的倍数
import torch  # PyTorch 主库：张量与 torch.cat 等操作
import torch.nn as nn  # 神经网络模块：Conv2d/BatchNorm2d/PReLU/ModuleList/Linear 等
import torch.nn.functional as F  # 函数式接口：avg_pool2d / adaptive_avg_pool2d / dropout

from core.nn import _ConvBNPReLU, _ConvBN, _BNPReLU  # 公共层（core/nn/basic.py）：卷积+BN+PReLU、卷积+BN、BN+PReLU

__all__ = ['EESP', 'EESPNet', 'eespnet']  # 模块公开符号清单


# EESP 单元：对称分离卷积模块——分组 1×1 投影 + 并行多膨胀率空洞深度卷积金字塔 + 分组 1×1 融合
class EESP(nn.Module):

    # 参数：in_channels 输入通道；out_channels 输出通道（须能被 k 整除）；stride 步长（2 时下采样）；
    # k 并行分支数（分组数）；r_lim 最大卷积核尺寸上限；down_method 下采样方式（'avg'：配合外层平均池化 / 'esp'：仅靠深度卷积）；norm_layer 归一化层
    def __init__(self, in_channels, out_channels, stride=1, k=4, r_lim=7, down_method='esp', norm_layer=nn.BatchNorm2d):
        super(EESP, self).__init__()  # 初始化 nn.Module 基类
        self.stride = stride  # 记录步长（forward 中判断是否走 'avg' 下采样分支）
        n = int(out_channels / k)  # 每个分支（每组）的通道数
        n1 = out_channels - (k - 1) * n  # 末组通道数（仅用于下面的整除性校验）
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'  # 断言下采样方式合法（原文 suppported 拼写保留原样）
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)  # 断言 out_channels 能被 k 整除（深度卷积分组要求）
        self.proj_1x1 = _ConvBNPReLU(in_channels, n, 1, stride=1, groups=k, norm_layer=norm_layer)  # 分组 1×1 投影：k 组并行降维，(N,in,H,W) → (N,n,H,W)

        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}  # 卷积核尺寸 → 膨胀率的查表映射（核越大膨胀率越大，感受野越大）
        self.k_sizes = list()  # 各分支的卷积核尺寸列表
        for i in range(k):  # 为 k 个分支依次生成核尺寸
            ksize = int(3 + 2 * i)  # 第 i 个分支的核尺寸 = 3 + 2i（即 3、5、7、9…）
            ksize = ksize if ksize <= r_lim else 3  # 超过上限 r_lim 的分支退回核尺寸 3
            self.k_sizes.append(ksize)  # 记录该分支核尺寸
        self.k_sizes.sort()  # 升序排序，保证膨胀率小的分支在前
        self.spp_dw = nn.ModuleList()  # 并行空洞深度卷积集合（空间金字塔主体）
        for i in range(k):  # 为每个分支构造深度卷积
            dilation = map_receptive_ksize[self.k_sizes[i]]  # 由核尺寸查表得到该分支的膨胀率
            self.spp_dw.append(nn.Conv2d(n, n, 3, stride, dilation, dilation=dilation, groups=n, bias=False))  # 3×3 空洞深度卷积（groups=n 逐通道；dilation 扩大感受野；stride 可下采样）
        self.conv_1x1_exp = _ConvBN(out_channels, out_channels, 1, 1, groups=k, norm_layer=norm_layer)  # 分组 1×1 融合卷积（带 BN、无激活——线性瓶颈，避免低维信息损失）
        self.br_after_cat = _BNPReLU(out_channels, norm_layer)  # 分支拼接后的 BN + PReLU
        self.module_act = nn.PReLU(out_channels)  # 模块末端激活
        self.downAvg = True if down_method == 'avg' else False  # 是否为 'avg' 下采样模式（是则本模块返回未激活结果交由外层处理）

    # 前向传播：x (N,in,H,W) → (N,out,H/stride,W/stride)
    def forward(self, x):
        output1 = self.proj_1x1(x)  # 分组 1×1 投影：(N,in,H,W) → (N,n,H,W)
        output = [self.spp_dw[0](output1)]  # 分支 0：空洞深度卷积，(N,n,H,W) → (N,n,H/stride,W/stride)
        for k in range(1, len(self.spp_dw)):  # 其余分支：逐分支计算并与前一分支累加（层级相连，改善梯度传播）
            out_k = self.spp_dw[k](output1)  # 分支 k 的空洞深度卷积输出
            out_k = out_k + output[k - 1]  # 与前一分支结果逐元素相加
            output.append(out_k)  # 加入输出列表
        expanded = self.conv_1x1_exp(self.br_after_cat(torch.cat(output, 1)))  # 沿通道拼接 k 个分支 → (N,out,H',W') → BN+PReLU → 分组 1×1 融合
        del output  # 删除中间列表引用，尽早释放显存
        if self.stride == 2 and self.downAvg:  # 'avg' 下采样模式：本单元仅负责新增通道，激活与残差交给外层 DownSampler 统一处理
            return expanded  # 直接返回未激活结果

        if expanded.size() == x.size():  # 仅当形状完全一致（stride=1 且 in==out）时加残差
            expanded = expanded + x  # 逐元素残差相加

        return self.module_act(expanded)  # PReLU 激活后返回


# DownSampler：EESPNet 的下采样模块——AvgPool 直通通道 + EESP 新增通道拼接，可选融合原始输入图像增强分支
class DownSampler(nn.Module):

    # 参数：in_channels 输入通道；out_channels 输出通道（须 > in_channels）；k EESP 分支数；r_lim 核上限；
    # reinf 是否启用输入图像增强分支；inp_reinf 增强分支输入通道数（RGB=3）；norm_layer 归一化层（默认 None，实际调用处均传入 BatchNorm2d）
    def __init__(self, in_channels, out_channels, k=4, r_lim=9, reinf=True, inp_reinf=3, norm_layer=None):
        super(DownSampler, self).__init__()  # 初始化 nn.Module 基类
        channels_diff = out_channels - in_channels  # EESP 分支负责的通道数（其余 in_channels 通道由 AvgPool 直通提供）
        self.eesp = EESP(in_channels, channels_diff, stride=2, k=k,  # EESP 分支：stride=2、'avg' 模式，产出 (out-in) 个新通道
                         r_lim=r_lim, down_method='avg', norm_layer=norm_layer)  # EESP 构造参数续行：核上限 r_lim / 'avg' 下采样方式 / 归一化层
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)  # 平均池化分支：空间减半、通道保持 in_channels
        if reinf:  # 启用输入图像增强分支时
            self.inp_reinf = nn.Sequential(  # 原图增强路径：3×3 卷积+BN+PReLU → 1×1 卷积+BN
                _ConvBNPReLU(inp_reinf, inp_reinf, 3, 1, 1),  # 增强分支第一层：3×3 卷积+BN+PReLU（保持 3 通道）
                _ConvBN(inp_reinf, out_channels, 1, 1))  # 增强分支第二层：1×1 卷积+BN 升维到 out_channels；Sequential 定义结束
        self.act = nn.PReLU(out_channels)  # 拼接/融合后的末端激活

    # 前向传播：x 上一级特征 (N,in,H,W)；x2 原始输入图像 (N,3,H0,W0)（可选）→ 输出 (N,out,H/2,W/2)
    def forward(self, x, x2=None):
        avg_out = self.avg(x)  # 平均池化直通：(N,in,H,W) → (N,in,H/2,W/2)
        eesp_out = self.eesp(x)  # EESP 新增通道：(N,in,H,W) → (N,out-in,H/2,W/2)
        output = torch.cat([avg_out, eesp_out], 1)  # 通道维拼接：(N,out,H/2,W/2)
        if x2 is not None:  # 提供了原始输入图像时，融合增强分支
            w1 = avg_out.size(2)  # 目标空间尺寸
            while True:  # 反复对原图 3×3 平均池化下采样，直到空间尺寸与主分支一致【注意】若尺寸永远无法精确相等（如特殊奇偶尺寸），此循环无法退出
                x2 = F.avg_pool2d(x2, kernel_size=3, padding=1, stride=2)  # 原图空间减半：(N,3,H0,W0) → (N,3,H0/2,W0/2)
                w2 = x2.size(2)  # 当前空间尺寸
                if w2 == w1:  # 尺寸对齐
                    break  # 退出下采样循环
            output = output + self.inp_reinf(x2)  # 增强分支映射到 out 通道后与主输出相加

        return self.act(output)  # PReLU 激活后返回


# EESPNet 主网络：卷积茎(level1) + level2~5 金字塔（每级由 DownSampler 进入，再堆叠 EESP）
class EESPNet(nn.Module):
    # 参数：num_classes 分类数；scale 网络宽度缩放系数；reinf 是否启用输入图像增强分支；norm_layer 归一化层
    def __init__(self, num_classes=1000, scale=1, reinf=True, norm_layer=nn.BatchNorm2d):
        super(EESPNet, self).__init__()  # 初始化 nn.Module 基类
        inp_reinf = 3 if reinf else None  # 增强分支输入通道：启用时为 RGB 3 通道，否则 None
        reps = [0, 3, 7, 3]  # 各 level 的 EESP 重复数：level3=3、level4=7、level5=3（reps[0] 占位未用）
        r_lim = [13, 11, 9, 7, 5]  # 各 level 的核尺寸上限（越深上限越小）
        K = [4] * len(r_lim)  # 各 level EESP 的分支数 k 均为 4

        # set out_channels
        base, levels, base_s = 32, 5, 0  # 基础通道 32、共 5 个 level、base_s 为缩放后的实际基础通道
        out_channels = [base] * levels  # 初始化 5 个 level 的输出通道列表
        for i in range(levels):  # 逐 level 计算输出通道
            if i == 0:  # 第一个 level 的通道计算
                base_s = int(base * scale)  # 缩放后的基础通道
                base_s = math.ceil(base_s / K[0]) * K[0]  # 向上取整到 k=4 的倍数（保证可被分组卷积整除）
                out_channels[i] = base if base_s > base else base_s  # 取不超过基准 32 的值【注意】scale≤1 时首层通道恒为 32，仅 scale>1 时放大
            else:  # 其余 level 通道按 2 的幂次递增
                out_channels[i] = base_s * pow(2, i)  # level i 通道 = base_s × 2^i
        if scale <= 1.5:  # 小尺度网络末端通道取 1024
            out_channels.append(1024)  # 追加末端输出通道
        elif scale in [1.5, 2]:  # 【注意】可疑写法：scale=1.5 已被上一分支(scale<=1.5)拦截，此条件实际只对 scale=2 生效，1280 对 1.5 不可达
            out_channels.append(1280)  # 大尺度网络末端通道取 1280
        else:  # 其他缩放系数
            raise ValueError("Unknown scale value.")  # 视为非法配置，抛出异常

        self.level1 = _ConvBNPReLU(3, out_channels[0], 3, 2, 1, norm_layer=norm_layer)  # level1 卷积茎：(N,3,H,W) → (N,C0,H/2,W/2)

        self.level2_0 = DownSampler(out_channels[0], out_channels[1], k=K[0], r_lim=r_lim[0],  # level2 入口下采样：(N,C0,H/2,W/2) → (N,C1,H/4,W/4)
                                    reinf=reinf, inp_reinf=inp_reinf, norm_layer=norm_layer)  # DownSampler 参数续行：增强分支开关 / 增强分支通道 / 归一化层，调用闭合

        self.level3_0 = DownSampler(out_channels[1], out_channels[2], k=K[1], r_lim=r_lim[1],  # level3 入口下采样：(N,C1,H/4,W/4) → (N,C2,H/8,W/8)
                                    reinf=reinf, inp_reinf=inp_reinf, norm_layer=norm_layer)  # DownSampler 参数续行：增强分支开关 / 增强分支通道 / 归一化层，调用闭合
        self.level3 = nn.ModuleList()  # level3 的 EESP 堆叠容器
        for i in range(reps[1]):  # 重复 3 个 EESP
            self.level3.append(EESP(out_channels[2], out_channels[2], k=K[2], r_lim=r_lim[2],  # 等通道 EESP（stride=1，内部带残差）
                                    norm_layer=norm_layer))  # EESP 参数续行：归一化层；构造与 append 调用闭合

        self.level4_0 = DownSampler(out_channels[2], out_channels[3], k=K[2], r_lim=r_lim[2],  # level4 入口下采样：(N,C2,H/8,W/8) → (N,C3,H/16,W/16)
                                    reinf=reinf, inp_reinf=inp_reinf, norm_layer=norm_layer)  # DownSampler 参数续行：增强分支开关 / 增强分支通道 / 归一化层，调用闭合
        self.level4 = nn.ModuleList()  # level4 的 EESP 堆叠容器
        for i in range(reps[2]):  # 重复 7 个 EESP
            self.level4.append(EESP(out_channels[3], out_channels[3], k=K[3], r_lim=r_lim[3],  # 等通道 EESP（stride=1，内部带残差）
                                    norm_layer=norm_layer))  # EESP 参数续行：归一化层；构造与 append 调用闭合

        self.level5_0 = DownSampler(out_channels[3], out_channels[4], k=K[3], r_lim=r_lim[3],  # level5 入口下采样：(N,C3,H/16,W/16) → (N,C4,H/32,W/32)
                                    reinf=reinf, inp_reinf=inp_reinf, norm_layer=norm_layer)  # DownSampler 参数续行：增强分支开关 / 增强分支通道 / 归一化层，调用闭合
        self.level5 = nn.ModuleList()  # level5 的 EESP 堆叠容器（末尾还追加两个卷积）
        for i in range(reps[2]):  # 【注意】可疑写法：level5 用的是 reps[2]=7 而非 reps[3]=3（与 reps 注释/论文配置的 level5 重复 3 次不符，疑似笔误或有意加深）
            self.level5.append(EESP(out_channels[4], out_channels[4], k=K[4], r_lim=r_lim[4],  # 等通道 EESP（stride=1，内部带残差）
                                    norm_layer=norm_layer))  # EESP 参数续行：归一化层；构造与 append 调用闭合

        self.level5.append(_ConvBNPReLU(out_channels[4], out_channels[4], 3, 1, 1,  # 追加 3×3 深度卷积（groups=通道数）：(N,C4,H/32,W/32) → (N,C4,H/32,W/32)
                                        groups=out_channels[4], norm_layer=norm_layer))  # 深度卷积参数续行：groups=通道数（逐通道卷积）；append 调用闭合
        self.level5.append(_ConvBNPReLU(out_channels[4], out_channels[5], 1, 1, 0,  # 追加分组 1×1 卷积升维到末端通道：(N,C4,·) → (N,C5,·)
                                        groups=K[4], norm_layer=norm_layer))  # 分组 1×1 卷积参数续行：groups=K[4]（4 组）；append 调用闭合

        self.fc = nn.Linear(out_channels[5], num_classes)  # 分类全连接：(N,C5) → (N,num_classes)

        for m in self.modules():  # 遍历全部子模块初始化权重
            if isinstance(m, nn.Conv2d):  # 卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming 正态初始化（relu）
                if m.bias is not None:  # 若带偏置
                    nn.init.constant_(m.bias, 0)  # 偏置置 0
            elif isinstance(m, nn.BatchNorm2d):  # BN 层
                nn.init.constant_(m.weight, 1)  # γ 置 1
                nn.init.constant_(m.bias, 0)  # β 置 0
            elif isinstance(m, nn.Linear):  # 全连接层
                nn.init.normal_(m.weight, std=0.001)  # 小标准差正态初始化
                if m.bias is not None:  # 若带偏置
                    nn.init.constant_(m.bias, 0)  # 偏置置 0

    # 前向传播：seg=True（默认）返回 level1~4 四个尺度的特征供分割头使用；seg=False 走分类分支返回 logits
    def forward(self, x, seg=True):
        out_l1 = self.level1(x)  # level1：(N,3,H,W) → (N,C0,H/2,W/2)

        out_l2 = self.level2_0(out_l1, x)  # level2：(N,C0,H/2,W/2) → (N,C1,H/4,W/4)（融合原图增强分支）

        out_l3_0 = self.level3_0(out_l2, x)  # level3 入口：(N,C1,H/4,W/4) → (N,C2,H/8,W/8)
        for i, layer in enumerate(self.level3):  # 依次通过 level3 的各 EESP
            if i == 0:  # 第一个 EESP
                out_l3 = layer(out_l3_0)  # 以 level3 入口特征为输入
            else:  # 其余 EESP
                out_l3 = layer(out_l3)  # 链式传递

        out_l4_0 = self.level4_0(out_l3, x)  # level4 入口：(N,C2,H/8,W/8) → (N,C3,H/16,W/16)
        for i, layer in enumerate(self.level4):  # 依次通过 level4 的各 EESP
            if i == 0:  # 第一个 EESP
                out_l4 = layer(out_l4_0)  # 以 level4 入口特征为输入
            else:  # 其余 EESP
                out_l4 = layer(out_l4)  # 链式传递

        if not seg:  # 分类模式：继续 level5 与分类头
            # 下采样进入 level5（英文原注释意为"已下采样"）
            out_l5_0 = self.level5_0(out_l4)  # down-sampled  # (N,C3,H/16,W/16) → (N,C4,H/32,W/32)
            for i, layer in enumerate(self.level5):  # 依次通过 level5 各层（含末尾两个卷积）
                if i == 0:  # 第一个 EESP
                    out_l5 = layer(out_l5_0)  # 以 level5 入口特征为输入
                else:  # 其余层
                    out_l5 = layer(out_l5)  # 链式传递

            output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)  # 全局自适应平均池化：(N,C5,H/32,W/32) → (N,C5,1,1)
            output_g = F.dropout(output_g, p=0.2, training=self.training)  # 训练阶段随机失活（p=0.2）
            output_1x1 = output_g.view(output_g.size(0), -1)  # 展平：(N,C5,1,1) → (N,C5)

            return self.fc(output_1x1)  # 全连接分类：(N,C5) → (N,num_classes)
        return out_l1, out_l2, out_l3, out_l4  # 分割模式：返回 4 个 level 的多尺度特征（1/2、1/4、1/8、1/16）


# EESPNet 工厂函数：pretrained 预训练开关；**kwargs 透传给 EESPNet 构造器
def eespnet(pretrained=False, **kwargs):
    model = EESPNet(**kwargs)  # 构造 EESPNet 主网络

    if pretrained:  # 若请求预训练权重
        raise ValueError("Don't support pretrained")  # 不支持，直接抛异常
    return model  # 返回随机初始化的模型


if __name__ == '__main__':  # 直接运行本文件时做前向自测
    img = torch.randn(1, 3, 224, 224)  # 随机输入 (1,3,224,224)
    model = eespnet()  # 默认配置构造 EESPNet
    out = model(img)  # 前向传播（seg 默认 True，返回 4 个尺度的特征元组）
