# ============================================================================
# 【文件说明】ocnet.py —— 目标上下文网络 OCNet（Object Context Network）分割模型：
#               基于 SegBaseModel 骨干（默认 ResNet101、膨胀卷积、1/8 分辨率），
#               头部 _OCHead 按 oc_arch 提供 base / pyramid / asp 三种自注意力结构。
# 【核心思想】自注意力块（non-local block）：对输入特征用 1×1 卷积分别生成
#               Query、Key（降到 key_channels=256）与 Value（降到 value_channels），
#               展平为 (N, H*W, Ck) / (N, Ck, H*W) / (N, H*W, Cv)；
#               相似度矩阵 sim_map = (Q·K^T) / sqrt(Ck)，形状 (N, H*W, H*W)，
#               在最后一维（Key 位置维）softmax 得到每个位置对所有位置的注意力权重，
#               再与 V 相乘把全局上下文聚合回每个位置，最后经零初始化的 1×1 卷积 W
#               投影回输出通道（训练初期等价于恒等映射）。
#               —— Base-OC：单尺度全局自注意力（scale>1 时先 MaxPool 降算力再上采样）；
#               —— Pyramid-OC：把特征图切成 scale×scale 个子块，块内独立自注意力后拼回；
#               —— ASP-OC：ASPP 空洞卷积分支（12/24/36）+ scale=2 的 Base-OC 分支拼接。
# 【论文】Yuhui Yuan, Jingdong Wang. "OCNet: Object Context Network for Scene
#         Parsing." arXiv preprint arXiv:1809.00916 (2018).
# ============================================================================
""" Object Context Network for Scene Parsing"""  # 模块级英文 docstring（原样保留）
import torch  # PyTorch 主库：张量运算、torch.load 等
import torch.nn as nn  # 神经网络模块：Conv2d / BatchNorm2d / Sequential / ModuleList 等
import torch.nn.functional as F  # 函数式接口：interpolate、softmax 等

from .segbase import SegBaseModel  # 分割模型基类：封装骨干网络构建与 base_forward
from .fcn import _FCNHead  # FCN 辅助头，用作 aux 分支的分类头

__all__ = ['OCNet', 'get_ocnet', 'get_base_ocnet_resnet101_citys',  # 模块对外导出的公开符号列表
           'get_pyramid_ocnet_resnet101_citys', 'get_asp_ocnet_resnet101_citys']  # 其余便捷构造函数名


# OCNet 主模型：继承 SegBaseModel，默认骨干 resnet101
class OCNet(SegBaseModel):  # OCNet 主模型类
    # 类 docstring：OCNet 模型说明与参数（原英文保留）
    r"""OCNet

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
        Yuhui Yuan, Jingdong Wang. "OCNet: Object Context Network for Scene Parsing."
        arXiv preprint arXiv:1809.00916 (2018).
    """  # OCNet 类英文 docstring 结束（原样保留）
    # 【注意】docstring 中 norm_layer 默认值写作 nn.BatchNorm（MXNet Gluon 风格遗留描述，
    # 废弃 API 痕迹），本仓库实际使用 PyTorch 的 nn.BatchNorm2d；
    # 且 docstring 写 default:'resnet50'，而函数签名默认值实为 'resnet101'
    # ---- __init__ 参数说明 ----
    # nclass: 类别数
    # backbone: 骨干类型，默认 'resnet101'
    # oc_arch: 上下文模块结构，'base' / 'pyramid' / 'asp' 三选一
    # aux: 是否启用辅助损失头，默认 False
    # pretrained_base: 是否加载骨干的 ImageNet 预训练权重
    # **kwargs: 透传给基类与头部（如 norm_layer）
    def __init__(self, nclass, backbone='resnet101', oc_arch='base', aux=False, pretrained_base=True, **kwargs):  # 构造函数
        super(OCNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)  # 基类构造：构建骨干、设置 aux
        self.head = _OCHead(nclass, oc_arch, **kwargs)  # 上下文头部：按 oc_arch 选择 Base/Pyramid/ASP-OC
        if self.aux:  # 若启用辅助损失
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)  # FCN 辅助头：作用于 stage3 特征（1024 通道）

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])  # 独占参数组标记：head（及 auxlayer）训练时可单独设置学习率

    # ---- forward 参数说明 ----
    # x: 输入图像 (N, 3, H, W)
    def forward(self, x):  # 前向：骨干 → 上下文头 → 上采样；可选辅助分支
        size = x.size()[2:]  # 记录输入空间尺寸 (H, W)
        _, _, c3, c4 = self.base_forward(x)  # 骨干前向：取 stage3（1024 通道）与 stage4（2048 通道）特征，(N, C, H/8, W/8)
        outputs = []  # 输出收集列表
        x = self.head(c4)  # 上下文头处理 c4 → 主 logits (N, nclass, H/8, W/8)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  # 双线性上采样回 (N, nclass, H, W)
        outputs.append(x)  # 主输出入列表

        if self.aux:  # 辅助分支
            auxout = self.auxlayer(c3)  # FCN 辅助头处理 stage3 特征
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)  # 上采样回原尺寸
            outputs.append(auxout)  # 辅助输出入列表
        return tuple(outputs)  # 返回 (主输出,) 或 (主输出, 辅助输出)


# OCNet 头部：3×3 卷积 2048→512（asp 结构除外）→ 所选上下文模块 → 1×1 卷积分类
class _OCHead(nn.Module):  # OCNet 分割头：继承 nn.Module
    # ---- __init__ 参数说明 ----
    # nclass: 类别数
    # oc_arch: 'base' / 'pyramid' / 'asp'，选择上下文模块类型
    # norm_layer: 归一化层类型，默认 BatchNorm2d
    # **kwargs: 透传给上下文模块
    def __init__(self, nclass, oc_arch, norm_layer=nn.BatchNorm2d, **kwargs):  # 构造函数
        super(_OCHead, self).__init__()  # 调用父类构造
        if oc_arch == 'base':  # Base-OC 结构：全局单尺度自注意力
            self.context = nn.Sequential(  # 上下文模块容器
                nn.Conv2d(2048, 512, 3, 1, padding=1, bias=False),  # 3×3 卷积降维 2048→512，步长 1、padding 1 保持尺寸
                norm_layer(512),  # BN 归一化
                nn.ReLU(True),  # 就地 ReLU
                BaseOCModule(512, 512, 256, 256, scales=([1]), norm_layer=norm_layer, **kwargs))  # Base-OC：Q/K 256 通道、V 256 通道、尺度 [1]（不池化）；【注意】scales=([1]) 的圆括号冗余但合法
        elif oc_arch == 'pyramid':  # Pyramid-OC 结构：多尺度分块自注意力
            self.context = nn.Sequential(  # 上下文模块容器
                nn.Conv2d(2048, 512, 3, 1, padding=1, bias=False),  # 3×3 卷积降维 2048→512
                norm_layer(512),  # BN 归一化
                nn.ReLU(True),  # 就地 ReLU
                PyramidOCModule(512, 512, 256, 512, scales=([1, 2, 3, 6]), norm_layer=norm_layer, **kwargs))  # Pyramid-OC：尺度 [1,2,3,6]，V 通道 512
        elif oc_arch == 'asp':  # ASP-OC 结构：ASPP + 自注意力
            self.context = ASPOCModule(2048, 512, 256, 512, norm_layer=norm_layer, **kwargs)  # ASP-OC：内部含空洞卷积分支与 scale=2 的 Base-OC
        else:  # 未知结构名
            raise ValueError("Unknown OC architecture!")  # 抛出值错误异常

        self.out = nn.Conv2d(512, nclass, 1)  # 1×1 卷积分类器：512 → nclass

    # ---- forward 参数说明 ----
    # x: 骨干输出特征 (N, 2048, H, W)
    def forward(self, x):  # 前向：上下文建模 → 分类
        x = self.context(x)  # 上下文模块：(N, 2048, H, W) → (N, 512, H, W)
        return self.out(x)  # 输出 logits (N, nclass, H, W)


# 基础自注意力块（non-local block）
# 职责：Q/K/V 由 1×1 卷积生成；Q·K^T/sqrt(Ck) 得相似度矩阵，softmax 后加权 V，
#       输出经零初始化 1×1 卷积 W 投影；scale>1 时先对特征 MaxPool 降算量再上采样回来
class BaseAttentionBlock(nn.Module):  # 自注意力块：继承 nn.Module
    """The basic implementation for self-attention block/non-local block."""  # 原英文 docstring（保留）

    # ---- __init__ 参数说明 ----
    # in_channels: 输入通道数
    # out_channels: 输出通道数
    # key_channels: Q/K 降维后的通道数 Ck
    # value_channels: V 降维后的通道数 Cv
    # scale: 空间下采样倍数（>1 时对输入 MaxPool，降低 (H*W)^2 相似度矩阵的计算量）
    # norm_layer: 归一化层类型
    # **kwargs: 预留参数
    def __init__(self, in_channels, out_channels, key_channels, value_channels,  # 构造函数（续行见下）
                 scale=1, norm_layer=nn.BatchNorm2d, **kwargs):  # 续行：scale 默认 1（不池化）
        super(BaseAttentionBlock, self).__init__()  # 调用父类构造
        self.scale = scale  # 保存池化倍数
        self.key_channels = key_channels  # 保存 Q/K 通道数 Ck
        self.value_channels = value_channels  # 保存 V 通道数 Cv
        if scale > 1:  # 仅当下采样倍数大于 1 时才创建池化层
            self.pool = nn.MaxPool2d(scale)  # scale×scale 最大池化（作用于 Q/K/V 的输入特征）

        self.f_value = nn.Conv2d(in_channels, value_channels, 1)  # Value 生成器：1×1 卷积 in→Cv
        self.f_key = nn.Sequential(  # Key 生成器：1×1 卷积 + BN + ReLU
            nn.Conv2d(in_channels, key_channels, 1),  # 1×1 卷积降维 in→Ck
            norm_layer(key_channels),  # BN 归一化
            nn.ReLU(True)  # 就地 ReLU
        )  # f_key 定义结束
        self.f_query = self.f_key  # 【注意】Query 直接复用 f_key 同一对象（Q、K 权重完全共享），与常见 Q/K 独立卷积的实现不同，属可疑写法（保持原样）
        self.W = nn.Conv2d(value_channels, out_channels, 1)  # 输出投影：1×1 卷积 Cv→out_channels
        nn.init.constant_(self.W.weight, 0)  # W 权重零初始化：训练初期注意力输出为 0（近似恒等）
        nn.init.constant_(self.W.bias, 0)  # W 偏置零初始化

    # ---- forward 参数说明 ----
    # x: 输入特征 (N, C, H, W)
    def forward(self, x):  # 前向：可选池化 → Q/K/V → 相似度 → softmax → 加权聚合 → 投影 → 上采样
        batch_size, c, w, h = x.size()  # 解包尺寸；【注意】变量名 w/h 实际拿到的是第 2/3 维（即 H/W），命名与含义颠倒但后续使用一致
        if self.scale > 1:  # 若需下采样
            x = self.pool(x)  # 对输入池化：(N, C, H, W) → (N, C, H/s, W/s)

        value = self.f_value(x).view(batch_size, self.value_channels, -1).permute(0, 2, 1)  # V：卷积后展平再转置 → (N, H'W', Cv)
        query = self.f_query(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1)  # Q：卷积后展平再转置 → (N, H'W', Ck)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)  # K：卷积后展平 → (N, Ck, H'W')

        sim_map = torch.bmm(query, key) * (self.key_channels ** -.5)  # 相似度矩阵：(N, H'W', Ck) × (N, Ck, H'W') → (N, H'W', H'W')，再除以 sqrt(Ck) 缩放（scaled dot-product）
        sim_map = F.softmax(sim_map, dim=-1)  # 在最后一维（Key 位置维）softmax：每个查询位置对所有键位置的注意力权重

        context = torch.bmm(sim_map, value).permute(0, 2, 1).contiguous()  # 加权聚合 V：(N, H'W', H'W') × (N, H'W', Cv) → (N, H'W', Cv)，再转置为 (N, Cv, H'W') 并连续化
        context = context.view(batch_size, self.value_channels, *x.size()[2:])  # 还原空间维度 → (N, Cv, H', W')
        context = self.W(context)  # 1×1 卷积投影 → (N, out_channels, H', W')
        if self.scale > 1:  # 若之前做过下采样
            context = F.interpolate(context, size=(w, h), mode='bilinear', align_corners=True)  # 双线性上采样回原分辨率 (N, out, H, W)

        return context  # 返回聚合了全局上下文的特征


# Base-OC 模块：整图（或多尺度相加）自注意力 + 输入特征拼接 + 1×1 投影
class BaseOCModule(nn.Module):  # Base-OC 上下文模块：继承 nn.Module
    """Base-OC"""  # 原英文 docstring（保留）

    # ---- __init__ 参数说明 ----
    # in_channels: 输入通道数
    # out_channels: 输出通道数
    # key_channels: Q/K 通道数
    # value_channels: V 通道数
    # scales: 注意力池化尺度列表（默认 [1]；多尺度时各路输出逐元素相加）
    # norm_layer: 归一化层类型
    # concat: 是否把输入特征与注意力输出在通道维拼接（默认 True）
    # **kwargs: 透传给注意力块
    def __init__(self, in_channels, out_channels, key_channels, value_channels,  # 构造函数（续行见下）
                 scales=([1]), norm_layer=nn.BatchNorm2d, concat=True, **kwargs):  # 续行：scales 默认 [1]（圆括号冗余但合法）
        super(BaseOCModule, self).__init__()  # 调用父类构造
        self.stages = nn.ModuleList([  # 多尺度注意力块列表
            BaseAttentionBlock(in_channels, out_channels, key_channels, value_channels, scale, norm_layer, **kwargs)  # 每个尺度构建一个自注意力块
            for scale in scales])  # 列表推导：按 scales 逐个构建
        in_channels = in_channels * 2 if concat else in_channels  # 若拼接输入，投影层输入通道数翻倍
        self.project = nn.Sequential(  # 融合投影：1×1 卷积 + BN + ReLU + Dropout
            nn.Conv2d(in_channels, out_channels, 1),  # 1×1 卷积：(2*in)→out（拼接时）或 in→out
            norm_layer(out_channels),  # BN 归一化
            nn.ReLU(True),  # 就地 ReLU
            nn.Dropout2d(0.05)  # 2D Dropout，丢弃概率 0.05
        )  # project 定义结束
        self.concat = concat  # 保存拼接开关

    # ---- forward 参数说明 ----
    # x: 输入特征 (N, in_channels, H, W)
    def forward(self, x):  # 前向：多尺度注意力（相加）→ 可选拼接输入 → 投影
        priors = [stage(x) for stage in self.stages]  # 各尺度注意力输出列表，每个 (N, out, H, W)
        context = priors[0]  # 以第一个尺度的输出为基准
        for i in range(1, len(priors)):  # 遍历其余尺度
            context += priors[i]  # 多尺度输出逐元素相加
        if self.concat:  # 若启用拼接
            context = torch.cat([context, x], 1)  # 注意力输出与原输入在通道维拼接 → (N, in+out, H, W)
        out = self.project(context)  # 1×1 投影回 out_channels → (N, out, H, W)
        return out  # 返回上下文增强特征


# 金字塔自注意力块：把特征图切成 scale×scale 个子块，块内独立做自注意力，再按原布局拼回
class PyramidAttentionBlock(nn.Module):  # 金字塔自注意力块：继承 nn.Module
    """The basic implementation for pyramid self-attention block/non-local block"""  # 原英文 docstring（保留）

    # ---- __init__ 参数说明 ----
    # in_channels / out_channels / key_channels / value_channels / scale / norm_layer / **kwargs
    # 含义同 BaseAttentionBlock（scale 此处为每条边的分块数，共 scale^2 块）
    def __init__(self, in_channels, out_channels, key_channels, value_channels,  # 构造函数（续行见下）
                 scale=1, norm_layer=nn.BatchNorm2d, **kwargs):  # 续行：scale 默认 1（即整图）
        super(PyramidAttentionBlock, self).__init__()  # 调用父类构造
        self.scale = scale  # 每条边的分块数（共 scale^2 个子块）
        self.value_channels = value_channels  # 保存 V 通道数
        self.key_channels = key_channels  # 保存 Q/K 通道数

        self.f_value = nn.Conv2d(in_channels, value_channels, 1)  # Value 生成器：1×1 卷积
        self.f_key = nn.Sequential(  # Key 生成器：1×1 卷积 + BN + ReLU
            nn.Conv2d(in_channels, key_channels, 1),  # 1×1 卷积降维
            norm_layer(key_channels),  # BN 归一化
            nn.ReLU(True)  # 就地 ReLU
        )  # f_key 定义结束
        self.f_query = self.f_key  # 【注意】Query 复用 f_key（Q、K 权重共享），与常见实现不同，可疑写法（保持原样）
        self.W = nn.Conv2d(value_channels, out_channels, 1)  # 输出投影：1×1 卷积 Cv→out
        nn.init.constant_(self.W.weight, 0)  # W 权重零初始化
        nn.init.constant_(self.W.bias, 0)  # W 偏置零初始化

    # ---- forward 参数说明 ----
    # x: 输入特征 (N, C, H, W)
    def forward(self, x):  # 前向：切块坐标计算 → 整图 Q/K/V → 逐块自注意力 → 拼回整图 → 投影
        batch_size, c, w, h = x.size()  # 解包尺寸；【注意】w/h 命名与实际 H/W 颠倒（同 BaseAttentionBlock）

        local_x = list()  # 各子块在第 2 维（高度）上的起止坐标，扁平存放 [start0, end0, start1, end1, ...]
        local_y = list()  # 各子块在第 3 维（宽度）上的起止坐标（【注意】local_x/local_y 与维度名称互换，但使用一致）
        step_w, step_h = w // self.scale, h // self.scale  # 每个子块的边长（整除）
        for i in range(self.scale):  # 遍历高度方向的块索引 i
            for j in range(self.scale):  # 遍历宽度方向的块索引 j
                start_x, start_y = step_w * i, step_h * j  # 当前块起始坐标
                end_x, end_y = min(start_x + step_w, w), min(start_y + step_h, h)  # 当前块结束坐标（防止越界）
                if i == (self.scale - 1):  # 高度方向最后一块
                    end_x = w  # 强制延伸到边界（整除余量并入最后一块）
                if j == (self.scale - 1):  # 宽度方向最后一块
                    end_y = h  # 强制延伸到边界
                local_x += [start_x, end_x]  # 记录该块的高度起止坐标
                local_y += [start_y, end_y]  # 记录该块的宽度起止坐标

        value = self.f_value(x)  # 整图计算 V：(N, Cv, H, W)（稍后按块切片）
        query = self.f_query(x)  # 整图计算 Q：(N, Ck, H, W)
        key = self.f_key(x)  # 整图计算 K：(N, Ck, H, W)

        local_list = list()  # 收集各子块的注意力输出（按 i*scale+j 的行优先顺序）
        local_block_cnt = (self.scale ** 2) * 2  # 坐标条目总数 = 块数 × 2（每块存起、止两个值）
        for i in range(0, local_block_cnt, 2):  # 每次步进 2：取一个块的 (start, end) 坐标对
            value_local = value[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]  # 切出当前块的 V
            query_local = query[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]  # 切出当前块的 Q
            key_local = key[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]  # 切出当前块的 K

            w_local, h_local = value_local.size(2), value_local.size(3)  # 当前块的空间尺寸
            value_local = value_local.contiguous().view(batch_size, self.value_channels, -1).permute(0, 2, 1)  # 块内 V 展平转置 → (N, n, Cv)
            query_local = query_local.contiguous().view(batch_size, self.key_channels, -1).permute(0, 2, 1)  # 块内 Q 展平转置 → (N, n, Ck)
            key_local = key_local.contiguous().view(batch_size, self.key_channels, -1)  # 块内 K 展平 → (N, Ck, n)

            sim_map = torch.bmm(query_local, key_local) * (self.key_channels ** -.5)  # 块内相似度：(N, n, Ck) × (N, Ck, n) → (N, n, n)，除以 sqrt(Ck) 缩放
            sim_map = F.softmax(sim_map, dim=-1)  # 在最后一维（块内 Key 位置维）softmax

            context_local = torch.bmm(sim_map, value_local).permute(0, 2, 1).contiguous()  # 块内加权聚合：(N, n, n) × (N, n, Cv) → (N, n, Cv) → 转置 (N, Cv, n)
            context_local = context_local.view(batch_size, self.value_channels, w_local, h_local)  # 还原块形状 (N, Cv, w_local, h_local)
            local_list.append(context_local)  # 收集该块结果

        context_list = list()  # 存放按行拼接好的整行结果
        for i in range(0, self.scale):  # 遍历块行 i
            row_tmp = list()  # 当前行内的各块
            for j in range(self.scale):  # 遍历行内块列 j
                row_tmp.append(local_list[j + i * self.scale])  # 取出第 (i, j) 块（行优先编号）
            context_list.append(torch.cat(row_tmp, 3))  # 行内沿宽度维（dim=3）拼接
        # 【注意】local_x 实际索引第 2 维（高度）、local_y 索引第 3 维（宽度），变量名与维度含义互换，
        # 但切块与拼接两处使用一致，功能正确（仅命名可疑）
        context = torch.cat(context_list, 2)  # 各行沿高度维（dim=2）拼接 → (N, Cv, H, W)
        context = self.W(context)  # 1×1 卷积投影 → (N, out_channels, H, W)

        return context  # 返回金字塔自注意力输出


# Pyramid-OC 模块：多尺度（scales=[1,2,3,6]）分块注意力输出拼接 + 通道扩张后的输入拼接 → 投影
class PyramidOCModule(nn.Module):  # Pyramid-OC 上下文模块：继承 nn.Module
    """Pyramid-OC"""  # 原英文 docstring（保留）

    # ---- __init__ 参数说明 ----
    # in_channels: 输入通道数；out_channels: 输出通道数
    # key_channels / value_channels: Q/K 与 V 的通道数
    # scales: 分块尺度列表（默认 [1]；如 [1,2,3,6] 表示整图 + 2×2 + 3×3 + 6×6 分块）
    # norm_layer: 归一化层类型；**kwargs: 透传给注意力块
    def __init__(self, in_channels, out_channels, key_channels, value_channels,  # 构造函数（续行见下）
                 scales=([1]), norm_layer=nn.BatchNorm2d, **kwargs):  # 续行：默认尺度列表 [1]
        super(PyramidOCModule, self).__init__()  # 调用父类构造
        self.stages = nn.ModuleList([  # 各尺度的金字塔注意力块列表
            PyramidAttentionBlock(in_channels, out_channels, key_channels, value_channels, scale, norm_layer, **kwargs)  # 每个尺度一个分块注意力块
            for scale in scales])  # 列表推导：按 scales 逐个构建
        self.up_dr = nn.Sequential(  # 输入特征的通道扩张（与 len(scales) 路注意力拼接对齐）
            nn.Conv2d(in_channels, in_channels * len(scales), 1),  # 1×1 卷积：in → in*len(scales)
            norm_layer(in_channels * len(scales)),  # BN 归一化
            nn.ReLU(True)  # 就地 ReLU
        )  # up_dr 定义结束
        self.project = nn.Sequential(  # 融合投影：1×1 卷积 + BN + ReLU + Dropout
            nn.Conv2d(in_channels * len(scales) * 2, out_channels, 1),  # 1×1 卷积：注意力路 len(scales)*out + 输入路 in*len(scales)（此处 in==out，共 2*in*len）→ out
            norm_layer(out_channels),  # BN 归一化
            nn.ReLU(True),  # 就地 ReLU
            nn.Dropout2d(0.05)  # 2D Dropout 0.05
        )  # project 定义结束

    # ---- forward 参数说明 ----
    # x: 输入特征 (N, in_channels, H, W)
    def forward(self, x):  # 前向：各尺度注意力 → 与通道扩张后的输入拼接 → 投影
        priors = [stage(x) for stage in self.stages]  # 各尺度注意力输出，每个 (N, out, H, W)
        context = [self.up_dr(x)]  # 输入经通道扩张后作为第一路，(N, in*len(scales), H, W)
        for i in range(len(priors)):  # 遍历各尺度输出
            context += [priors[i]]  # 逐路追加到列表
        context = torch.cat(context, 1)  # 通道维拼接 → (N, in*len(scales) + len(scales)*out, H, W)（本配置下 2048+4×512=4096）
        out = self.project(context)  # 1×1 投影 → (N, out_channels, H, W)
        return out  # 返回上下文增强特征


# ASP-OC 模块：ASPP 风格空洞卷积多分支 + Base-OC 自注意力分支，五路拼接后投影
class ASPOCModule(nn.Module):  # ASP-OC 上下文模块：继承 nn.Module
    """ASP-OC"""  # 原英文 docstring（保留）

    # ---- __init__ 参数说明 ----
    # in_channels: 输入通道（2048）；out_channels: 输出通道（512）
    # key_channels / value_channels: Q/K 与 V 的通道数
    # atrous_rates: 三个空洞卷积分支的膨胀率（默认 12/24/36，对应 1/8 分辨率下的多尺度感受野）
    # norm_layer: 归一化层类型；**kwargs: 透传给 Base-OC
    def __init__(self, in_channels, out_channels, key_channels, value_channels,  # 构造函数（续行见下）
                 atrous_rates=(12, 24, 36), norm_layer=nn.BatchNorm2d, **kwargs):  # 续行：默认膨胀率 12/24/36
        super(ASPOCModule, self).__init__()  # 调用父类构造
        self.context = nn.Sequential(  # 自注意力分支：3×3 卷积降维 → Base-OC（scale=2 池化、不拼接输入）
            nn.Conv2d(in_channels, out_channels, 3, padding=1),  # 3×3 卷积降维 2048→512；【注意】未设 bias=False，与 b1~b4 风格不一致（默认带 bias）
            norm_layer(out_channels),  # BN 归一化
            nn.ReLU(True),  # 就地 ReLU
            BaseOCModule(out_channels, out_channels, key_channels, value_channels, ([2]), norm_layer, False, **kwargs))  # Base-OC：尺度 [2]（K/V 先池化 2 倍省算力），concat=False（不拼接输入，由本模块统一拼接）

        rate1, rate2, rate3 = tuple(atrous_rates)  # 解包三个空洞率
        # 空洞分支 1：膨胀率 rate1（默认 12）
        self.b1 = nn.Sequential(  # 顺序容器
            nn.Conv2d(in_channels, out_channels, 3, padding=rate1, dilation=rate1, bias=False),  # 3×3 空洞卷积，padding=rate1 保持尺寸
            norm_layer(out_channels),  # BN 归一化
            nn.ReLU(True))  # 就地 ReLU
        # 空洞分支 2：膨胀率 rate2（默认 24）
        self.b2 = nn.Sequential(  # 顺序容器
            nn.Conv2d(in_channels, out_channels, 3, padding=rate2, dilation=rate2, bias=False),  # 3×3 空洞卷积
            norm_layer(out_channels),  # BN 归一化
            nn.ReLU(True))  # 就地 ReLU
        # 空洞分支 3：膨胀率 rate3（默认 36）
        self.b3 = nn.Sequential(  # 顺序容器
            nn.Conv2d(in_channels, out_channels, 3, padding=rate3, dilation=rate3, bias=False),  # 3×3 空洞卷积
            norm_layer(out_channels),  # BN 归一化
            nn.ReLU(True))  # 就地 ReLU
        # 分支 4：1×1 卷积（捕获逐点/通道上下文）
        self.b4 = nn.Sequential(  # 顺序容器
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 1×1 卷积
            norm_layer(out_channels),  # BN 归一化
            nn.ReLU(True))  # 就地 ReLU

        self.project = nn.Sequential(  # 五路拼接后的融合投影
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),  # 1×1 卷积：5×512 → 512
            norm_layer(out_channels),  # BN 归一化
            nn.ReLU(True),  # 就地 ReLU
            nn.Dropout2d(0.1)  # 2D Dropout 0.1
        )  # project 定义结束

    # ---- forward 参数说明 ----
    # x: 骨干输出特征 (N, 2048, H, W)
    def forward(self, x):  # 前向：五路并行 → 通道拼接 → 融合投影
        feat1 = self.context(x)  # 自注意力分支输出 (N, 512, H, W)
        feat2 = self.b1(x)  # 空洞率 12 分支输出 (N, 512, H, W)
        feat3 = self.b2(x)  # 空洞率 24 分支输出 (N, 512, H, W)
        feat4 = self.b3(x)  # 空洞率 36 分支输出 (N, 512, H, W)
        feat5 = self.b4(x)  # 1×1 卷积分支输出 (N, 512, H, W)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)  # 通道维拼接 → (N, 5*512, H, W)
        out = self.project(out)  # 融合投影 → (N, 512, H, W)
        return out  # 返回上下文增强特征


# 便捷构造函数：按数据集 / 骨干 / 结构 / 预训练选项构建 OCNet
# ---- 参数说明 ----
# dataset: 数据集名（决定类别数），默认 'citys'
# backbone: 骨干类型，默认 'resnet50'
# oc_arch: 'base' / 'pyramid' / 'asp'，默认 'base'
# pretrained: 是否加载完整模型预训练权重
# root: 预训练权重文件存放根目录
# pretrained_base: 是否加载骨干的 ImageNet 预训练权重
# **kwargs: 透传给 OCNet 构造函数
def get_ocnet(dataset='citys', backbone='resnet50', oc_arch='base', pretrained=False, root='~/.torch/models',  # 构造入口（续行见下）
              pretrained_base=True, **kwargs):  # 续行：骨干预训练开关与透传参数
    acronyms = {  # 数据集名 → 预训练权重文件名中的缩写映射
        'pascal_voc': 'pascal_voc',  # Pascal VOC
        'pascal_aug': 'pascal_aug',  # Pascal VOC 增强版
        'ade20k': 'ade',  # ADE20K → 缩写 ade
        'coco': 'coco',  # COCO
        'citys': 'citys',  # Cityscapes
    }  # 缩写映射结束
    from ..data.dataloader import datasets  # 局部导入数据集注册表（避免循环导入），用于查询类别数
    model = OCNet(datasets[dataset].NUM_CLASS, backbone=backbone, oc_arch=oc_arch,  # 按数据集类别数构建 OCNet（续行见下）
                  pretrained_base=pretrained_base, **kwargs)  # 续行：传入骨干预训练开关
    if pretrained:  # 若需加载完整模型预训练权重
        from .model_store import get_model_file  # 局部导入权重文件路径解析工具
        device = torch.device(kwargs['local_rank'])  # 【注意】直接取 kwargs['local_rank']，未传该键时会 KeyError；用 int 构造 device 的写法也较脆弱
        model.load_state_dict(torch.load(get_model_file('%s_ocnet_%s_%s' % (  # 拼出权重文件名 '<oc_arch>_ocnet_<backbone>_<数据集缩写>' 并加载（【注意】torch.load 未指定 weights_only，新版 PyTorch 会告警）
            oc_arch, backbone, acronyms[dataset]), root=root),  # 续行：填充结构 / 骨干 / 数据集缩写
            map_location=device))  # 将权重张量映射到指定设备
    return model  # 返回构建好的模型


def get_base_ocnet_resnet101_citys(**kwargs):  # 便捷构造：Cityscapes + ResNet101 + Base-OC
    return get_ocnet('citys', 'resnet101', 'base', **kwargs)  # 委托 get_ocnet


def get_pyramid_ocnet_resnet101_citys(**kwargs):  # 便捷构造：Cityscapes + ResNet101 + Pyramid-OC
    return get_ocnet('citys', 'resnet101', 'pyramid', **kwargs)  # 委托 get_ocnet


def get_asp_ocnet_resnet101_citys(**kwargs):  # 便捷构造：Cityscapes + ResNet101 + ASP-OC
    return get_ocnet('citys', 'resnet101', 'asp', **kwargs)  # 委托 get_ocnet


if __name__ == '__main__':  # 直接运行本文件时的简易自测入口
    img = torch.randn(1, 3, 256, 256)  # 构造随机输入 (1, 3, 256, 256)
    model = get_asp_ocnet_resnet101_citys()  # 构建 ASP-OC 版 OCNet
    outputs = model(img)  # 前向推理，验证输出形状
