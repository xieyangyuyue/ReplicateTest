# ============================================================================
# 【文件说明】encnet.py —— 上下文编码网络 EncNet（Context Encoding）分割模型：
#               基于 SegBaseModel 骨干（ResNet 系列、膨胀卷积），分割头 _EncHead
#               含上下文编码模块 EncModule（字典学习式编码 + SE 通道重标定 + SE-loss）。
# 【核心思想】1) 上下文编码（Encoding 层，源自 Deep TEN）：将每个像素的 512 维特征
#                  与 K=32 个可学习码字（codewords）计算带平滑因子的缩放平方距离，
#                  softmax 得到像素-码字分配权重，再对"像素特征-码字"的加权残差
#                  按全部像素求和，聚合出 (N, K, D) 的上下文语义描述，对 K 求均值
#                  得到 (N, 512) 的全局上下文向量；
#               2) 上下文向量经全连接 + Sigmoid 得到 512 维通道门控 y，
#                  输出 F.relu_(x + x*y)（SE 风格的通道注意力重标定，逐通道缩放）；
#               3) SE-loss：另一个线性层从上下文向量预测"图中出现了哪些类别"，
#                  作为辅助监督（多标签分类）；
#               4) 可选 lateral 侧连：stage2/stage3 浅层特征经 1×1 卷积后与深层
#                  特征拼接融合，恢复空间细节。
# 【论文】Hang Zhang, Kristin Dana, et al. "Context Encoding for Semantic
#         Segmentation." CVPR 2018
# ============================================================================
"""Context Encoding for Semantic Segmentation"""  # 模块级英文 docstring（原样保留）
import torch  # PyTorch 主库：张量运算、nn.Parameter、torch.load 等
import torch.nn as nn  # 神经网络模块：Conv2d / BatchNorm / Sequential / Linear 等
import torch.nn.functional as F  # 函数式接口：interpolate、softmax、relu_ 等

from .segbase import SegBaseModel  # 分割模型基类：封装骨干网络构建与 base_forward
from .fcn import _FCNHead  # FCN 辅助头，用作 aux 分支的分类头

__all__ = ['EncNet', 'EncModule', 'get_encnet', 'get_encnet_resnet50_ade',  # 模块对外导出的公开符号列表
           'get_encnet_resnet101_ade', 'get_encnet_resnet152_ade']  # 其余便捷构造函数名


# EncNet 主模型：继承 SegBaseModel（本类无 docstring，原文如此）
class EncNet(SegBaseModel):  # EncNet 主模型类
    # ---- __init__ 参数说明 ----
    # nclass: 类别数
    # backbone: 骨干类型，默认 'resnet50'
    # aux: 是否启用辅助损失（FCN 头作用于 stage3 特征）
    # se_loss: 是否启用语义编码损失（预测图中出现的类别）
    # lateral: 是否启用浅层侧连融合（默认 False）
    # pretrained_base: 是否加载骨干的 ImageNet 预训练权重
    # **kwargs: 透传（norm_layer 等）
    def __init__(self, nclass, backbone='resnet50', aux=True, se_loss=True, lateral=False,  # 构造函数（续行见下）
                 pretrained_base=True, **kwargs):  # 续行：骨干预训练开关与透传参数
        super(EncNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)  # 调用基类构造：构建骨干、设置 aux
        self.head = _EncHead(2048, nclass, se_loss=se_loss, lateral=lateral, **kwargs)  # 编码头：输入 2048 为 ResNet stage4 输出通道数
        if aux:  # 若启用辅助损失
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)  # FCN 辅助头：作用于 stage3 特征（1024 通道）

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])  # 独占参数组标记：head（及 auxlayer）训练时可单独设置学习率

    # ---- forward 参数说明 ----
    # x: 输入图像 (N, 3, H, W)
    def forward(self, x):  # 前向：骨干 → 编码头（含 SE 重标定 / SE-loss）→ 上采样；可选辅助
        size = x.size()[2:]  # 记录输入空间尺寸 (H, W)
        features = self.base_forward(x)  # 骨干前向，返回 (c1, c2, c3, c4) 四阶段特征

        x = list(self.head(*features))  # 头部接收全部四阶段特征，输出列表 [主 logits, (可选)SE-loss 预测]
        x[0] = F.interpolate(x[0], size, mode='bilinear', align_corners=True)  # 主 logits 双线性上采样 → (N, nclass, H, W)
        if self.aux:  # 辅助分支
            auxout = self.auxlayer(features[2])  # FCN 辅助头处理 stage3 特征（c3，1024 通道）
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)  # 上采样回原尺寸
            x.append(auxout)  # 追加辅助输出
        return tuple(x)  # 返回 (主输出, [SE-loss 输出], [辅助输出]) 组成的元组


# 编码头：3×3 卷积降维 →（可选）浅层侧连融合 → 上下文编码模块 → Dropout + 1×1 分类
class _EncHead(nn.Module):  # EncNet 分割头：继承 nn.Module
    # ---- __init__ 参数说明 ----
    # in_channels: 输入通道（2048）
    # nclass: 类别数
    # se_loss: 是否构造 SE-loss 分类层
    # lateral: 是否构造浅层侧连（【注意】本类默认形参为 True，但 EncNet 构造时默认传入 False）
    # norm_layer / norm_kwargs: 归一化层类型与额外参数
    # **kwargs: 透传给 EncModule
    def __init__(self, in_channels, nclass, se_loss=True, lateral=True,  # 构造函数（续行见下）
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):  # 续行：归一化配置
        super(_EncHead, self).__init__()  # 调用父类构造
        self.lateral = lateral  # 保存侧连开关
        self.conv5 = nn.Sequential(  # 深层特征降维卷积块：2048→512
            nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),  # 3×3 卷积，padding=1 保持尺寸，无 bias（后接 BN）
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),  # 归一化层；norm_kwargs 为 None 时展开空字典
            nn.ReLU(True)  # 就地 ReLU
        )  # conv5 定义结束
        if lateral:  # 若启用侧连
            self.connect = nn.ModuleList([  # 两个浅层特征的 1×1 适配卷积（stage2、stage3 各一个）
                nn.Sequential(  # stage2（512 通道）→ 512
                    nn.Conv2d(512, 512, 1, bias=False),  # 1×1 卷积
                    norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),  # BN
                    nn.ReLU(True)),  # 就地 ReLU
                nn.Sequential(  # stage3（1024 通道）→ 512
                    nn.Conv2d(1024, 512, 1, bias=False),  # 1×1 卷积降维
                    norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),  # BN
                    nn.ReLU(True)),  # 就地 ReLU
            ])  # connect 定义结束
            self.fusion = nn.Sequential(  # 三路拼接后的融合卷积：3×512 → 512
                nn.Conv2d(3 * 512, 512, 3, padding=1, bias=False),  # 3×3 卷积
                norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),  # BN
                nn.ReLU(True)  # 就地 ReLU
            )  # fusion 定义结束
        self.encmodule = EncModule(512, nclass, ncodes=32, se_loss=se_loss,  # 上下文编码模块：码字数 K=32（续行见下）
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)  # 续行：透传归一化配置
        self.conv6 = nn.Sequential(  # 主分类头：Dropout + 1×1 卷积
            nn.Dropout(0.1, False),  # Dropout p=0.1、inplace=False
            nn.Conv2d(512, nclass, 1)  # 1×1 卷积分类器：512 → nclass
        )  # conv6 定义结束

    # ---- forward 参数说明 ----
    # *inputs: 骨干四阶段特征 (c1, c2, c3, c4)；其中 c1（inputs[0]）未被使用
    def forward(self, *inputs):  # 前向：深层降维 →（可选）侧连融合 → 编码模块 → 分类
        feat = self.conv5(inputs[-1])  # 对 c4 降维：(N, 2048, H, W) → (N, 512, H, W)
        if self.lateral:  # 若启用侧连
            c2 = self.connect[0](inputs[1])  # stage2 特征适配：512→512
            c3 = self.connect[1](inputs[2])  # stage3 特征降维：1024→512
            feat = self.fusion(torch.cat([feat, c2, c3], 1))  # 三路通道拼接 (N, 1536, H, W) → 融合卷积 (N, 512, H, W)
        outs = list(self.encmodule(feat))  # 上下文编码 + SE 重标定，得到 [增强特征, (可选)SE-loss 预测]
        outs[0] = self.conv6(outs[0])  # 增强特征 → 主 logits (N, nclass, H, W)
        return tuple(outs)  # 返回 (主 logits, [SE-loss 预测])


# 上下文编码模块：Encoding 字典聚合 → FC+Sigmoid 通道门控（SE）→ 残差缩放；可选 SE-loss 层
class EncModule(nn.Module):  # 上下文编码模块：继承 nn.Module
    # ---- __init__ 参数说明 ----
    # in_channels: 输入通道（512）
    # nclass: 类别数（SE-loss 输出维度）
    # ncodes: 码字个数 K（默认 32）
    # se_loss: 是否构造 SE-loss 线性层
    # norm_layer / norm_kwargs: 归一化配置
    # **kwargs: 预留参数
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True,  # 构造函数（续行见下）
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):  # 续行：归一化配置
        super(EncModule, self).__init__()  # 调用父类构造
        self.se_loss = se_loss  # 保存 SE-loss 开关
        self.encoding = nn.Sequential(  # 编码管线：1×1 卷积 → BN → ReLU → Encoding 字典聚合 → BN1d → ReLU → 对码字求均值
            nn.Conv2d(in_channels, in_channels, 1, bias=False),  # 1×1 卷积（通道不变，做特征变换）
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),  # BN
            nn.ReLU(True),  # 就地 ReLU
            Encoding(D=in_channels, K=ncodes),  # 字典编码层：D=512 维特征、K=32 个码字，输出 (N, K, D)
            nn.BatchNorm1d(ncodes),  # 对 K 个码字做 BatchNorm1d（输入按 (N, K, D) 处理，K 作为通道维）
            nn.ReLU(True),  # 就地 ReLU
            Mean(dim=1)  # 在 K 维（dim=1）求均值 → (N, D)，得到全局上下文描述向量
        )  # encoding 定义结束
        self.fc = nn.Sequential(  # 门控生成：全连接 + Sigmoid，输出 (0,1) 通道权重
            nn.Linear(in_channels, in_channels),  # 全连接 512→512
            nn.Sigmoid()  # Sigmoid 压缩到 (0,1)
        )  # fc 定义结束
        if self.se_loss:  # 若启用 SE-loss
            self.selayer = nn.Linear(in_channels, nclass)  # 从上下文向量预测各类别是否出现（多标签），512→nclass

    # ---- forward 参数说明 ----
    # x: 输入特征 (N, 512, H, W)
    def forward(self, x):  # 前向：上下文编码 → 门控 → SE 通道重标定（+ SE-loss）
        en = self.encoding(x)  # 上下文编码向量 (N, 512)
        b, c, _, _ = x.size()  # 取批大小 b 与通道数 c（H、W 不需要）
        gamma = self.fc(en)  # 门控权重 (N, 512)，Sigmoid 输出
        y = gamma.view(b, c, 1, 1)  # 变形为 (N, C, 1, 1)，以便逐通道广播
        outputs = [F.relu_(x + x * y)]  # SE 重标定：x + x*y（原特征 + 加权特征）后就地 ReLU，形状不变 (N, C, H, W)（就地操作作用于新张量，安全）
        if self.se_loss:  # 若启用 SE-loss
            outputs.append(self.selayer(en))  # 追加类别出现概率预测 (N, nclass)
        return tuple(outputs)  # 返回 (重标定特征, [SE-loss 预测])


# 字典编码层（源自 Deep TEN / "Deep Template Network" 的 encoding 层实现）
# 职责：学习 K 个 D 维码字，将 N 个像素特征按 softmax 分配权重聚合为 K 组加权残差表示
class Encoding(nn.Module):  # 编码层：继承 nn.Module
    # ---- __init__ 参数说明 ----
    # D: 特征维度（512）
    # K: 码字个数
    def __init__(self, D, K):  # 构造函数
        super(Encoding, self).__init__()  # 调用父类构造
        # init codewords and smoothing factor（原英文注释，保留）
        self.D, self.K = D, K  # 保存特征维度 D 与码字数 K
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)  # 码字矩阵 (K, D)，可学习参数
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)  # 每个码字的平滑因子 (K,)，可学习参数
        self.reset_params()  # 调用初始化方法

    def reset_params(self):  # 初始化码字与平滑因子
        std1 = 1. / ((self.K * self.D) ** (1 / 2))  # 均匀分布边界：1/sqrt(K*D)
        self.codewords.data.uniform_(-std1, std1)  # 码字在 [-std1, std1] 内均匀初始化
        self.scale.data.uniform_(-1, 0)  # 平滑因子在 [-1, 0] 内均匀初始化（负值，起"负距离缩放"作用，源自 Deep TEN）

    # ---- forward 参数说明 ----
    # X: 输入特征，4D (N, D, H, W) 或 3D (N, D, n)
    def forward(self, X):  # 前向：像素-码字分配权重计算 → 加权残差聚合
        # input X is a 4D tensor（原英文注释，保留）
        assert (X.size(1) == self.D)  # 校验通道维等于 D
        B, D = X.size(0), self.D  # 取批大小 B 与特征维 D
        if X.dim() == 3:  # 三维输入分支
            # BxDxN -> BxNxD（原英文注释，保留）
            X = X.transpose(1, 2).contiguous()  # (N, D, n) → (N, n, D) 并连续化
        elif X.dim() == 4:  # 四维输入分支（本模型实际路径）
            # BxDxHxW -> Bx(HW)xD（原英文注释，保留）
            X = X.view(B, D, -1).transpose(1, 2).contiguous()  # (N, D, H, W) → (N, D, H*W) → (N, H*W, D)
        else:  # 其他维度输入
            raise RuntimeError('Encoding Layer unknown input dims!')  # 抛出运行时错误
        # assignment weights BxNxK（原英文注释，保留）
        A = F.softmax(self.scale_l2(X, self.codewords, self.scale), dim=2)  # 像素-码字缩放距离 (N, n, K)，在码字维（dim=2）softmax 得分配权重
        # aggregate（原英文注释，保留）
        E = self.aggregate(A, X, self.codewords)  # 按分配权重聚合残差 → (N, K, D)
        return E  # 返回编码结果

    def __repr__(self):  # 自定义打印格式（输出形如 "Encoding(N x512=>32x512)"：N 个 D 维特征被编码为 K 个 D 维码字表示）
        # 【注意】下面两行是同一 return 语句的反斜杠续行；行尾 '\' 之后不能再加注释
        # （否则破坏续行语法），续行内容为拼接类名与维度信息
        return self.__class__.__name__ + '(' \
               + 'N x' + str(self.D) + '=>' + str(self.K) + 'x' + str(self.D) + ')'  # 拼接 "N x{D}=>{K}x{D}" 与收尾 ")"，完成 repr 字符串

    # ---- scale_l2 参数说明 ----
    # X: (N, n, D) 像素特征；C: (K, D) 码字；S: (K,) 平滑因子
    @staticmethod  # 静态方法（不依赖实例状态）
    def scale_l2(X, C, S):  # 计算像素与各码字之间的平滑平方距离
        S = S.view(1, 1, C.size(0), 1)  # 平滑因子变形为 (1, 1, K, 1) 以便广播
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))  # X 扩展为 (N, n, K, D)
        C = C.unsqueeze(0).unsqueeze(0)  # 码字扩展为 (1, 1, K, D)
        SL = S * (X - C)  # 平滑因子乘残差，(N, n, K, D)
        SL = SL.pow(2).sum(3)  # 最后一维平方求和 → (N, n, K) 的缩放平方距离
        return SL  # 返回距离矩阵

    # ---- aggregate 参数说明 ----
    # A: (N, n, K) 分配权重；X: (N, n, D) 像素特征；C: (K, D) 码字
    @staticmethod  # 静态方法
    def aggregate(A, X, C):  # 按分配权重聚合像素-码字残差
        A = A.unsqueeze(3)  # 权重变形为 (N, n, K, 1)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))  # X 扩展为 (N, n, K, D)
        C = C.unsqueeze(0).unsqueeze(0)  # 码字扩展为 (1, 1, K, D)
        E = A * (X - C)  # 加权残差，(N, n, K, D)
        E = E.sum(1)  # 对全部像素（n 维）求和 → (N, K, D)
        return E  # 返回编码结果


# 均值层：在指定维度求均值的简单包装（nn.Module 化以便放入 Sequential）
class Mean(nn.Module):  # 均值层：继承 nn.Module
    # ---- __init__ 参数说明 ----
    # dim: 求均值的维度
    # keep_dim: 是否保留该维度（默认 False）
    def __init__(self, dim, keep_dim=False):  # 构造函数
        super(Mean, self).__init__()  # 调用父类构造
        self.dim = dim  # 保存目标维度
        self.keep_dim = keep_dim  # 保存是否保留维度

    # ---- forward 参数说明 ----
    # input: 输入张量（【注意】参数名 input 遮蔽了 Python 内建函数，原样保留）
    def forward(self, input):  # 前向：按 dim 求均值
        return input.mean(self.dim, self.keep_dim)  # 如 (N, K, D) 在 dim=1 上求均值 → (N, D)


# 便捷构造函数：按数据集 / 骨干 / 预训练选项构建 EncNet
# ---- 参数说明 ----
# dataset: 数据集名（决定类别数），默认 'pascal_voc'
# backbone: 骨干类型，默认 'resnet50'
# pretrained: 是否加载完整模型预训练权重
# root: 预训练权重文件存放根目录
# pretrained_base: 是否加载骨干的 ImageNet 预训练权重
# **kwargs: 透传给 EncNet 构造函数
def get_encnet(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',  # 构造入口（续行见下）
               pretrained_base=True, **kwargs):  # 续行：骨干预训练开关与透传参数
    acronyms = {  # 数据集名 → 预训练权重文件名中的缩写映射
        'pascal_voc': 'pascal_voc',  # Pascal VOC
        'pascal_aug': 'pascal_aug',  # Pascal VOC 增强版
        'ade20k': 'ade',  # ADE20K → 缩写 ade
        'coco': 'coco',  # COCO
        'citys': 'citys',  # Cityscapes
    }  # 缩写映射结束
    from ..data.dataloader import datasets  # 局部导入数据集注册表（避免循环导入），用于查询类别数
    model = EncNet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建 EncNet
    if pretrained:  # 若需加载完整模型预训练权重
        from .model_store import get_model_file  # 局部导入权重文件路径解析工具
        device = torch.device(kwargs['local_rank'])  # 【注意】直接取 kwargs['local_rank']，未传该键时会 KeyError；用 int 构造 device 的写法也较脆弱
        model.load_state_dict(torch.load(get_model_file('encnet_%s_%s' % (backbone, acronyms[dataset]), root=root),  # 拼出权重文件名 'encnet_<backbone>_<数据集缩写>' 并加载（【注意】torch.load 未指定 weights_only，新版 PyTorch 会告警）
                              map_location=device))  # 将权重张量映射到指定设备
    return model  # 返回构建好的模型


def get_encnet_resnet50_ade(**kwargs):  # 便捷构造：ADE20K + ResNet50 版 EncNet
    return get_encnet('ade20k', 'resnet50', **kwargs)  # 委托 get_encnet


def get_encnet_resnet101_ade(**kwargs):  # 便捷构造：ADE20K + ResNet101 版 EncNet
    return get_encnet('ade20k', 'resnet101', **kwargs)  # 委托 get_encnet


def get_encnet_resnet152_ade(**kwargs):  # 便捷构造：ADE20K + ResNet152 版 EncNet
    return get_encnet('ade20k', 'resnet152', **kwargs)  # 委托 get_encnet


if __name__ == '__main__':  # 直接运行本文件时的简易自测入口
    img = torch.randn(2, 3, 224, 224)  # 构造随机输入 (2, 3, 224, 224)
    model = get_encnet_resnet50_ade()  # 构建 ADE20K + ResNet50 版 EncNet
    outputs = model(img)  # 前向推理，验证输出形状
