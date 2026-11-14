# ============================================================================
# 【文件说明】segbase.py —— 所有语义分割模型的公共基类 SegBaseModel。
#              负责按名称构建 ResNet(v1s 系列)预训练骨干（可选空洞卷积或 JPU 两种模式），
#              提供统一的骨干前向接口 base_forward 以及 evaluate / demo 工具方法。
#              工程位置：core/models/segbase.py，是 FCN(v2)/PSPNet/DeepLab 等模型的共同父类。
# 【核心思想】"预训练分类骨干 + 任务分割头"的通用范式：骨干输出 4 个阶段特征 (c1~c4)；
#            dilated 模式下 layer3/4 使用空洞卷积，使 c3/c4 保持 1/8 分辨率；
#            jpu 模式下骨干不膨胀，改由 JPU(联合金字塔上采样)模块融合多尺度特征并恢复到 1/8。
# ============================================================================
"""Base Model for Semantic Segmentation"""
import torch.nn as nn  # PyTorch 神经网络模块：提供 nn.Module 基类及各类网络层

from ..nn import JPU  # 从 core/nn 包导入联合金字塔上采样模块（Joint Pyramid Upsampling）
from .base_models.resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s  # 导入三种深度的 ResNet-v1s 骨干网络构造函数

__all__ = ['SegBaseModel']  # 模块公开接口：仅导出 SegBaseModel 基类


class SegBaseModel(nn.Module):  # 分割模型公共基类：继承 nn.Module，获得参数注册/保存/设备迁移能力
    # 类 docstring：基类说明与 backbone 参数可选取值（原英文保留）
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """  # 上方英文 docstring 原样保留：说明 backbone 参数的可选取值

    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数（含背景），决定分割头输出通道数
    # aux：是否启用辅助损失分支（在中间特征上额外接一个分割头）
    # backbone：骨干类型 'resnet50' / 'resnet101' / 'resnet152'（默认 resnet50）
    # jpu：是否用 JPU 模块替代空洞卷积骨干（True 时骨干 dilated=False）
    # pretrained_base：是否加载骨干的 ImageNet 预训练权重
    # **kwargs：其余参数透传给骨干构造函数与 JPU（如 norm_layer、norm_kwargs）
    def __init__(self, nclass, aux, backbone='resnet50', jpu=False, pretrained_base=True, **kwargs):  # 构造基类：解析骨干类型并构建预训练网络
        super(SegBaseModel, self).__init__()  # 调用父类 nn.Module 构造函数，完成模块树初始化
        kwargs.pop('local_rank', None)  # local_rank 只服务于 DDP/权重加载，ResNet 骨干构造函数不接受该参数
        dilated = False if jpu else True  # 膨胀开关：启用 JPU 时骨干不用空洞卷积（多尺度交给 JPU 处理），否则骨干用空洞卷积保持 1/8 分辨率
        self.aux = aux  # 记录辅助分支开关，供子类 forward / demo 判断
        self.nclass = nclass  # 记录分割类别数
        if backbone == 'resnet50':  # 骨干分支：50 层
            self.pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)  # 构建 ResNet50-v1s，四阶段输出通道 256/512/1024/2048
        elif backbone == 'resnet101':  # 骨干分支：101 层
            self.pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)  # 构建 ResNet101-v1s 骨干
        elif backbone == 'resnet152':  # 骨干分支：152 层
            self.pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)  # 构建 ResNet152-v1s 骨干
        else:  # 未知骨干名称
            raise RuntimeError('unknown backbone: {}'.format(backbone))  # 抛运行时错误：不支持的骨干类型

        self.jpu = JPU([512, 1024, 2048], width=512, **kwargs) if jpu else None  # jpu=True 时构建 JPU：输入通道 [512,1024,2048] 对应 c2/c3/c4，内部统一宽度 512；否则置 None 表示不使用

    # ----------------------- base_forward 参数说明 -----------------------
    # x：输入图像张量 (N, 3, H, W)
    # 返回：普通模式为 (c1, c2, c3, c4) 四阶段特征元组；JPU 模式返回 JPU 融合后的特征
    def base_forward(self, x):  # 骨干前向：逐层提取 4 个阶段的特征图，供子类分割头使用
        """forwarding pre-trained network"""  # 原注释保留：预训练网络前向
        x = self.pretrained.conv1(x)  # 7x7 卷积 stride=2 → (N, 64, H/2, W/2)
        x = self.pretrained.bn1(x)  # 第一层 BatchNorm，通道数不变
        x = self.pretrained.relu(x)  # ReLU 激活，通道数不变
        x = self.pretrained.maxpool(x)  # 3x3 最大池化 stride=2 → (N, 64, H/4, W/4)
        c1 = self.pretrained.layer1(x)  # stage1 → (N, 256, H/4, W/4)，分辨率最高的语义特征
        c2 = self.pretrained.layer2(c1)  # stage2 stride=2 → (N, 512, H/8, W/8)
        c3 = self.pretrained.layer3(c2)  # stage3 → dilated 模式 (N, 1024, H/8, W/8)；JPU 模式 (N, 1024, H/16, W/16)
        c4 = self.pretrained.layer4(c3)  # stage4 → dilated 模式 (N, 2048, H/8, W/8)；JPU 模式 (N, 2048, H/16, W/16)

        if self.jpu:  # 若构建了 JPU 模块
            return self.jpu(c1, c2, c3, c4)  # 四阶段特征送入 JPU 融合上采样，返回融合特征（分辨率 1/8）
        else:  # 普通空洞卷积骨干模式
            return c1, c2, c3, c4  # 直接返回 4 个阶段特征图组成的元组

    # ----------------------- evaluate 参数说明 -----------------------
    # x：输入图像张量 (N, 3, H, W)；返回主预测张量 (N, nclass, H, W)
    def evaluate(self, x):  # 推理接口：只取 forward 输出元组的第一个（主输出），忽略辅助输出
        """evaluating network with inputs and targets"""  # 原注释保留：带输入的评估前向
        return self.forward(x)[0]  # 调用子类实现的 forward 并取第 0 个 → (N, nclass, H, W)

    # ----------------------- demo 参数说明 -----------------------
    # x：输入图像张量 (N, 3, H, W)；返回主预测张量 (N, nclass, H, W)
    def demo(self, x):  # 演示接口：前向并剥离辅助输出，仅返回主预测
        pred = self.forward(x)  # 子类 forward 的完整输出（主输出 + 可选辅助输出组成的元组）
        if self.aux:  # 若启用了辅助分支
            pred = pred[0]  # 仅保留主输出，丢弃辅助输出
        return pred  # 返回主预测张量 (N, nclass, H, W)
