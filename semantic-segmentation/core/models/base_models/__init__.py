# ============================================================================
# 【文件说明】__init__.py —— 骨干网络（backbone）子包 base_models 的聚合入口。
#              通过逐个 `from .xxx import *` 将各骨干网络模块的公开符号统一
#              re-export 到 core.models.base_models 命名空间，分割模型侧只需
#              `from .base_models import xxx` 即可取得任意骨干的构造函数。
# 【核心思想】语义分割 = 编码器（本目录的 CNN 骨干，负责逐级下采样、提取
#              多尺度语义特征）+ 解码器（分割头，负责恢复分辨率并逐像素分类）；
#              本文件不含任何模型逻辑，仅做导入转发，是骨干与分割模型之间的
#              桥接层。
# 【工程位置】core/models/base_models/__init__.py；被 core/models 下的
#              denseaspp / bisenet 等模型文件及本包内部各模块引用。
# ============================================================================
from .densenet import *  # 导入 DenseNet 系骨干（含膨胀 dilated 变体）的全部公开符号
from .resnet import *  # 导入经典 ResNet（torchvision 风格）的 ResNet 类与 resnet18~152 工厂函数
from .resnetv1b import *  # 导入 ResNetV1b/V1s 骨干（gluon-cv 移植、支持空洞卷积，本工程分割模型的默认编码器）
from .vgg import *  # 导入 VGG 系骨干（plain 卷积堆叠，FCN 等模型的编码器）
from .eespnet import *  # 导入 EESPNet 轻量骨干（ESP 多尺度卷积模块，面向嵌入式实时分割）
from .xception import *  # 导入 Xception 骨干（深度可分离卷积，DeepLabV3+ 的编码器）
