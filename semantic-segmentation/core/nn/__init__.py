# ============================================================================
# 【文件说明】__init__.py —— core/nn 包（分割网络基础组件库）的入口文件，
#             聚合子模块中的公开符号，供外部以 `from core.nn import XXX` 统一导入。
# 【核心思想】利用 Python 包导入机制：jpu.py 提供 JPU（联合金字塔上采样）模块，
#             basic.py 提供 _ConvBNReLU / _DepthwiseConv / InvertedResidual 等
#             基础构件，两者经 `import *` 汇总到包命名空间，简化各分割模型的引用。
# ============================================================================
"""Seg NN Modules"""  # 原有 docstring：语义分割神经网络模块
from .jpu import *  # 导入 jpu.py 中 __all__ 声明的符号（JPU 联合金字塔上采样模块）
from .basic import *  # 导入 basic.py 中 __all__ 声明的符号（卷积/BN/激活组合件、倒残差块等）
