# ============================================================================
# 【文件说明】__init__.py —— core/models 包的初始化文件：将 model_zoo 中的模型获取
#              接口提升到包级别，使外部可直接 from core.models import get_model 等；
#              同时该文件的存在使 core/models 成为可导入的 Python 包。
# ============================================================================
"""Model Zoo"""  # 原 docstring 保留：模型动物园
from .model_zoo import get_model, get_model_list  # 从 model_zoo 导入"按名建模型"与"列出模型清单"两个接口供包外使用（【注意】未导入 get_segmentation_model，需从 model_zoo 子模块获取）
