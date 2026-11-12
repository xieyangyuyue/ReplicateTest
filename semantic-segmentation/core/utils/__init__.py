# ============================================================================
# 【文件说明】__init__.py —— core/utils 工具包的入口文件。
# 职责：将最常用的工具函数从各子模块汇聚导出，供外部以
#       `from core.utils import download, makedirs ...` 方式直接引用，
#       屏蔽内部子模块的组织细节。
# 【注意】distributed / parallel / logger / visualize 等子模块未在此导出，
#         使用时需显式 `from core.utils.distributed import ...`。
# ============================================================================
"""Utility functions."""
from __future__ import absolute_import  # Python2 兼容的绝对导入声明（Py3 下为无操作，保留不影响运行）

from .download import download, check_sha1  # 导入下载工具：download（带进度条的流式下载）、check_sha1（SHA1 完整性校验）
from .filesystem import makedirs, try_import_pycocotools  # 导入文件系统工具：makedirs（递归建目录）、try_import_pycocotools（懒加载/自动安装 pycocotools）
