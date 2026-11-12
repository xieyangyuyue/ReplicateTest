# ============================================================================
# 【文件说明】filesystem.py —— 文件系统与依赖包导入工具。
# 核心逻辑：
#   1) makedirs：mkdir -p 式递归建目录，通过捕获 EEXIST 异常容忍“目录
#      已存在”（也天然容忍多进程并发创建同一目录的竞态条件）；
#   2) try_import / try_import_cv2：尝试导入指定包，失败时抛出带自定义
#      安装提示的 ImportError；
#   3) import_try_install：导入失败时自动调用 pip --user 安装再重试导入，
#      若仍失败则把 user site-packages 追加进 sys.path 后再导一次；
#   4) try_import_pycocotools：pycocotools 的懒加载/自动安装封装——先探测
#      导入，失败则先装 Cython，再按平台选源：Windows 从 GitHub 源码装，
#      其余平台装 PyPI 版。
# 【注意】pip 自动安装逻辑依赖旧的 pip.main / pip._internal.main 接口，
#         现代 pip（>=10 乃至 >=23）已移除或弃用，该路径在新环境基本失效。
# ============================================================================
"""Filesystem utility functions."""
from __future__ import absolute_import  # Python2 兼容的绝对导入声明（Py3 下为无操作）
import os  # 提供目录创建等系统调用
import errno  # 提供 EEXIST 等错误码常量，用于识别“目录已存在”异常


# 参数：path——目标目录路径
def makedirs(path):
    # docstring：目录不存在时递归创建（效果等价 mkdir -p）
    """Create directory recursively if not exists.
    Similar to `makedir -p`, you can skip checking existence before this function.
    Parameters
    ----------
    path : str
        Path of the desired dir
    """
    try:  # 先直接尝试创建（比“先判断再创建”少一次系统调用，且天然规避并发竞态）
        os.makedirs(path)  # 递归创建多级目录
    except OSError as exc:  # 创建过程抛出 OS 级异常
        if exc.errno != errno.EEXIST:  # 错误号不是“已存在”(EEXIST)——例如权限不足
            raise  # 其他错误原样向上重抛
        # errno==EEXIST：目录已被（可能是并发的其他进程）创建，视为成功，静默忽略
        # 现代等价写法：os.makedirs(path, exist_ok=True)


# 参数：package——包名字符串；message——导入失败时的自定义错误提示（None 则重抛原始异常）
def try_import(package, message=None):
    # docstring：尝试导入指定包，支持导入失败时的自定义错误提示
    """Try import specified package, with custom message support.
    Parameters
    ----------
    package : str
        The name of the targeting package.
    message : str, default is None
        If not None, this function will raise customized error message when import error is found.
    Returns
    -------
    module if found, raise ImportError otherwise
    """
    try:  # 尝试动态导入
        return __import__(package)  # 导入顶层包并返回模块对象（【注意】不支持 a.b.c 形式的子模块导入）
    except ImportError as e:  # 导入失败
        if not message:  # 未提供自定义提示
            raise e  # 原样重抛原始 ImportError（保留原始堆栈）
        raise ImportError(message)  # 换成自定义提示再抛（会丢失原始异常上下文）


# 无参数：探测导入 cv2（OpenCV）
def try_import_cv2():
    # docstring：运行时尝试导入 cv2（成功返回模块，失败抛带安装提示的 ImportError）
    """Try import cv2 at runtime.
    Returns
    -------
    cv2 module if found. Raise ImportError otherwise
    """
    # 安装提示文案（该行以反斜杠续行，行尾不能追加注释）
    msg = "cv2 is required, you can install by package manager, e.g. 'apt-get', \
        or `pip install opencv-python --user` (note that this is unofficial PYPI package)."
    return try_import('cv2', msg)  # 复用 try_import 导入 cv2，失败时给出安装建议


# 参数：package——包名；extern_url——包不在 PyPI 上时的外部安装地址（如 git+https 源码地址）
def import_try_install(package, extern_url=None):
    # docstring：尝试导入指定包，未安装则先用 pip 安装再导入
    """Try import the specified package.
    If the package not installed, try use pip to install and import if success.
    Parameters
    ----------
    package : str
        The name of the package trying to import.
    extern_url : str or None, optional
        The external url if package is not hosted on PyPI.
        For example, you can install a package using:
         "pip install git+http://github.com/user/repo/tarball/master/egginfo=xxx".
        In this case, you can pass the url to the extern_url.
    Returns
    -------
    <class 'Module'>
        The imported python module.
    """
    try:  # 先尝试直接导入（包已安装的最常见路径）
        return __import__(package)  # 成功则直接返回模块
    except ImportError:  # 未安装：进入自动安装流程
        # 【注意】下面两条 pip 导入在现代环境基本失效：
        # pip>=10 移除顶层 pip.main；pip._internal.main 在更新版 pip 中亦被移除/改造
        try:  # 优先旧版 pip 的命令行入口
            from pip import main as pipmain  # 旧版 pip(<10) 可用；新版会抛 ImportError
        except ImportError:  # 旧入口不存在
            from pip._internal import main as pipmain  # pip>=10 的内部入口（新版同样不稳）

        # trying to install package
        url = package if extern_url is None else extern_url  # 未提供外部地址就按包名从 PyPI 装
        # 用 pip 命令行入口执行安装：装到用户目录（--user）；失败时 pip 会抛 SystemExit
        pipmain(['install', '--user', url])  # will raise SystemExit Error if fails

        # trying to load again
        try:  # 安装成功后重试导入
            return __import__(package)  # 直接再导一次
        except ImportError:  # 装完仍导不到：多半是 --user 目录不在当前 sys.path
            import sys  # 用于修改模块搜索路径
            import site  # 用于查询用户 site-packages 目录
            user_site = site.getusersitepackages()  # 用户级包安装目录
            if user_site not in sys.path:  # 该目录尚未加入搜索路径
                sys.path.append(user_site)  # 追加到 sys.path 末尾
            return __import__(package)  # 再次尝试导入
    return __import__(package)  # 【注意】死代码：上方 try/except 各分支均有 return 或抛异常，本行永远执行不到


# 【注意】悬空字符串：位于模块中部，不是任何函数的 docstring，仅作说明文字（无实际作用）
"""Import helper for pycocotools"""


# NOTE: for developers
# please do not import any pycocotools in __init__ because we are trying to lazy
# import pycocotools to avoid install it for other users who may not use it.
# only import when you actually use it


# 无参数：懒加载/自动安装 pycocotools（本仓库 __init__.py 导出，供数据集代码按需调用）
def try_import_pycocotools():
    """Tricks to optionally install and import pycocotools"""  # 单行 docstring：pycocotools 的懒加载/自动安装技巧
    # first we can try import pycocotools
    try:  # 优先直接导入（已安装则零成本通过）
        import pycocotools as _  # 仅探测可导入性，别名 _ 表示不使用该模块对象
    except ImportError:  # 未安装：进入安装流程
        import os  # 用于判断操作系统类型（局部重复导入模块级 os，无害）
        # we need to install pycootools, which is a bit tricky
        # pycocotools sdist requires Cython, numpy(already met)
        import_try_install('cython')  # 先装编译依赖 Cython（numpy 假定主环境已满足）
        # pypi pycocotools is not compatible with windows
        win_url = 'git+https://github.com/zhreshold/cocoapi.git#subdirectory=PythonAPI'  # Windows 专用源码安装地址（PyPI 版在 Windows 编译不过）
        try:  # 按平台选择安装源
            if os.name == 'nt':  # Windows 平台
                import_try_install('pycocotools', win_url)  # 从 GitHub 源码安装
            else:  # Linux / macOS
                import_try_install('pycocotools')  # 直接安装 PyPI 版
        except ImportError:  # 安装后仍导入失败
            faq = 'cocoapi FAQ'  # 【注意】仅占位字符串，并非可点击的真实 FAQ 链接
            raise ImportError('Cannot import or install pycocotools, please refer to %s.' % faq)  # 抛出带指引信息的导入错误
