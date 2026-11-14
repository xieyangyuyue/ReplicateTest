# ============================================================================
# 【文件说明】model_store.py —— 预训练模型权重文件的本地缓存管理与下载工具。
#              维护 ResNet 骨干权重的 SHA1 校验表，从 encoding 官方 S3 仓库下载 zip
#              并解压到 ~/.torch/models；分割模型权重只做本地查找。
#              工程位置：core/models/model_store.py，被各模型工厂函数的 pretrained 分支调用。
# 【核心思想】以"文件名 + SHA1 哈希"保证权重完整性：本地命中则校验后复用；
#            未命中或校验失败则从远程仓库下载 zip、解压出 .pth、再次校验后返回路径。
# 【注意】get_model_file 并不会真正下载分割权重，本地缺失时直接抛异常（见函数内注释）。
# ============================================================================
"""Model store which provides pretrained models."""
from __future__ import print_function  # Python 2 兼容导入：Python 3 中 print 本为函数，此处为无操作【注意】遗留写法

import os  # 操作系统接口：路径展开/拼接、目录创建、文件删除
import zipfile  # zip 压缩包处理：解压下载下来的权重压缩包

from ..utils.download import download, check_sha1  # 自研下载工具：download 下载文件到本地、check_sha1 校验文件哈希

__all__ = ['get_model_file', 'get_resnet_file']  # 模块公开接口：两个权重文件定位函数

_model_sha1 = {name: checksum for checksum, name in [  # 字典推导式：把 (SHA1, 模型名) 列表反转为 {模型名: 完整 SHA1} 映射
    # 【已更新】官方旧仓库 hangzh.s3.amazonaws.com 已失效（403），改用 PyTorch-Encoding
    # 当前使用的 wasabisys 仓库；权重名也由旧的 resnet50/101/152 改为
    # resnet50s/101s/152s（"s" 即 deep_stem 版本，与本文件的 resnet*_v1s 结构一致）。
    ('a75c83cfc89a56a4e8ba71b14f1ec67e923787b3', 'resnet50s'),  # ResNet50s（deep_stem）骨干权重的 SHA1 校验值
    ('03a0f310d6447880f1b22a83bd7d1aa7fc702c6e', 'resnet101s'),  # ResNet101s（deep_stem）骨干权重的 SHA1 校验值
    ('36670e8bc2428ecd5b7db1578538e2dd23872813', 'resnet152s'),  # ResNet152s（deep_stem）骨干权重的 SHA1 校验值
]}  # (SHA1, 模型名) 列表与外层字典推导结束，得到 {模型名: SHA1} 完整校验表

# 本工程对外仍沿用 resnet50/101/152 的旧名称，这里映射到新版仓库中的实际权重名
_model_alias = {'resnet50': 'resnet50s', 'resnet101': 'resnet101s', 'resnet152': 'resnet152s'}  # 旧名称 → 新权重名
# 旧版仓库/旧缓存的文件名与 SHA1（结构与 v1s 完全一致）；若本地已有旧权重则直接复用，避免重新下载
_legacy_model_sha1 = {name: checksum for checksum, name in [  # {旧模型名: 旧 SHA1}
    ('25c4b50959ef024fcc050213a06b614899f94b3d', 'resnet50'),
    ('2a57e44de9c853fa015b172309a1ee7e2d0e4e2a', 'resnet101'),
    ('0d43d698c66aceaa2bc0309f55efdd7ff4b143af', 'resnet152'),
]}
_legacy_alias = {v: k for k, v in _model_alias.items()}  # 新权重名 → 旧缓存名（resnet50s → resnet50）

encoding_repo_url = 'https://s3.us-west-1.wasabisys.com/encoding'  # encoding 官方权重仓库新地址，可用环境变量 ENCODING_REPO 覆盖
_url_format = '{repo_url}models/{file_name}.zip'  # 下载 URL 模板：仓库地址 + models/ + 文件名.zip（仓库地址末尾会自动补斜杠）


# ----------------------- short_hash 参数说明 -----------------------
# name：模型名称（如 'resnet50'），必须已登记在 _model_sha1 中
# 返回：该模型 SHA1 的前 8 位短哈希（用于构成缓存文件名）
def short_hash(name):  # 取模型权重 SHA1 的前 8 位短哈希
    name = _model_alias.get(name, name)  # 兼容旧接口名：resnet50 → resnet50s
    if name not in _model_sha1:  # 名称不在登记表中
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))  # 抛值错误：该模型无预训练权重记录
    return _model_sha1[name][:8]  # 截取 SHA1 字符串前 8 个字符


# ----------------------- get_resnet_file 参数说明 -----------------------
# name：模型名称，限 'resnet50' / 'resnet101' / 'resnet152'
# root：权重缓存根目录，默认 '~/.torch/models'
# 返回：本地 .pth 权重文件的完整路径（必要时先下载解压）
def get_resnet_file(name, root='~/.torch/models'):  # 获取 ResNet 骨干权重路径：查缓存 → SHA1 校验 → 必要时下载解压
    name = _model_alias.get(name, name)  # 兼容旧接口名：resnet50 → resnet50s（新版仓库中的 deep_stem 权重）
    file_name = '{name}-{short_hash}'.format(name=name, short_hash=short_hash(name))  # 拼接缓存文件名，形如 resnet50s-a75c83cf
    root = os.path.expanduser(root)  # 将 ~ 展开为用户主目录的绝对路径

    file_path = os.path.join(root, file_name + '.pth')  # 最终 .pth 权重文件的完整路径
    sha1_hash = _model_sha1[name]  # 取该模型对应的完整 SHA1 校验值
    if os.path.exists(file_path):  # 本地已存在权重文件
        if check_sha1(file_path, sha1_hash):  # SHA1 校验通过
            return file_path  # 直接返回缓存路径，免去下载
        else:  # 校验失败（文件损坏或不匹配）
            print('Mismatch in the content of model file {} detected.' +  # 【注意】bug：.format 只作用于下一行字符串，本行的 {} 占位符不会被填充，提示信息里不会出现文件名
                  ' Downloading again.'.format(file_path))  # 提示内容不匹配，将重新下载（格式化实际作用在无占位符的这段字符串上）

    # 兼容旧缓存：新版文件名不同，但旧权重与本工程 v1s 结构完全一致，本地有就直接复用
    legacy_name = _legacy_alias.get(name)  # 新权重名 → 旧缓存名（如 resnet50s → resnet50）
    if legacy_name is not None and legacy_name in _legacy_model_sha1:  # 该模型存在旧版权重记录
        legacy_file = os.path.join(root, '{}-{}.pth'.format(legacy_name, _legacy_model_sha1[legacy_name][:8]))  # 旧缓存文件名，如 resnet50-25c4b509.pth
        if os.path.exists(legacy_file) and check_sha1(legacy_file, _legacy_model_sha1[legacy_name]):  # 旧缓存存在且校验通过
            print('Using legacy cached model file {}'.format(legacy_file))  # 提示复用旧缓存
            return legacy_file  # 直接返回旧权重路径，免去下载

    if not os.path.exists(file_path):  # 新版缓存与旧版缓存都不可用
        print('Model file {} is not found. Downloading.'.format(file_path))  # 提示未找到缓存，开始下载

    if not os.path.exists(root):  # 缓存目录不存在
        os.makedirs(root)  # 递归创建缓存目录

    zip_file_path = os.path.join(root, file_name + '.zip')  # 临时 zip 压缩包的完整保存路径
    repo_url = os.environ.get('ENCODING_REPO', encoding_repo_url)  # 优先读取环境变量 ENCODING_REPO 覆盖默认仓库地址
    if repo_url[-1] != '/':  # 仓库地址末尾缺少斜杠
        repo_url = repo_url + '/'  # 补齐斜杠，保证 URL 模板拼接正确
    download(_url_format.format(repo_url=repo_url, file_name=file_name),  # 按 URL 模板拼出下载地址并下载权重 zip 包
             path=zip_file_path,  # 下载内容保存到本地 zip 路径
             overwrite=True)  # 覆盖已存在的同名 zip 文件
    with zipfile.ZipFile(zip_file_path) as zf:  # 打开下载好的 zip 压缩包（with 确保用完关闭）
        zf.extractall(root)  # 解压全部内容到缓存目录，得到 .pth 权重文件
    os.remove(zip_file_path)  # 删除临时 zip 包，节省磁盘空间

    if check_sha1(file_path, sha1_hash):  # 对解压出的 .pth 再做一次 SHA1 校验
        return file_path  # 校验通过，返回权重路径
    else:  # 校验失败
        raise ValueError('Downloaded file has different hash. Please try again.')  # 抛值错误：下载文件哈希不符，请重试


# ----------------------- get_model_file 参数说明 -----------------------
# name：分割模型权重名（如 'fcn_resnet50_pascal_voc'）
# root：权重缓存根目录，默认 '~/.torch/models'
# 返回：本地 .pth 权重文件完整路径；找不到则抛异常
def get_model_file(name, root='~/.torch/models'):  # 获取分割模型权重路径：仅本地查找，不负责下载
    root = os.path.expanduser(root)  # 展开 ~ 为用户主目录
    file_path = os.path.join(root, name + '.pth')  # 拼接 .pth 文件完整路径（name 需自带数据集后缀）
    if os.path.exists(file_path):  # 本地存在该权重文件
        return file_path  # 返回路径
    else:  # 本地不存在
        raise ValueError('Model file is not found. Downloading or trainning.')  # 【注意】直接抛异常且不提供下载（与模块 docstring 的"提供预训练模型"不符），另 'trainning' 为拼写错误（应为 training）
