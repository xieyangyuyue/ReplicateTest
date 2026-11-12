# ============================================================================
# 【文件说明】cityscapes.py —— Cityscapes 街景语义分割数据集的校验与解压脚本
# 【注意】本脚本【不提供下载 URL】：Cityscapes 需在官网（www.cityscapes-dataset.com）
#   注册后手动下载两个 zip 包，并放到运行脚本时的工作目录（CWD）下：
#   1) gtFine_trainvaltest.zip       —— 精细标注（34 类，含忽略标签 255）
#   2) leftImg8bit_trainvaltest.zip  —— 原始图像
#   脚本只负责：sha1 校验 → zipfile 解压 → （可选）建立符号链接。
# 解压后的目录结构（默认安装到 ~/.torch/datasets/citys）：
#   citys/
#   ├── gtFine/{train,val,test}/<城市名>/*_gtFine_labelIds.png 等标注文件
#   ├── leftImg8bit/{train,val,test}/<城市名>/*_leftImg8bit.png 原图
#   └── downloads/                                          # 脚本创建但未实际使用的缓存目录
# 运行方式：python core/data/downloader/cityscapes.py [--download-dir 自定义数据集目录]
# ============================================================================
"""Prepare Cityscapes dataset"""  # 原有模块 docstring 保留：准备 Cityscapes 数据集
import os  # 标准库：路径与文件系统操作
import sys  # 标准库：把仓库根目录加入模块搜索路径
import argparse  # 标准库：命令行参数解析
import zipfile  # 标准库：zip 压缩包的读取与解压

# TODO: optim code
cur_path = os.path.abspath(os.path.dirname(__file__))  # 当前脚本所在目录的绝对路径
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]  # 连续三次取父目录，得到仓库根目录
sys.path.append(root_path)  # 保证可导入仓库根目录下的 core 包

from core.utils import download, makedirs, check_sha1  # 工具函数：download（本函数未实际使用）、makedirs、check_sha1（sha1 校验）

_TARGET_DIR = os.path.expanduser('~/.torch/datasets/citys')  # 数据集默认安装目录（符号链接位置）


# 形参说明: 无参数，配置全部来自命令行
def parse_args():  # 解析命令行参数
    parser = argparse.ArgumentParser(  # 创建参数解析器
        description='Initialize ADE20K dataset.',  # 【注意】复制粘贴错误：文案写的是 ADE20K，实际应为 Cityscapes
        epilog='Example: python prepare_cityscapes.py',  # 帮助信息末尾的示例命令
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # help 中自动附上各参数默认值
    parser.add_argument('--download-dir', default=None, help='dataset directory on disk')  # 可选参数：数据集在磁盘上的自定义目录
    args = parser.parse_args()  # 解析命令行
    return args  # 返回参数命名空间


# 形参说明: path 数据集安装目录; overwrite 是否覆盖（本函数体内并未使用该参数）
def download_city(path, overwrite=False):  # 校验并解压手动下载好的 Cityscapes 两个 zip 包
    _CITY_DOWNLOAD_URLS = [  # 【注意】列表元素是"本地文件名"而非 URL：(zip 文件名, 期望 sha1)
        ('gtFine_trainvaltest.zip', '99f532cb1af174f5fcc4c5bc8feea8c66246ddbc'),  # 精细标注包及其 sha1 校验和
        ('leftImg8bit_trainvaltest.zip', '2c0b77ce9933cc635adda307fbba5566f5d9d404')]  # 原图包及其 sha1 校验和
    download_dir = os.path.join(path, 'downloads')  # 预留的压缩包缓存目录（本函数创建了它却未真正使用）
    makedirs(download_dir)  # 递归创建该目录
    for filename, checksum in _CITY_DOWNLOAD_URLS:  # 遍历两个待处理的 zip 包
        # 【注意】check_sha1 直接按"当前工作目录"下的相对文件名校验：既没有下载步骤、
        # 也没有拼接 download_dir 路径，因此两个 zip 必须事先手动放到 CWD；
        # 校验失败（包括文件不存在）时抛 UserWarning——它是 Warning 子类而非常规异常，用法不规范
        if not check_sha1(filename, checksum):  # 校验本地文件的 sha1 是否与期望值一致
            # sha1 校验失败：抛 UserWarning（提示串以反斜杠续行跨四行拼接，续行行尾无法加注释）
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              # 提示串第 2 段：仓库可能过期或本次下载不完整
                              'The repo may be outdated or download may be incomplete. ' \
                              # 提示串第 3 段：若覆盖过默认 repo_url 下载源，建议切回默认源
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(filename))  # 提示：文件哈希不匹配（可能下载不完整或版本不符）
        # extract
        with zipfile.ZipFile(filename, "r") as zip_ref:  # 以只读模式打开 zip（文件须位于当前工作目录）
            zip_ref.extractall(path=path)  # 将全部内容解压到安装目录 <path>
        print("Extracted", filename)  # 打印解压完成日志


if __name__ == '__main__':  # 作为脚本直接运行时的主流程
    args = parse_args()  # 解析命令行参数
    makedirs(os.path.expanduser('~/.torch/datasets'))  # 确保 ~/.torch/datasets 存在
    if args.download_dir is not None:  # 用户指定了自定义数据集目录
        if os.path.isdir(_TARGET_DIR):  # 默认安装位置已被占用
            os.remove(_TARGET_DIR)  # 删除旧链接；【注意】若为真实目录会抛 IsADirectoryError
        # make symlink
        os.symlink(args.download_dir, _TARGET_DIR)  # 建立符号链接：~/.torch/datasets/citys → 自定义目录
    else:  # 未指定自定义目录：校验并解压到默认安装目录
        download_city(_TARGET_DIR, overwrite=False)  # 未指定目录时：校验并解压到默认目录
