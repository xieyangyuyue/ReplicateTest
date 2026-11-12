# ============================================================================
# 【文件说明】sbu_shadow.py —— SBU Shadow（Stony Brook 阴影分割）数据集的下载与解压脚本
# 下载来源：http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip
#   （4085 张含阴影照片及配套掩码，用于阴影/前景二值分割）
# 解压后的目录结构（默认安装到 ~/.torch/datasets/sbu）：
#   sbu/
#   ├── SBU-shadow/…      # 解压出的数据集内容（图片与阴影掩码）
#   └── SBU-shadow.zip    # 下载的压缩包（直接存放在 <path> 根下）
# 运行方式：python core/data/downloader/sbu_shadow.py [--download-dir 自定义目录] [--no-download] [--overwrite]
# ============================================================================
"""Prepare SBU Shadow datasets"""  # 原有模块 docstring 保留：准备 SBU Shadow 数据集
import os  # 标准库：路径与文件系统操作
import sys  # 标准库：把仓库根目录加入模块搜索路径
import argparse  # 标准库：命令行参数解析
import zipfile  # 标准库：zip 压缩包的读取与解压

# TODO: optim code
cur_path = os.path.abspath(os.path.dirname(__file__))  # 当前脚本所在目录的绝对路径
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]  # 连续三次取父目录，得到仓库根目录
sys.path.append(root_path)  # 保证可导入仓库根目录下的 core 包

from core.utils import download, makedirs  # 工具函数：download（下载）、makedirs（递归建目录）

_TARGET_DIR = os.path.expanduser('~/.torch/datasets/sbu')  # 数据集默认安装目录（符号链接位置）


# 形参说明: 无参数，配置全部来自命令行
def parse_args():  # 解析命令行参数
    parser = argparse.ArgumentParser(  # 创建参数解析器
        description='Initialize SBU Shadow dataset.',  # 工具描述
        epilog='Example: python sbu_shadow.py --download-dir ~/SBU-shadow',  # 帮助信息末尾的示例命令
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # help 中自动附上各参数默认值
    parser.add_argument('--download-dir', type=str, default=None, help='dataset directory on disk')  # 可选：数据集在磁盘上的自定义目录
    parser.add_argument('--no-download', action='store_true', help='disable automatic download if set')  # 置位后禁用自动下载
    parser.add_argument('--overwrite', action='store_true',  # 开关参数：置位后强制重新下载覆盖
                        help='overwrite downloaded files if set, in case they are corrupted')  # 用途：已下载文件损坏时覆盖重下
    args = parser.parse_args()  # 解析命令行
    return args  # 返回参数命名空间


#####################################################################################
# Download and extract SBU shadow datasets into ``path``
# （中文补充）SBU Shadow 数据集的下载与解压函数

# 形参说明: path 数据集安装目录; overwrite 是否覆盖已下载文件
def download_sbu(path, overwrite=False):  # 下载 SBU-shadow.zip 并解压到 path
    _DOWNLOAD_URLS = [  # 下载 URL 列表；【注意】与其它脚本不同，这里没有 sha1 校验和，下载不做完整性校验
        ('http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip'),  # 数据集唯一压缩包；【注意】括号内只有单个字符串时并不是元组，此写法等价于裸字符串
    ]  # URL 列表定义结束
    download_dir = os.path.join(path, 'downloads')  # 预留的压缩包缓存目录
    makedirs(download_dir)  # 递归创建缓存目录
    for url in _DOWNLOAD_URLS:  # 遍历 URL 列表
        filename = download(url, path=path, overwrite=overwrite)  # 【注意】下载目录传的是 path 而非上面创建的 download_dir，downloads/ 实际未被使用（与 ade20k.py 不一致）
        # extract
        with zipfile.ZipFile(filename, "r") as zf:  # 以只读模式打开 zip
            zf.extractall(path=path)  # 将全部内容解压到 <path>
        print("Extracted", filename)  # 打印解压完成日志


if __name__ == '__main__':  # 作为脚本直接运行时的主流程
    args = parse_args()  # 解析命令行参数
    makedirs(os.path.expanduser('~/.torch/datasets'))  # 确保 ~/.torch/datasets 存在
    if args.download_dir is not None:  # 用户指定了自定义数据集目录
        if os.path.isdir(_TARGET_DIR):  # 默认安装位置已被占用
            os.remove(_TARGET_DIR)  # 删除旧链接；【注意】若为真实目录会抛 IsADirectoryError
        # make symlink
        os.symlink(args.download_dir, _TARGET_DIR)  # 建立符号链接：~/.torch/datasets/sbu → 自定义目录
    else:  # 未指定自定义目录：下载并解压到默认目录
        download_sbu(_TARGET_DIR, overwrite=False)  # 未指定目录时：下载并解压到默认目录（不覆盖已有文件）
