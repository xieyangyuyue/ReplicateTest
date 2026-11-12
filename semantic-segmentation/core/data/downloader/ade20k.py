# ============================================================================
# 【文件说明】ade20k.py —— ADE20K（MIT Places2 Challenge）语义分割数据集的下载与解压脚本
# 下载来源：MIT CSAIL 服务器（data.csail.mit.edu/places/ADEchallenge/），共两个 zip 包：
#   1) ADEChallengeData2016.zip —— 训练/验证集（20000 张训练 + 2000 张验证，150 个场景类别）
#   2) release_test.zip         —— 官方测试集（无公开标注）
# 解压后的目录结构（默认安装到 ~/.torch/datasets/ade，即 _TARGET_DIR）：
#   ade/
#   ├── ADEChallengeData2016/
#   │   ├── images/training|validation/*.jpg          # 原图
#   │   └── annotations/training|validation/*.png      # 语义标注图
#   ├── release_test/...                               # 测试集图片
#   └── downloads/                                     # 缓存的 zip 压缩包
# 运行方式：python core/data/downloader/ade20k.py [--download-dir 自定义数据集目录]
# ============================================================================
"""Prepare ADE20K dataset"""  # 原有模块 docstring 保留：准备 ADE20K 数据集
import os  # 标准库：路径拼接、目录/文件判断等文件系统操作
import sys  # 标准库：修改 sys.path 以便导入仓库根目录下的 core 包
import argparse  # 标准库：命令行参数解析
import zipfile  # 标准库：zip 压缩包的读取（ZipFile）与解压（extractall）

# TODO: optim code
cur_path = os.path.abspath(os.path.dirname(__file__))  # 当前脚本所在目录（core/data/downloader）的绝对路径
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]  # 连续三次取父目录：downloader → data → core → 仓库根目录
sys.path.append(root_path)  # 把仓库根目录追加到模块搜索路径，保证能 import core

from core.utils import download, makedirs  # 复用工具函数：download（带 sha1 校验的下载）、makedirs（递归建目录）

_TARGET_DIR = os.path.expanduser('~/.torch/datasets/ade')  # 数据集默认安装目录（展开 ~）


# 形参说明: 无参数，配置全部来自命令行
def parse_args():  # 解析命令行参数
    parser = argparse.ArgumentParser(  # 创建参数解析器
        description='Initialize ADE20K dataset.',  # 工具描述（帮助信息首行）
        epilog='Example: python setup_ade20k.py',  # 帮助信息末尾的示例命令（沿用了旧脚本名 setup_ade20k.py）
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 该格式器会自动在 help 中附上各参数默认值
    parser.add_argument('--download-dir', default=None, help='dataset directory on disk')  # 可选参数：数据集在磁盘上的自定义目录
    args = parser.parse_args()  # 解析命令行并得到参数命名空间
    return args  # 返回参数对象


# 形参说明: path 数据集安装目录; overwrite 是否覆盖已下载的压缩包（默认 False）
def download_ade(path, overwrite=False):  # 下载 ADE20K 的两个 zip 包并解压到 path
    _AUG_DOWNLOAD_URLS = [  # (下载 URL, sha1 校验和) 二元组列表
        ('http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip',  # 训练+验证集压缩包（150 类）
         '219e1696abb36c8ba3a3afe7fb2f4b4606a897c7'),  # 该包对应的 sha1 校验和，用于完整性校验
        (  # 第二个 (URL, sha1) 二元组起始行：测试集压缩包及其校验和（换行书写）
            'http://data.csail.mit.edu/places/ADEchallenge/release_test.zip',  # 测试集压缩包（无标注）
            'e05747892219d10e9243933371a497e905a4860c'), ]  # 对应 sha1；【注意】官方源为 http 明文协议，下载失败时可尝试改用 https
    download_dir = os.path.join(path, 'downloads')  # 压缩包缓存目录：<path>/downloads
    makedirs(download_dir)  # 递归创建缓存目录（已存在则不报错）
    for url, checksum in _AUG_DOWNLOAD_URLS:  # 遍历待下载文件列表
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)  # 下载到缓存目录（本地已存在且 sha1 匹配则跳过重下），返回本地文件路径
        # extract
        with zipfile.ZipFile(filename, "r") as zip_ref:  # 以只读模式打开 zip 压缩包（with 保证用完自动关闭句柄）
            zip_ref.extractall(path=path)  # 将压缩包内全部内容解压到目标目录 <path> 下


if __name__ == '__main__':  # 作为脚本直接运行时执行主流程
    args = parse_args()  # 解析命令行参数
    makedirs(os.path.expanduser('~/.torch/datasets'))  # 确保 ~/.torch/datasets 父目录存在
    if args.download_dir is not None:  # 用户指定了自定义数据集目录
        if os.path.isdir(_TARGET_DIR):  # 默认安装位置已被占用（真实目录或指向目录的符号链接均返回 True）
            os.remove(_TARGET_DIR)  # 删除旧链接；【注意】若 _TARGET_DIR 是真实目录而非符号链接，os.remove 会抛 IsADirectoryError，严格来说应先判断 islink 或改用 shutil.rmtree
        # make symlink
        os.symlink(args.download_dir, _TARGET_DIR)  # 建立符号链接：~/.torch/datasets/ade → 自定义目录
    download_ade(_TARGET_DIR, overwrite=False)  # 下载并解压数据集到 _TARGET_DIR（经符号链接落入自定义目录；不覆盖已有文件）
