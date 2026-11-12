# ============================================================================
# 【文件说明】mscoco.py —— MS COCO 2017 语义分割数据集的下载与解压脚本
# 下载来源：http://images.cocodataset.org（COCO 官方文件服务器），共三个 zip 包：
#   1) train2017.zip（118287 张训练图片）
#   2) annotations_trainval2017.zip（instances/captions/person_keypoints 标注 JSON）
#   3) val2017.zip（5000 张验证图片）
#   源码中另有两个被注释掉的包：stuff_annotations_trainval2017.zip（stuff 标注）与
#   test2017.zip（测试集，参赛用），需要时自行取消注释。
# 解压后的目录结构（--download-dir 指定，默认 ~/mscoco/）：
#   mscoco/
#   ├── train2017/*.jpg        # 训练图片
#   ├── val2017/*.jpg          # 验证图片
#   └── annotations/*.json     # 各任务标注文件
# 整体流程：检查目录完整性 → 下载解压 → 在 ~/.torch/datasets/coco 建符号链接 → 安装/导入 pycocotools
# 运行方式：python core/data/downloader/mscoco.py [--download-dir ~/mscoco] [--no-download] [--overwrite]
# ============================================================================
"""Prepare MS COCO datasets"""  # 原有模块 docstring 保留：准备 MS COCO 数据集
import os  # 标准库：路径与文件系统操作
import sys  # 标准库：把仓库根目录加入模块搜索路径
import argparse  # 标准库：命令行参数解析
import zipfile  # 标准库：zip 压缩包的读取与解压

# TODO: optim code
cur_path = os.path.abspath(os.path.dirname(__file__))  # 当前脚本所在目录的绝对路径
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]  # 连续三次取父目录，得到仓库根目录
sys.path.append(root_path)  # 保证可导入仓库根目录下的 core 包

from core.utils import download, makedirs, try_import_pycocotools  # 工具函数：download（带 sha1 的下载）、makedirs、try_import_pycocotools（尝试安装并导入 pycocotools）

_TARGET_DIR = os.path.expanduser('~/.torch/datasets/coco')  # 数据集默认安装目录（最终符号链接的位置）


# 形参说明: 无参数，配置全部来自命令行
def parse_args():  # 解析命令行参数
    parser = argparse.ArgumentParser(  # 创建参数解析器
        description='Initialize MS COCO dataset.',  # 工具描述
        epilog='Example: python mscoco.py --download-dir ~/mscoco',  # 帮助信息末尾的示例命令
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # help 中自动附上各参数默认值
    parser.add_argument('--download-dir', type=str, default='~/mscoco/', help='dataset directory on disk')  # 数据集根目录（默认 ~/mscoco/）
    parser.add_argument('--no-download', action='store_true', help='disable automatic download if set')  # 置位后禁用自动下载，只做本地检查
    parser.add_argument('--overwrite', action='store_true',  # 开关参数：置位后强制重新下载覆盖
                        help='overwrite downloaded files if set, in case they are corrupted')  # 用途：已下载文件损坏时覆盖重下
    args = parser.parse_args()  # 解析命令行
    return args  # 返回参数命名空间


# 形参说明: path 数据集根目录; overwrite 是否覆盖已存在的压缩包
def download_coco(path, overwrite=False):  # 下载 COCO 2017 的三个 zip 包并解压到 path
    _DOWNLOAD_URLS = [  # (下载 URL, sha1 校验和) 二元组列表
        ('http://images.cocodataset.org/zips/train2017.zip',  # 训练集图片压缩包（约 18GB）
         '10ad623668ab00c62c096f0ed636d6aff41faca5'),  # 对应的 sha1 校验和
        ('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',  # train/val 标注压缩包
         '8551ee4bb5860311e79dace7e79cb91e432e78b3'),  # 对应的 sha1 校验和
        ('http://images.cocodataset.org/zips/val2017.zip',  # 验证集图片压缩包
         '4950dc9d00dbe1c933ee0170f5797584351d2a41'),  # 对应的 sha1 校验和
        # ('http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip',
        # '46cdcf715b6b4f67e980b529534e79c2edffe084'),
        # test2017.zip, for those who want to attend the competition.
        # ('http://images.cocodataset.org/zips/test2017.zip',
        #  '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'),
    ]
    makedirs(path)  # 递归创建数据集根目录（已存在则跳过）
    for url, checksum in _DOWNLOAD_URLS:  # 依次处理每个压缩包
        filename = download(url, path=path, overwrite=overwrite, sha1_hash=checksum)  # 下载到 path（已存在且 sha1 匹配则复用），返回本地文件路径
        # extract
        with zipfile.ZipFile(filename) as zf:  # 打开 zip（省略了 "r"，默认即只读模式）
            zf.extractall(path=path)  # 将全部内容解压到 <path>


if __name__ == '__main__':  # 作为脚本直接运行时的主流程
    args = parse_args()  # 解析命令行参数
    path = os.path.expanduser(args.download_dir)  # 展开 ~ 得到数据集根目录
    # 【注意】反斜杠续行（续行上无法加注释）：若根目录、train2017/、val2017/、annotations/
    # 任一不存在，则进入下载分支
    if not os.path.isdir(path) or not os.path.isdir(os.path.join(path, 'train2017')) \
            or not os.path.isdir(os.path.join(path, 'val2017')) or not os.path.isdir(os.path.join(path, 'annotations')):  # val2017 / annotations 目录缺失时同样进入下载分支
        if args.no_download:  # 用户禁用了自动下载
            raise ValueError(('{} is not a valid directory, make sure it is present.'  # 目录不完整时抛 ValueError
                              ' Or you should not disable "--no-download" to grab it'.format(path)))  # 提示：不应禁用 --no-download，或先自行放好数据
        else:  # 未禁用下载：自动下载并解压缺失数据
            download_coco(path, overwrite=args.overwrite)  # 自动下载并解压缺失的数据

    # make symlink
    makedirs(os.path.expanduser('~/.torch/datasets'))  # 确保 ~/.torch/datasets 存在
    if os.path.isdir(_TARGET_DIR):  # 默认链接位置已被占用
        os.remove(_TARGET_DIR)  # 移除旧链接；【注意】若为真实目录会抛 IsADirectoryError（应先判断 islink 或改用 shutil.rmtree）
    os.symlink(path, _TARGET_DIR)  # 建立符号链接：~/.torch/datasets/coco → 实际数据目录
    try_import_pycocotools()  # 尝试导入 pycocotools，缺失时自动安装（COCO 标注解析的依赖）
