# ============================================================================
# 【文件说明】pascal_voc.py —— PASCAL VOC 语义分割数据集的下载与解压脚本
# 下载来源（Oxford 官方服务器 host.robots.ox.ac.uk，http 协议；该服务器经常不可用，
#   常用镜像为 pjreddie.com/media/files/）：
#   1) VOCtrainval_06-Nov-2007.tar —— VOC2007 训练+验证集
#   2) VOCtest_06-Nov-2007.tar     —— VOC2007 测试集
#   3) VOCtrainval_11-May-2012.tar —— VOC2012 训练+验证集
# 另有 VOC 增强分割集（SBD，Berkeley benchmark.tgz）由 download_aug() 下载；
#   原 www.eecs.berkeley.edu 链接已失效，现改用 www2 + https（torchvision 使用的地址）。
#   若 <download-dir>/benchmark.tgz 已存在且 sha1 匹配，download() 会直接复用本地包，不联网。
# 解压后的目录结构（--download-dir 指定，默认 ~/.torch/datasets/VOCdevkit）：
#   VOC2007/{JPEGImages, SegmentationClass, SegmentationObject, ImageSets}  # 由 VOCdevkit 拆出
#   VOC2012/{JPEGImages, SegmentationClass, SegmentationObject, ImageSets}  # 由 VOCdevkit 拆出
#   VOCaug/dataset/{img/*.jpg, cls/*.mat, train.txt, val.txt, trainval.txt} # SBD 增强集（train+val 合并）
# 流程：下载 tar → safe_extract 安全解压（防路径穿越）→ 重组目录 → 生成 trainval.txt → 建符号链接
# 运行方式：python core/data/downloader/pascal_voc.py [--download-dir ~/.torch/datasets/VOCdevkit] [--no-download] [--overwrite]
# ============================================================================
"""Prepare PASCAL VOC datasets"""  # 原有模块 docstring 保留：准备 PASCAL VOC 数据集
import os  # 标准库：路径与文件系统操作
import sys  # 标准库：把仓库根目录加入模块搜索路径
import shutil  # 标准库：目录移动（move）与递归删除（rmtree）
import argparse  # 标准库：命令行参数解析
import tarfile  # 标准库：tar/tar.gz 压缩包的读取与解压

# TODO: optim code
cur_path = os.path.abspath(os.path.dirname(__file__))  # 当前脚本所在目录的绝对路径
root_path = os.path.split(os.path.split(os.path.split(cur_path)[0])[0])[0]  # 连续三次取父目录，得到仓库根目录
sys.path.append(root_path)  # 保证可导入仓库根目录下的 core 包

from core.utils import download, makedirs  # 工具函数：download（带 sha1 校验的下载）、makedirs（递归建目录）

_TARGET_DIR = os.path.expanduser('~/.torch/datasets/voc')  # 数据集默认安装目录（符号链接位置）


# 形参说明: 无参数，配置全部来自命令行
def parse_args():  # 解析命令行参数
    parser = argparse.ArgumentParser(  # 创建参数解析器
        description='Initialize PASCAL VOC dataset.',  # 工具描述
        epilog='Example: python pascal_voc.py --download-dir ~/.torch/datasets/VOCdevkit',  # 帮助信息末尾的示例命令
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # help 中自动附上各参数默认值
    parser.add_argument('--download-dir', type=str, default='~/.torch/datasets/VOCdevkit/', help='dataset directory on disk')  # 数据集根目录（默认 ~/.torch/datasets/VOCdevkit/）
    parser.add_argument('--no-download', action='store_true', help='disable automatic download if set')  # 置位后禁用自动下载，只做本地检查
    parser.add_argument('--overwrite', action='store_true',  # 开关参数：置位后强制重新下载覆盖
                        help='overwrite downloaded files if set, in case they are corrupted')  # 用途：已下载文件损坏时覆盖重下
    args = parser.parse_args()  # 解析命令行
    return args  # 返回参数命名空间


#####################################################################################
# Download and extract VOC datasets into ``path``
# （中文补充）下面是 VOC2007/VOC2012 原始数据集的下载与解压函数

# 形参说明: path 数据集根目录; overwrite 是否覆盖已下载的压缩包
def download_voc(path, overwrite=False):  # 下载三个 VOC tar 包并安全解压到 path
    _DOWNLOAD_URLS = [  # (tar 包 URL, sha1 校验和) 二元组列表
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',  # VOC2007 训练+验证集
         '34ed68851bce2a36e2a223fa52c661d592c66b3c'),  # 对应的 sha1 校验和
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',  # VOC2007 测试集
         '41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1'),  # 对应的 sha1 校验和
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',  # VOC2012 训练+验证集
         '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')]  # 【注意】该 sha1 与 mscoco.py 中被注释的 test2017.zip 校验和完全相同，疑似复制粘贴错误
    makedirs(path)  # 递归创建目标目录（已存在则跳过）
    for url, checksum in _DOWNLOAD_URLS:  # 依次下载并解压每个 tar 包
        filename = download(url, path=path, overwrite=overwrite, sha1_hash=checksum)  # 下载（已存在且 sha1 匹配则跳过），返回本地文件路径
        # extract
        with tarfile.open(filename) as tar:  # 以上下文管理器打开 tar 包（自动关闭句柄）
            # ↓↓↓ 以下两个嵌套函数实现"安全解压"：先校验再解压，防止 tar 路径穿越攻击（CVE-2007-4559）
            # 形参说明: directory 解压根目录; target 待检查的成员解压路径
            def is_within_directory(directory, target):  # 判断 target 是否位于 directory 内部（防成员路径逃逸）

                abs_directory = os.path.abspath(directory)  # 解压根目录的绝对路径
                abs_target = os.path.abspath(target)  # 压缩包成员解压后的绝对路径

                prefix = os.path.commonprefix([abs_directory, abs_target])  # 两者的公共前缀；【注意】commonprefix 按字符而非路径分量比较，/dir 与 /dir_evil 也会得到前缀 /dir，存在误判可能（正确做法用 os.path.relpath/realpath 判断）

                return prefix == abs_directory  # 公共前缀恰为根目录则认为成员路径安全

            # 形参说明: tar 已打开的 tarfile 对象; path 解压目标目录; members 可选的成员子集; numeric_owner 仅按数字 uid/gid 还原属主
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):  # 逐一校验所有成员路径后再调用 extractall

                for member in tar.getmembers():  # 遍历压缩包内全部成员（文件/目录/链接）
                    member_path = os.path.join(path, member.name)  # 该成员解压后的完整路径
                    if not is_within_directory(path, member_path):  # 若成员路径逃逸出解压根目录
                        raise Exception("Attempted Path Traversal in Tar File")  # 抛通用异常中止，拒绝解压（防路径穿越）

                tar.extractall(path, members, numeric_owner=numeric_owner)  # 校验全部通过后，真正把（可选成员子集）解压到 path


            safe_extract(tar, path=path)  # 执行安全解压（解压结果为 <path>/VOCdevkit/VOC2007、VOC2012 等）


#####################################################################################
# Download and extract the VOC augmented segmentation dataset into ``path``
# （中文补充）下面是 VOC 增强分割数据集（SBD，Hariharan et al. 的 benchmark_RELEASE）的下载与解压函数

# 形参说明: path 数据集根目录; overwrite 是否覆盖已下载的压缩包
def download_aug(path, overwrite=False):  # 下载 SBD benchmark.tgz，解压并整理为 VOCaug 目录
    _AUG_DOWNLOAD_URLS = [  # (URL, sha1) 二元组列表
        ('https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz',  # SBD（Semantic Boundaries Dataset）压缩包；www 原站已失效，改用 www2 + https
         '7129e0a480c2d6afb02b517bb18ac54283bfaa35')]  # 对应的 sha1 校验和
    makedirs(path)  # 递归创建目标目录
    for url, checksum in _AUG_DOWNLOAD_URLS:  # 遍历下载列表
        filename = download(url, path=path, overwrite=overwrite, sha1_hash=checksum)  # 下载并校验 sha1，返回本地路径
        # extract
        with tarfile.open(filename) as tar:  # 打开 tgz 压缩包
            # 【注意】以下两个嵌套函数与 download_voc 中的定义完全重复（未抽成公共函数）
            # 形参说明: directory 解压根目录; target 待检查的成员解压路径
            def is_within_directory(directory, target):  # 判断 target 是否位于 directory 内部（防路径穿越）

                abs_directory = os.path.abspath(directory)  # 解压根目录的绝对路径
                abs_target = os.path.abspath(target)  # 成员解压后的绝对路径

                prefix = os.path.commonprefix([abs_directory, abs_target])  # 公共前缀（按字符比较，存在理论误判）

                return prefix == abs_directory  # 前缀等于根目录则安全

            # 形参说明: tar 已打开的 tarfile 对象; path 解压目标目录; members 可选成员子集; numeric_owner 仅按数字 uid/gid 还原属主
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):  # 先校验全部成员再解压

                for member in tar.getmembers():  # 遍历压缩包全部成员
                    member_path = os.path.join(path, member.name)  # 成员解压路径
                    if not is_within_directory(path, member_path):  # 检测路径穿越
                        raise Exception("Attempted Path Traversal in Tar File")  # 拒绝解压并抛异常

                tar.extractall(path, members, numeric_owner=numeric_owner)  # 校验通过后执行解压


            safe_extract(tar, path=path)  # 安全解压（解压结果为 <path>/benchmark_RELEASE）
            shutil.move(os.path.join(path, 'benchmark_RELEASE'),  # 把解压出的 benchmark_RELEASE 目录
                        os.path.join(path, 'VOCaug'))  # 移动/重命名为 <path>/VOCaug
            filenames = ['VOCaug/dataset/train.txt', 'VOCaug/dataset/val.txt']  # SBD 自带的训练/验证列表文件（相对 path）
            # generate trainval.txt
            with open(os.path.join(path, 'VOCaug/dataset/trainval.txt'), 'w') as outfile:  # 以写模式创建 trainval.txt
                for fname in filenames:  # 依次处理 train.txt 与 val.txt
                    fname = os.path.join(path, fname)  # 拼接为绝对路径
                    with open(fname) as infile:  # 以读模式打开列表文件
                        for line in infile:  # 逐行读取样本编号
                            outfile.write(line)  # 追加写入 trainval.txt（合并训练+验证列表）


if __name__ == '__main__':  # 作为脚本直接运行时的主流程
    args = parse_args()  # 解析命令行参数
    path = os.path.expanduser(args.download_dir)  # 展开 ~ 得到数据集根目录（默认 ~/.torch/datasets/VOCdevkit/）
    # 【已修复】原代码首个条件用 os.path.isfile(path) 判断一个"目录"，恒为 False，
    # 导致即使 VOC2007/VOC2012 都已就绪也每次重复解压、shutil.move 到已存在目录而报错，
    # 且配合 --no-download 会必然抛 ValueError。改为只检查两个数据目录是否就绪。
    if not os.path.isdir(os.path.join(path, 'VOC2007')) \
    or not os.path.isdir(os.path.join(path, 'VOC2012')):  # 任一缺失才触发下载
        if args.no_download:  # 用户禁用了自动下载
            raise ValueError(('{} is not a valid directory, make sure it is present.'  # 目录不完整时抛 ValueError
                ' Or you should not disable "--no-download" to grab it'.format(path)))  # 提示：先自行放好数据或放开下载开关
        else:  # 未禁用下载：下载并解压三个 VOC tar 包
            download_voc(path, overwrite=args.overwrite)  # 下载并解压三个 VOC tar 包（得到 <path>/VOCdevkit/...）
            shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2007'), os.path.join(path, 'VOC2007'))  # 把 VOC2007 从 VOCdevkit 中移出，平铺到根目录
            shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2012'), os.path.join(path, 'VOC2012'))  # 同样移出 VOC2012
            shutil.rmtree(os.path.join(path, 'VOCdevkit'))  # 删除移空后的 VOCdevkit 壳目录

    if not os.path.isdir(os.path.join(path, 'VOCaug')):  # VOC 增强集（SBD）尚未就绪
        if args.no_download:  # 用户禁用了自动下载
            raise ValueError(('{} is not a valid directory, make sure it is present.'  # 抛 ValueError 提示目录缺失
                ' Or you should not disable "--no-download" to grab it'.format(path)))  # 提示：先自行放好数据或放开下载开关
        else:  # 未禁用下载：下载并解压 SBD 增强集
            download_aug(path, overwrite=args.overwrite)  # 下载并解压 SBD，整理出 <path>/VOCaug

    # make symlink
    makedirs(os.path.expanduser('~/.torch/datasets'))  # 确保 ~/.torch/datasets 存在
    if os.path.lexists(_TARGET_DIR):  # 默认链接位置已被占用（含失效的悬空链接）
        if os.path.islink(_TARGET_DIR):  # 仅当是符号链接时才安全删除
            os.remove(_TARGET_DIR)
        else:  # 真实目录：绝不自动删除其中数据，交给用户手动处理
            raise RuntimeError('{} 已存在且不是符号链接，请手动备份/移除后重试'.format(_TARGET_DIR))
    os.symlink(path, _TARGET_DIR)  # 建立符号链接：~/.torch/datasets/voc → 实际数据目录
