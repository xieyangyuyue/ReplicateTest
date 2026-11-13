# ============================================================================
# 【文件说明】pascal_voc.py —— PASCAL VOC 2012 语义分割数据集的下载与解压脚本
# 下载来源（Oxford 官方服务器已迁移，旧的 host.robots.ox.ac.uk 会重定向到
#   thor.robots.ox.ac.uk，因此这里直接使用 https 的 thor 地址；该服务器偶尔不可用，
#   另提供 pjreddie.com 镜像作为备选）：
#   1) VOCtrainval_11-May-2012.tar —— VOC2012 训练+验证集（本脚本只使用该数据集）
# 可选增强数据集（SBD，Berkeley benchmark.tgz）由 download_aug() 下载；
#   原 www.eecs.berkeley.edu 链接已失效，现改用 www2 + https（torchvision 使用的地址）。
#   默认不下载增强集，需显式加 --aug 开启（--no-aug 可显式关闭）。
#   若 <download-dir>/benchmark.tgz 已存在且 sha1 匹配，download() 会直接复用本地包，不联网。
# 解压后的目录结构（--download-dir 指定，默认 ~/.torch/datasets/VOCdevkit）：
#   VOC2012/{JPEGImages, SegmentationClass, SegmentationObject, ImageSets}  # 由 VOCdevkit 拆出
#   VOCaug/dataset/{img/*.jpg, cls/*.mat, train.txt, val.txt, trainval.txt} # SBD 增强集（train+val 合并，仅 --aug 时）
# 流程：下载 tar → safe_extract 安全解压（防路径穿越）→ 重组目录 → 生成 trainval.txt → 建符号链接
# 符号链接（落在本项目内，供 dataloader 默认 root='../datasets/voc' 使用）：
#   <项目根>/datasets/voc -> <download-dir>（默认 ~/.torch/datasets/VOCdevkit）
# 运行方式：python core/data/downloader/pascal_voc.py [--download-dir ~/.torch/datasets/VOCdevkit] [--no-download] [--overwrite] [--aug]
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

_TARGET_DIR = os.path.join(root_path, 'datasets', 'voc')  # 项目内符号链接位置：<项目根>/datasets/voc（由脚本路径推导，不依赖当前工作目录）


# 形参说明: 无参数，配置全部来自命令行
def parse_args():  # 解析命令行参数
    parser = argparse.ArgumentParser(  # 创建参数解析器
        description='Initialize PASCAL VOC 2012 dataset (optionally with the augmented SBD set).',  # 工具描述
        epilog='Example: python pascal_voc.py --download-dir ~/.torch/datasets/VOCdevkit --aug',  # 帮助信息末尾的示例命令
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # help 中自动附上各参数默认值
    parser.add_argument('--download-dir', type=str, default='~/.torch/datasets/VOCdevkit/', help='dataset directory on disk')  # 数据集根目录（默认 ~/.torch/datasets/VOCdevkit/）
    parser.add_argument('--no-download', action='store_true', help='disable automatic download if set')  # 置位后禁用自动下载，只做本地检查
    parser.add_argument('--overwrite', action='store_true',  # 开关参数：置位后强制重新下载覆盖
                        help='overwrite downloaded files if set, in case they are corrupted')  # 用途：已下载文件损坏时覆盖重下
    parser.add_argument('--aug', dest='aug', action='store_true',  # 开关参数：置位后额外下载 VOCaug（SBD 增强集）
                        help='download the augmented VOC dataset (SBD/VOCaug) if set')  # 只有显式开启才会下载增强集
    parser.add_argument('--no-aug', dest='aug', action='store_false',  # 显式关闭增强集（默认即为关闭，保留该参数便于脚本书写）
                        help='do not download the augmented VOC dataset (SBD/VOCaug)')  # 与 --aug 互斥，后出现的生效
    parser.set_defaults(aug=False)  # 默认只准备 VOC2012，不下载 SBD 增强集
    args = parser.parse_args()  # 解析命令行
    return args  # 返回参数命名空间


#####################################################################################
# Download and extract VOC datasets into ``path``
# （中文补充）下面是 VOC2012 原始数据集的下载与解压函数

# 形参说明: path 数据集根目录; overwrite 是否覆盖已下载的压缩包
def download_voc(path, overwrite=False):  # 下载 VOC2012 tar 包并安全解压到 path
    _VOC2012_SHA1 = '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'  # VOC2012 trainval 包的 sha1 校验和
    _DOWNLOAD_URLS = [  # 官方源 + 镜像源，按顺序尝试，任一可用即可
        'https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',  # Oxford 官方直链（host.robots.ox.ac.uk 已重定向到此处）
        'https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar',  # 镜像：文件同名、大小一致
    ]
    makedirs(path)  # 递归创建目标目录（已存在则跳过）
    filename = None  # 下载成功后的本地文件路径
    errors = []  # 记录每个源失败的原因，便于最终报错
    for url in _DOWNLOAD_URLS:  # 依次尝试各下载源
        try:
            filename = download(url, path=path, overwrite=overwrite, sha1_hash=_VOC2012_SHA1)  # 下载（已存在且 sha1 匹配则跳过），返回本地文件路径
            break  # 下载成功即跳出循环
        except Exception as e:  # 网络失败或哈希不匹配：换下一个镜像重试
            errors.append('  {}: {}'.format(url, e))  # 记录失败原因
            print('Download failed from {}, trying next mirror...'.format(url))  # 提示切换到下一个镜像
    if filename is None:  # 所有源都失败
        raise RuntimeError('Failed to download VOC2012 dataset from all mirrors:\n{}'.format('\n'.join(errors)))  # 汇总报错
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


        safe_extract(tar, path=path)  # 执行安全解压（解压结果为 <path>/VOCdevkit/VOC2012）


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
    # 导致即使 VOC2012 已就绪也每次重复解压、shutil.move 到已存在目录而报错，
    # 且配合 --no-download 会必然抛 ValueError。改为只检查数据目录是否就绪。
    if not os.path.isdir(os.path.join(path, 'VOC2012')):  # VOC2012 缺失才触发下载
        if args.no_download:  # 用户禁用了自动下载
            raise ValueError(('{} is not a valid directory, make sure it is present.'  # 目录不完整时抛 ValueError
                ' Or you should not disable "--no-download" to grab it'.format(path)))  # 提示：先自行放好数据或放开下载开关
        else:  # 未禁用下载：下载并解压 VOC2012 tar 包
            download_voc(path, overwrite=args.overwrite)  # 下载并解压 VOC2012 tar 包（得到 <path>/VOCdevkit/VOC2012）
            shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2012'), os.path.join(path, 'VOC2012'))  # 把 VOC2012 从 VOCdevkit 中移出，平铺到根目录
            if os.path.isdir(os.path.join(path, 'VOCdevkit')):  # 移空后的 VOCdevkit 壳目录可能仍存在
                shutil.rmtree(os.path.join(path, 'VOCdevkit'))  # 删除空壳目录（先判断避免不存在时报错）

    if args.aug and not os.path.isdir(os.path.join(path, 'VOCaug')):  # 仅当用户通过 --aug 开启，且 VOCaug 尚未就绪
        if args.no_download:  # 用户禁用了自动下载
            raise ValueError(('{} is not a valid directory, make sure it is present.'  # 抛 ValueError 提示目录缺失
                ' Or you should not disable "--no-download" to grab it'.format(path)))  # 提示：先自行放好数据或放开下载开关
        else:  # 未禁用下载：下载并解压 SBD 增强集
            download_aug(path, overwrite=args.overwrite)  # 下载并解压 SBD，整理出 <path>/VOCaug

    # make symlink：<项目根>/datasets/voc -> 实际下载目录
    makedirs(os.path.dirname(_TARGET_DIR))  # 确保项目内的 datasets 目录存在（不存在则创建）
    link_target = os.path.abspath(os.path.expanduser(path))  # 链接目标必须用绝对路径；否则相对路径会相对“链接所在目录”解析而失效
    if os.path.lexists(_TARGET_DIR):  # 链接位置已被占用（含失效的悬空链接）
        if os.path.islink(_TARGET_DIR):  # 仅当是符号链接时才安全删除
            os.remove(_TARGET_DIR)  # 移除旧链接，稍后重建
        else:  # 真实目录/普通文件：绝不自动删除其中数据，交给用户手动处理
            raise RuntimeError('{} 已存在且不是符号链接，请手动备份/移除后重试'.format(_TARGET_DIR))
    os.symlink(link_target, _TARGET_DIR)  # 建立符号链接：<项目根>/datasets/voc → 实际数据目录
    print('Created symlink: {} -> {}'.format(_TARGET_DIR, link_target))  # 打印链接结果，便于确认落点
