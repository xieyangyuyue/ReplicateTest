# ============================================================================
# 【文件说明】utils.py —— 数据集下载/校验/解压的通用工具函数集（源自
#             torchvision 的 dataset 工具风格）：下载进度条回调
#             gen_bar_updater、MD5 完整性校验 check_integrity、建目录
#             makedir_exist_ok、断点复用下载 download_url、防路径穿越的
#             tar 安全解压 download_extract。供 datasets/ 目录下的数据
#             准备脚本（如 ade20k.py、cityscapes.py）调用。
# 【数据集】本文件不绑定具体数据集，是所有数据集准备脚本的公共工具模块。
# ============================================================================
import os  # 标准库：路径处理（expanduser/isfile/join/makedirs/abspath/commonprefix）
import hashlib  # 标准库：计算文件 MD5 校验和
import errno  # 标准库：错误码常量（用于识别 EEXIST“目录已存在”错误）
import tarfile  # 标准库：tar 归档的读取与解压
from six.moves import urllib  # 【注意】six 为 Python2/3 兼容库，官方已停止维护（废弃 API）；此导入等价于 import urllib.request
from torch.utils.model_zoo import tqdm  # 【注意】借道 torch.utils.model_zoo 获取 tqdm 进度条；新版 torch 已移除该入口，属于脆弱/废弃导入，稳妥写法是 from tqdm import tqdm

def gen_bar_updater():  # 构造下载进度条回调函数（供 urlretrieve 的 reporthook 参数使用）
    pbar = tqdm(total=None)  # 创建总长度未知的 tqdm 进度条（total=None 表示稍后动态设定）

    # urlretrieve 的回调签名：(count, block_size, total_size) 分别为已下载块数、每块字节数、文件总字节数
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:  # 首次回调且服务器返回了总大小
            pbar.total = total_size  # 把进度条总长度设为文件总字节数
        progress_bytes = count * block_size  # 已下载字节数 = 已下载块数 × 每块大小
        pbar.update(progress_bytes - pbar.n)  # 按增量刷新进度条（pbar.n 为当前已显示的字节数）

    return bar_update  # 返回闭包回调函数

def check_integrity(fpath, md5=None):  # 校验文件完整性；fpath 文件路径，md5 期望的十六进制校验和（可选）
    if md5 is None:  # 未提供校验和
        return True  # 无需校验，直接视为通过
    if not os.path.isfile(fpath):  # 文件不存在
        return False  # 校验失败
    md5o = hashlib.md5()  # 创建 MD5 哈希对象
    with open(fpath, 'rb') as f:  # 二进制只读打开文件
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):  # 每次读 1MB，直到读到空字节串为止（避免大文件一次性载入内存）
            md5o.update(chunk)  # 累加更新哈希值
    md5c = md5o.hexdigest()  # 计算最终的十六进制 MD5 校验和
    if md5c != md5:  # 与期望值不一致
        return False  # 校验失败（文件损坏或不匹配）
    return True  # 校验通过

def makedir_exist_ok(dirpath):  # 递归创建目录（已存在则忽略），等价于 os.makedirs(..., exist_ok=True)
    try:  # 尝试创建
        os.makedirs(dirpath)  # 递归创建多级目录
    except OSError as e:  # 创建过程中发生 OS 错误
        if e.errno == errno.EEXIST:  # 错误原因是目录已存在
            pass  # 已存在视为成功，静默忽略
        else:  # 其他错误（如权限不足、磁盘满）
            pass  # 【注意】此处同样静默 pass，非 EEXIST 错误被吞掉（目录实际未创建却无任何提示），属于可疑写法

def download_url(url, root, filename=None, md5=None):  # 从 url 下载文件到 root 目录；filename 目标文件名（缺省取 url 末段），md5 用于完整性校验
    """Download a file from a url and place it in root."""  # 原英文 docstring：从 url 下载文件并保存到 root 目录
    root = os.path.expanduser(root)  # 把路径中的 ~ 展开为用户主目录
    if not filename:  # 未指定文件名
        filename = os.path.basename(url)  # 用 url 的最后一段作为文件名
    fpath = os.path.join(root, filename)  # 目标文件的完整保存路径

    makedir_exist_ok(root)  # 确保 root 目录存在（不存在则创建）

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):  # 文件已存在且 MD5 校验通过
        print('Using downloaded and verified file: ' + fpath)  # 提示复用已下载文件，跳过下载
    else:  # 需要下载
        try:  # 先尝试按原 url 下载
            print('Downloading ' + url + ' to ' + fpath)  # 打印下载信息
            urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())  # 下载文件并挂进度条回调（【注意】urlretrieve 属于 urllib 的旧式 legacy 接口）
        except OSError:  # 下载失败（如 https 握手问题）
            if url[:5] == 'https':  # 原 url 是 https 协议
                url = url.replace('https:', 'http:')  # 降级为 http 协议重试
                print('Failed download. Trying https -> http instead.'  # 提示 https 失败后降级为 http 重试（与下一行字符串隐式拼接）
                      ' Downloading ' + url + ' to ' + fpath)  # 打印降级后的下载信息
                urllib.request.urlretrieve(url, fpath, reporthook=gen_bar_updater())  # 用 http 链接再次下载

def download_extract(url, root, filename, md5):  # 下载 tar 压缩包并解压到 root；url 下载地址，filename 压缩包文件名，md5 校验和
    download_url(url, root, filename, md5)  # 先下载（文件已存在且校验通过则直接复用）
    with tarfile.open(os.path.join(root, filename), "r") as tar:  # 打开 tar 归档（"r" 模式自动识别压缩格式）
        # 判断 target 路径是否位于 directory 目录内（防 tar 路径穿越攻击的安全检查）
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)  # directory 的绝对路径
            abs_target = os.path.abspath(target)  # target 的绝对路径
        
            prefix = os.path.commonprefix([abs_directory, abs_target])  # 【注意】commonprefix 是字符级前缀而非路径组件级比较（如 /a/b 与 /a/bc 也会误判同前缀），安全检查可被绕过，更稳妥做法是用 os.path.realpath 比较
            
            return prefix == abs_directory  # 公共前缀恰为目录路径则认为 target 在目录内
        
        # 安全解压：先逐一检查所有成员路径不越界，再统一解压
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():  # 遍历归档内所有成员
                member_path = os.path.join(path, member.name)  # 该成员解压后的完整路径
                if not is_within_directory(path, member_path):  # 成员路径越出了解压目录
                    raise Exception("Attempted Path Traversal in Tar File")  # 拒绝解压，防止路径穿越攻击
        
            # 解压全部（已检查过的）成员到指定目录
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=root)  # 执行安全解压，解压到 root 目录
