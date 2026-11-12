# ============================================================================
# 【文件说明】download.py —— 预训练权重等文件的下载工具。
# 核心逻辑：
#   1) check_sha1：分块（1MB）读取文件增量计算 SHA1，与期望哈希比对；
#      比对长度取两者较小值，即允许“前缀匹配”（哈希记录不完整时也能校验）；
#   2) download：根据 url/path 推导本地文件名（path 为目录时取 url 末段名），
#      仅当“强制 overwrite / 本地不存在 / 指定哈希但校验不过”时才真正下载；
#      下载用 requests 流式模式按 1KB 分块写盘：响应头带 content-length 时
#      用 tqdm 显示以 KB 为单位的进度条，否则朴素循环写入；下载完成后
#      若指定了哈希会再次校验，不匹配则抛出警告。
# ============================================================================
"""Download files with progress bar."""
import os  # 路径展开、父目录创建、存在性判断
import hashlib  # 计算 SHA1 摘要
import requests  # HTTP 库：stream=True 流式下载大文件
from tqdm import tqdm  # 终端进度条

# 参数：filename——待校验文件的路径；sha1_hash——期望的十六进制 SHA1 字符串
def check_sha1(filename, sha1_hash):
    # docstring：校验文件内容的 SHA1 哈希是否与期望哈希一致
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()  # 创建 SHA1 哈希对象
    with open(filename, 'rb') as f:  # 二进制只读打开（文本模式可能转义字节，破坏哈希）
        while True:  # 分块循环读取
            data = f.read(1048576)  # 每次读 1MB（1048576 字节），避免大文件一次性载入内存
            if not data:  # 读到 EOF（空块）
                break  # 结束读取循环
            sha1.update(data)  # 增量更新哈希状态

    sha1_file = sha1.hexdigest()  # 得到的十六进制摘要字符串
    l = min(len(sha1_file), len(sha1_hash))  # 取两串长度的较小值 l：只比对前 l 位（前缀匹配策略）
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]  # 前 l 位完全一致即视为校验通过（【注意】变量名 l 易与数字 1 混淆）

# 参数：url——下载地址；path——目标目录或完整文件路径（默认存当前目录）；
#      overwrite——本地已存在时是否强制重新下载；sha1_hash——期望哈希（用于完整性校验）
def download(url, path=None, overwrite=False, sha1_hash=None):
    # docstring：下载指定 URL 的文件（含参数、返回值与覆盖/哈希校验策略说明）
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:  # 未指定保存路径
        fname = url.split('/')[-1]  # 取 url 最后一段作为文件名，落在当前工作目录
    else:  # 指定了路径
        path = os.path.expanduser(path)  # 展开 ~ 等 shell 风格变量
        if os.path.isdir(path):  # 路径是已存在的目录
            fname = os.path.join(path, url.split('/')[-1])  # 目录 + url 末段文件名
        else:  # 路径视为完整文件名
            fname = path  # 直接使用该路径

    # 需要下载的三种情况：强制覆盖 / 本地不存在 / 指定了哈希但本地文件校验不过（视为坏缓存）
    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))  # 求目标文件的绝对父目录
        if not os.path.exists(dirname):  # 父目录不存在
            os.makedirs(dirname)  # 递归创建父目录

        print('Downloading %s from %s...'%(fname, url))  # 打印下载提示信息
        r = requests.get(url, stream=True)  # 发起 GET 请求；stream=True 延迟读取响应体，边下边写
        if r.status_code != 200:  # HTTP 状态码非 200（失败）
            raise RuntimeError("Failed downloading url %s"%url)  # 抛出运行时错误终止下载
        total_length = r.headers.get('content-length')  # 从响应头取文件总字节数（服务器可能不返回）
        with open(fname, 'wb') as f:  # 二进制写打开目标文件
            # 响应头没有 content-length：无法计算百分比，只能朴素分块写入
            if total_length is None: # no content length header
                for chunk in r.iter_content(chunk_size=1024):  # 按 1KB 分块迭代响应体
                    # keep-alive 可能产生空块，过滤掉再写盘
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)  # 写入一个 1KB 数据块
            else:  # 有总长度：可用 tqdm 展示进度条
                total_length = int(total_length)  # 总字节数转 int
                for chunk in tqdm(r.iter_content(chunk_size=1024),  # 仍按 1KB 分块迭代
                                  total=int(total_length / 1024. + 0.5),  # 总块数 = 字节数/1024 四舍五入（+0.5 实现 round half up）
                                  unit='KB', unit_scale=False, dynamic_ncols=True):  # 进度单位 KB、不做单位缩放、宽度自适应终端
                    f.write(chunk)  # 逐块写入文件

        # 【注意】此处把 UserWarning（警告类）当异常 raise——因 Warning 继承自 Exception 而能运行，
        # 语义上更应是 warnings.warn(...) 或 raise RuntimeError(...)；按要求保持原样不改
        if sha1_hash and not check_sha1(fname, sha1_hash):  # 下载完成后再次校验哈希
            # 校验失败：抛出警告类异常提示哈希不匹配（多行提示串以反斜杠续行，行尾无法再加注释）
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              # 提示：仓库可能已过期或本次下载不完整
                              'The repo may be outdated or download may be incomplete. ' \
                              # 提示：若覆盖了默认下载源，可考虑切回默认仓库
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))  # format 填入文件名后抛出该警告

    return fname  # 返回最终文件路径（本地缓存有效时直接复用，不重新下载）