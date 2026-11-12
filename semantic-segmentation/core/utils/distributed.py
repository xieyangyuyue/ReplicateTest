# ============================================================================
# 【文件说明】distributed.py —— 分布式（多进程/多卡）训练的通信原语与数据采样器。
# 参考实现：facebookresearch/maskrcnn-benchmark 的 comm.py 与 samplers。
# 【核心概念】
#   world_size：参与训练的进程（GPU）总数；rank：当前进程的全局编号（从 0 起），
#   rank==0 为主进程（负责日志打印、保存权重等只需做一次的工作）。
#   进程组由外部建立：本仓库在 scripts/train.py、scripts/eval.py 中调用
#   torch.distributed.init_process_group(backend="nccl", init_method="env://")
#   （NCCL 后端 + 环境变量 MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE 寻址），
#   之后本模块所有 dist.* 集合通信默认走该全局进程组；NCCL 要求参与通信的
#   张量必须在 CUDA 上，且各 rank 上的张量形状与顺序完全一致（这也是
#   reduce_dict 要先对键排序再堆叠的原因），否则会死锁或出错。
# 【核心逻辑】
#   1) get_world_size / get_rank / is_main_process：进程信息查询，分布式
#      未启用/未初始化时安全降级为单进程（world_size=1、rank=0）；
#   2) synchronize：barrier 屏障，所有进程到齐才继续；
#   3) all_gather：对任意可 pickle 对象的跨进程收集——pickle 序列化成字节
#      张量后走张量通信通道；因 torch.all_gather 不支持不同 shape，先互传
#      各自字节长度、把数据 pad 到全局最大长度再 gather，最后按真实长度
#      截取并反序列化还原对象；
#   4) reduce_dict / reduce_loss_dict：把各进程的指标/损失字典 reduce（求和）
#      到 rank0 再除以 world_size 取平均。为什么需要归约：DDP 下每个进程
#      只看到 1/world_size 的数据，其损失只是局部值；要在日志里展示“全局
#      平均损失”就必须跨进程求和平均（模型梯度的同步由 DDP 在反向时自动
#      all_reduce 求平均完成，与此处对损失指标的手工归约同理）。通信本身
#      不需要梯度，故包在 torch.no_grad() 中以省显存、避免误建计算图；
#   5) DistributedSampler：把数据集切成互不重叠的 rank 分片，配合 DDP
#      保证每张图每个 epoch 只被一个进程消费；打乱用 epoch 做随机种子，
#      保证各进程打乱顺序一致后再按 rank 偏移切分；
#   6) IterationBasedBatchSampler：把按 epoch 迭代改造成按 iteration 迭代
#      （语义分割常以迭代步数而非轮数安排训练）。
# ============================================================================
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""
import math  # math.ceil：向上取整，用于把数据集均分到每个进程
import pickle  # 把任意 Python 对象序列化为字节流，复用张量通信通道传输
import torch  # 构造 ByteTensor/IntTensor 通信载体与确定性随机索引
import torch.utils.data as data  # 数据工具：RandomSampler/SequentialSampler/BatchSampler
import torch.distributed as dist  # 分布式集合通信：all_gather/reduce/barrier 等

from torch.utils.data.sampler import Sampler, BatchSampler  # 采样器基类，供 DistributedSampler/IterationBasedBatchSampler 继承

__all__ = ['get_world_size', 'get_rank', 'synchronize', 'is_main_process',  # 模块公开接口清单（import * 时生效）
           'all_gather', 'make_data_sampler', 'make_batch_data_sampler',  # 【注意】未包含 DistributedSampler/IterationBasedBatchSampler，但二者仍可显式 import
           'reduce_dict', 'reduce_loss_dict']  # 字典/损失字典的跨进程归约接口


# reference: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/comm.py
# 返回全局进程总数 world_size；分布式不可用/未初始化时安全降级为 1（按单进程处理）
def get_world_size():
    if not dist.is_available():  # PyTorch 编译时未包含分布式支持
        return 1  # 视为单进程
    if not dist.is_initialized():  # 尚未调用 init_process_group 建立进程组
        return 1  # 同样按单进程处理
    return dist.get_world_size()  # 返回当前进程组内的进程总数


# 返回当前进程的全局编号 rank；分布式不可用/未初始化时降级为 0（即“主进程”）
def get_rank():
    if not dist.is_available():  # 分布式支持不可用
        return 0  # 单进程即 rank 0
    if not dist.is_initialized():  # 进程组未初始化
        return 0  # 默认 rank 0
    return dist.get_rank()  # 返回当前进程在进程组中的编号


# 判断是否为主进程（rank==0）：日志打印、权重保存等操作只需主进程执行一次
def is_main_process():
    return get_rank() == 0  # rank 为 0 即主进程


# 进程间同步（barrier）：分布式训练时对齐所有进程的执行进度
def synchronize():
    # docstring：分布式训练时在所有进程间同步（barrier）的辅助函数
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():  # 分布式支持不可用：无需同步
        return  # 直接返回
    if not dist.is_initialized():  # 进程组未初始化（单进程）：无需同步
        return  # 直接返回
    world_size = dist.get_world_size()  # 读取进程总数
    if world_size == 1:  # 只有单个进程
        return  # barrier 无意义，跳过
    dist.barrier()  # 屏障：所有进程都到达此处才一起继续，用于对齐各进程执行进度（如评估/保存前同步）


# 跨进程收集任意可 pickle 对象：每个进程各提交一份 data，返回按 rank 排序的对象列表
def all_gather(data):
    # docstring：对任意可 pickle 对象（不限于张量）执行跨进程 all_gather
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()  # 进程总数
    if world_size == 1:  # 单进程：无需通信
        return [data]  # 包一层列表，保持与多进程时一致的返回结构

    # serialized to a Tensor
    buffer = pickle.dumps(data)  # 任意 Python 对象 → 字节串
    storage = torch.ByteStorage.from_buffer(buffer)  # 字节串 → 字节存储（【注意】ByteStorage 属废弃 API，PyTorch 2.x 已移除，新版应改用 torch.frombuffer）
    tensor = torch.ByteTensor(storage).to("cuda")  # 包装成字节张量并搬到 GPU（NCCL 只能通信 CUDA 张量；【注意】写死 "cuda"，不支持纯 CPU 训练）

    # obtain Tensor size of each rank
    local_size = torch.IntTensor([tensor.numel()]).to("cuda")  # 本进程数据的字节长度包成 1 元 IntTensor 搬上 GPU
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]  # 为每个 rank 预分配接收长度的占位张量（初始 0，gather 后被覆写）
    dist.all_gather(size_list, local_size)  # 集合通信①：交换各 rank 的字节长度
    size_list = [int(size.item()) for size in size_list]  # GPU 上的标量转成 Python int
    max_size = max(size_list)  # 全局最大字节长度（gather 要求各端形状一致）

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []  # 各 rank 数据的接收缓冲列表
    for _ in size_list:  # 为每个 rank 预分配缓冲
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))  # 统一分配 max_size 长度的空字节张量
    if local_size != max_size:  # 本进程数据短于最大长度
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")  # 构造补齐用的填充字节
        tensor = torch.cat((tensor, padding), dim=0)  # 拼接补齐到 max_size
    dist.all_gather(tensor_list, tensor)  # 集合通信②：交换真正的数据字节（已对齐长度）

    data_list = []  # 反序列化结果列表（下标即 rank）
    for size, tensor in zip(size_list, tensor_list):  # 遍历每个 rank 的接收缓冲
        buffer = tensor.cpu().numpy().tobytes()[:size]  # 搬回 CPU 转字节串，并按该 rank 的真实长度截掉 padding
        data_list.append(pickle.loads(buffer))  # 反序列化还原为原始 Python 对象

    return data_list  # 返回 list[rank]，第 i 项即 rank i 传入的 data


# 参数：input_dict——键为字符串、值为同形状 CUDA 张量的字典；average——True 求平均 / False 仅求和
# 【注意】dist.reduce 只把结果写到目标进程：仅 rank0 拿到归约值，其余 rank 返回的仍是本地原值
def reduce_dict(input_dict, average=True):
    # docstring：跨进程归约字典取值，使 rank0 拿到求和（可选平均）后的结果
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()  # 进程总数
    if world_size < 2:  # 单进程：无需通信
        return input_dict  # 原样返回
    with torch.no_grad():  # 指标归约不需要梯度：关闭梯度记录以省显存并避免污染计算图
        names = []  # 排序后的键名列表
        values = []  # 排序后的值列表
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):  # 按字典序遍历键：保证各 rank 堆叠出的张量逐位对应（集合通信的硬性要求）
            names.append(k)  # 记录键名
            values.append(input_dict[k])  # 记录对应的值张量
        values = torch.stack(values, dim=0)  # 堆叠为一维张量：第 i 个元素对应 names[i]
        dist.reduce(values, dst=0)  # 默认 SUM 归约到 dst=0（rank0）：每个元素变为各进程对应位置之和
        if dist.get_rank() == 0 and average:  # 仅 rank0 持有总和，且要求平均
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size  # 总和 ÷ 进程数 = 全局平均值
        reduced_dict = {k: v for k, v in zip(names, values)}  # 还原为 {原键名: 归约后的值} 字典
    return reduced_dict  # rank0 得到全局平均；其余 rank 为本地值（调用方只应在主进程使用结果）


# 参数：loss_dict——训练/验证返回的损失字典（值为各进程本地损失的标量张量）
# 与 reduce_dict 的区别：无条件求平均（训练日志展示全局平均损失用）
def reduce_loss_dict(loss_dict):
    # docstring：跨进程归约损失字典，使 rank0 拿到各进程平均后的结果
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()  # 进程总数
    if world_size < 2:  # 单进程：无需通信
        return loss_dict  # 原样返回
    with torch.no_grad():  # 损失归约仅用于日志展示，不需要梯度
        loss_names = []  # 排序后的损失名列表
        all_losses = []  # 排序后的损失值列表
        for k in sorted(loss_dict.keys()):  # 按字典序遍历（保证各 rank 顺序一致，可安全堆叠通信）
            loss_names.append(k)  # 记录损失名
            all_losses.append(loss_dict[k])  # 记录损失值
        all_losses = torch.stack(all_losses, dim=0)  # 堆叠为一维张量：第 i 个元素对应 loss_names[i]
        dist.reduce(all_losses, dst=0)  # SUM 归约到 rank0：各进程损失逐项求和
        if dist.get_rank() == 0:  # 仅 rank0 持有求和结果
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size  # 除以进程数得到全局平均损失
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}  # 还原为 {损失名: 平均值} 字典
    return reduced_losses  # rank0 为全局平均；其余 rank 为本地值（仅主进程打印日志）


# 参数：dataset——数据集对象；shuffle——是否打乱；distributed——是否处于分布式模式
# 作用：按训练模式选择“样本级”采样器（返回给 DataLoader 的 sampler 参数）
def make_data_sampler(dataset, shuffle, distributed):
    if distributed:  # 分布式：每个进程只采样互不重叠的数据分片
        return DistributedSampler(dataset, shuffle=shuffle)  # 内部按 world_size/rank 自动切分数据集
    if shuffle:  # 单机模式且需要打乱
        sampler = data.sampler.RandomSampler(dataset)  # 无放回随机采样
    else:  # 顺序遍历（如验证集）
        sampler = data.sampler.SequentialSampler(dataset)  # 按数据集原始顺序采样
    return sampler  # 返回构造好的采样器


# 参数：sampler——底层样本级采样器；images_per_batch——每个 batch 的图片数；
#      num_iters——总迭代步数（None 表示按 epoch 跑完整个采样器）；start_iter——起始迭代号（断点续训用）
# 作用：把样本流切成 batch（返回给 DataLoader 的 batch_sampler 参数）
def make_batch_data_sampler(sampler, images_per_batch, num_iters=None, start_iter=0):
    batch_sampler = data.sampler.BatchSampler(sampler, images_per_batch, drop_last=True)  # 把采样器产出的索引切成 batch；drop_last 丢弃不满一个 batch 的尾部，保证各进程 batch 数一致（避免集合通信对不齐而死锁）
    if num_iters is not None:  # 指定了按迭代步数训练
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)  # 包装成“循环取数直到满 num_iters 步”的批采样器
    return batch_sampler  # 返回可直接交给 DataLoader(batch_sampler=...) 的批采样器


# Code is copy-pasted from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/distributed.py
# 分布式采样器：与 DDP 配合——每个进程持有一份完整模型副本（DistributedDataParallel 包装），
# 各进程用本采样器加载互不重叠的数据分片并行前向；反向时 DDP 自动对各副本梯度做
# all_reduce 求平均，保证所有副本参数始终一致
class DistributedSampler(Sampler):
    # 类 docstring：把数据加载限制在数据集子集的采样器（配合 DDP，各进程加载互斥分片）
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    # 参数：dataset——数据集（只用其长度）；num_replicas——进程/副本数（默认取 world_size）；
    #      rank——当前进程编号（默认取 dist.get_rank()）；shuffle——是否按 epoch 打乱
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:  # 未显式指定进程数
            if not dist.is_available():  # 分布式支持不可用
                raise RuntimeError("Requires distributed package to be available")  # 无法推导 world_size，直接报错
            num_replicas = dist.get_world_size()  # 从进程组读取 world_size
        if rank is None:  # 未显式指定进程编号
            if not dist.is_available():  # 分布式支持不可用
                raise RuntimeError("Requires distributed package to be available")  # 同样无法推导，报错
            rank = dist.get_rank()  # 从进程组读取当前 rank
        self.dataset = dataset  # 保存数据集引用
        self.num_replicas = num_replicas  # 进程（副本）数
        self.rank = rank  # 当前进程编号
        self.epoch = 0  # 当前轮次号：作为打乱种子；set_epoch 更新后下一轮换一种打乱方式
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))  # 每进程样本数 = 总数/进程数 向上取整（*1.0 是 Py2 时代的整除转浮点写法，Py3 下多余）
        self.total_size = self.num_samples * self.num_replicas  # 补齐后的总样本数（≥ 原数据集大小，能被进程数整除）
        self.shuffle = shuffle  # 是否打乱

    # 生成本进程本轮要消费的样本索引（含打乱与分片逻辑）
    def __iter__(self):
        if self.shuffle:  # 需要打乱
            # deterministically shuffle based on epoch
            g = torch.Generator()  # 独立的随机数发生器，不污染全局随机状态
            g.manual_seed(self.epoch)  # 种子 = epoch：所有进程用同一种子 → 生成完全相同的打乱顺序（这是正确切分的前提）
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # 确定性地随机排列全体样本索引
        else:  # 不打乱
            indices = torch.arange(len(self.dataset)).tolist()  # 顺序索引 0..N-1

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]  # 把头部样本重复追加到尾部，补齐到 total_size（保证能被进程数整除，代价是少量样本每轮被重复采样）
        assert len(indices) == self.total_size  # 校验补齐后长度正确

        # subsample
        offset = self.num_samples * self.rank  # 本进程分片的起始偏移量
        indices = indices[offset: offset + self.num_samples]  # 截取连续的 num_samples 条 → 各进程分片互不重叠、并集为全集
        assert len(indices) == self.num_samples  # 校验分片长度正确

        return iter(indices)  # 包装成迭代器，供上层 BatchSampler 按 batch 消费

    # 每进程每个 epoch 的样本数
    def __len__(self):
        return self.num_samples  # 每进程每个 epoch 的样本数

    # 参数：epoch——轮次号；更新打乱种子，使每个 epoch 的数据顺序都不同
    def set_epoch(self, epoch):
        self.epoch = epoch  # 更新轮次号：下一轮 __iter__ 用新种子重新打乱，避免每个 epoch 数据顺序都相同


# 按 iteration（迭代步）计数的批采样器：把底层 BatchSampler 的“按 epoch 产数”
# 包装成“循环消费直到取满指定迭代数”，供以步数安排训练的语义分割任务使用
class IterationBasedBatchSampler(BatchSampler):
    # 类 docstring：包装 BatchSampler，循环重采样直到取满指定迭代数
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    # 参数：batch_sampler——被包装的底层批采样器；num_iterations——目标总迭代数；start_iter——起始迭代号（断点续训）
    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler  # 保存底层批采样器
        self.num_iterations = num_iterations  # 目标迭代总数
        self.start_iter = start_iter  # 起始迭代号

    # 生成器：持续产出 batch，直到达到 num_iterations 个迭代（供 DataLoader 消费）
    def __iter__(self):
        iteration = self.start_iter  # 从断点迭代号开始计数
        while iteration <= self.num_iterations:  # 【注意】边界用了 <=（配合下方 break 后实际产出数量不变，但会多消费一个 batch 才退出；写 < 更直观）
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):  # 底层采样器支持 set_epoch（如 DistributedSampler）
                self.batch_sampler.sampler.set_epoch(iteration)  # 用当前迭代号充当“epoch”更新打乱种子（周期性重新打乱数据）
            for batch in self.batch_sampler:  # 消费底层采样器的一个完整 epoch（耗尽后自动重新开始）
                iteration += 1  # 已产出一个 batch，计数加一
                if iteration > self.num_iterations:  # 已达到目标迭代数
                    break  # 丢弃当前多余的 batch 并退出
                yield batch  # 产出这个 batch（生成器：DataLoader 按需拉取）

    # 声明的总迭代数（与产出数量一致）
    def __len__(self):
        return self.num_iterations  # 声明的总迭代数（供进度条/日志计算用）


# 【注意】模块自测入口为空实现：直接运行本文件不做任何事（死代码）
if __name__ == '__main__':
    pass  # 占位语句：无自测逻辑
