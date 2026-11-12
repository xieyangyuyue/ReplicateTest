# ============================================================================
# 【文件说明】parallel.py —— DataParallel 风格的模型/损失函数多卡并行封装
#（移植自 PyTorch-Encoding，专为本仓库的语义分割场景设计）。
# 背景问题：nn.DataParallel 原生只对“模型”并行，且会把各卡输出 gather
# 回同一张卡再返回——语义分割输出是高分辨率特征图/概率图，聚到一张卡
# 上再计算损失会瞬间爆显存。
# 本文件的解决思路（成对使用两个类）：
#   DataParallelModel 覆写 gather 为“原样返回”，使各卡输出列表保留在
#   各自 GPU 上（输入沿 batch 维 scatter → 模型 replicate 到各卡分别
#   前向 → 输出不聚合）；
#   DataParallelCriterion 覆写 forward：把 targets 同样 scatter 到各卡，
#   多线程在各卡上并行计算损失，再用自定义 Reduce 节点把各卡损失（支持
#   标量或 dict 形式的多项损失，逐元素归约）求和后除以卡数取平均。
# 反向传播：Reduce.backward 用 Broadcast 把平均损失的梯度广播回每张卡，
# 梯度沿各卡子图回传并在模型副本上累加——等效于对损失做了多卡均衡，
# 显存占用在各卡间更平均（这正是“损失函数也要并行”的原因）。
# 【注意】依赖 torch.cuda.comm / torch.nn.parallel._functions 等私有/
#         已废弃 API，新版本 PyTorch 中可能不可用；官方现推荐 DDP 方案。
# ============================================================================
"""Utils for Semantic Segmentation"""
import threading  # 多线程驱动各 GPU 上的损失模块副本并行计算
import torch  # 张量与自动求导机制
import torch.cuda.comm as comm  # GPU 间通信原语：reduce_add 跨卡求和（【注意】属内部/废弃 API）
from torch.nn.parallel.data_parallel import DataParallel  # 官方 DataParallel 基类：提供 scatter/replicate 等机制
from torch.nn.parallel._functions import Broadcast  # 官方内部的 Broadcast 自动求导节点：把张量复制到多卡（【注意】私有 API）
from torch.autograd import Function  # 自定义可反向的 autograd 节点基类

__all__ = ['DataParallelModel', 'DataParallelCriterion']  # 对外导出两个并行封装类


# 自定义 autograd 节点：前向把多张卡上的输出（各卡损失）求和；
# 反向把总和对各输入的梯度广播回各卡，保证各卡子图都能收到梯度
class Reduce(Function):
    @staticmethod  # 旧式（兼容）autograd.Function 写法：forward/backward 均为静态方法
    def forward(ctx, *inputs):  # inputs：来自各 GPU 的输出张量（每卡一个，顺序即 device_ids 顺序）
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]  # 记录每个输入所在的 GPU 号（保留原始顺序，反向时按此顺序发回梯度）
        inputs = sorted(inputs, key=lambda i: i.get_device())  # 按卡号升序排序：comm.reduce_add 要求输入按设备编号有序
        return comm.reduce_add(inputs)  # 跨卡逐元素求和，结果落在卡号最小的 GPU 上

    @staticmethod  # 反向传播：把对“求和结果”的梯度广播回各卡
    def backward(ctx, gradOutputs):  # gradOutputs：损失对“求和结果”的梯度
        return Broadcast.apply(ctx.target_gpus, gradOutputs)  # 把同一梯度广播回 forward 时记录的各卡（顺序用 target_gpus 恢复）


# 模型并行封装：与 DataParallelCriterion 配对使用（核心是覆写 gather 不聚合输出）
class DataParallelModel(DataParallel):
    # 类 docstring：数据并行封装说明（对用户屏蔽单/多 GPU 差异，含用法示例）
    """Data parallelism

    Hide the difference of single/multiple GPUs to the user.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used.

    Parameters
    ----------
    module : object
        Network to be parallelized.
    sync : bool
        enable synchronization (default: False).
    Inputs:
        - **inputs**: list of input
    Outputs:
        - **outputs**: list of output
    Example::
        >>> net = DataParallelModel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)  # input_var can be on any device, including CPU
    """

    def gather(self, outputs, output_device):  # 覆写官方 gather：outputs 为各卡输出列表（正常实现会搬到 output_device 并拼接）
        return outputs  # 关键改动：不做任何聚合/搬移，输出列表原样返回——各卡输出留在各自 GPU 上，供 DataParallelCriterion 就地消费

    def replicate(self, module, device_ids):  # 覆写 replicate（实际为透传）
        modules = super(DataParallelModel, self).replicate(module, device_ids)  # 调父类逻辑：把 module 复制到各 device_ids
        return modules  # 原样返回副本列表


# Reference: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/parallel.py
# 损失函数并行封装：与 DataParallelModel 配对——在各卡上就地计算损失再归约求平均，
# 避免把大分辨率输出聚合到单卡导致的显存瓶颈
class DataParallelCriterion(DataParallel):
    # 类 docstring：多 GPU 并行计算损失，均衡语义分割任务的显存占用
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.

    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.

    Example::
        >>> net = DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """

    def forward(self, inputs, *targets, **kwargs):  # inputs：DataParallelModel 的输出列表（每卡一份）；targets：监督标签；kwargs：透传的关键字参数
        # the inputs should be the outputs of DataParallelModel
        if not self.device_ids:  # 未指定 GPU（纯 CPU 场景）
            return self.module(inputs, *targets, **kwargs)  # 退化为直接调用原始损失模块
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)  # 把 targets（连同 kwargs）沿 batch 维切块搬到各卡，与各卡输出一一对应（DataParallel 的 scatter 机制）
        if len(self.device_ids) == 1:  # 单卡场景
            return self.module(inputs, *targets[0], **kwargs[0])  # 直接在该卡上计算损失，无需归约
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])  # 把损失模块复制到每个有输出的卡（切片长度=输出数，与各卡输出一一对应）
        outputs = criterion_parallel_apply(replicas, inputs, targets, kwargs)  # 多线程在各卡上并行计算各自的损失（得到各卡损失列表）
        return Reduce.apply(*outputs) / len(outputs)  # 用 Reduce 节点跨卡求和（反向可广播梯度）再除以卡数 → 各卡损失的平均值


# 递归地在嵌套结构（list/tuple/dict）中找到第一个张量，用于推断数据所在 GPU
def get_a_var(obj):
    if isinstance(obj, torch.Tensor):  # 本身就是张量
        return obj  # 直接返回

    if isinstance(obj, list) or isinstance(obj, tuple):  # 列表或元组
        for result in map(get_a_var, obj):  # 对每个元素递归查找
            if isinstance(result, torch.Tensor):  # 找到第一个张量
                return result  # 返回之

    if isinstance(obj, dict):  # 字典
        for result in map(get_a_var, obj.items()):  # 对每个 (键, 值) 二元组递归查找
            if isinstance(result, torch.Tensor):  # 找到第一个张量
                return result  # 返回之
    return None  # 整个结构里没有张量（【注意】调用方对其调 get_device() 会因 None 抛 AttributeError）


# 参数：modules——各卡的损失模块副本列表；inputs——各卡输出；targets——各卡标签；
#      kwargs_tup——各卡关键字参数元组；devices——各卡设备号列表
def criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None, devices=None):
    # docstring：在各设备上并行执行各损失模块副本（位置/关键字参数与长度约束说明）
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional), attr:'targets' (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        targets (tensor): targets to the modules
        devices (list of int or torch.device): CUDA devices
    :attr:`modules`, :attr:`inputs`, :attr:'targets' :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)  # 校验：模块数与输入数一致（逐卡对应）
    assert len(targets) == len(inputs)  # 校验：标签数与输入数一致
    if kwargs_tup is not None:  # 显式给定了关键字参数元组
        assert len(modules) == len(kwargs_tup)  # 校验：同样逐卡对应
    else:  # 未给定
        kwargs_tup = ({},) * len(modules)  # 每卡用空 dict 占位
    if devices is not None:  # 显式给定了设备列表
        assert len(modules) == len(devices)  # 校验：同样逐卡对应
    else:  # 未给定
        devices = [None] * len(modules)  # 占位 None（worker 内再从输入张量推断设备）
    lock = threading.Lock()  # 保护共享 results 字典的线程锁
    results = {}  # 各线程写回：卡号 → 该卡输出或异常对象
    grad_enabled = torch.is_grad_enabled()  # 记录主线程的梯度开关（grad 模式是线程局部变量，子线程需手动同步）

    def _worker(i, module, input, target, kwargs, device=None):  # 单卡 worker：在第 i 张卡上计算损失
        torch.set_grad_enabled(grad_enabled)  # 把主线程的梯度模式同步到本线程（保证损失计算在训练态下进行）
        if device is None:  # 未显式指定设备号
            device = get_a_var(input).get_device()  # 从该卡输入里找第一个张量，推断所在 GPU 号
        try:  # 捕获线程内异常（线程里的异常不会自动传播到主线程）
            with torch.cuda.device(device):  # 切换当前 CUDA 设备上下文到该卡
                output = module(*(list(input) + target), **kwargs)  # 拼接 输入+标签 作为位置参数调用损失模块（如 dict 形式的多项损失也在此逐卡计算）
            with lock:  # 加锁写共享字典
                results[i] = output  # 保存该卡的正常输出
        except Exception as e:  # 任何异常
            with lock:  # 加锁
                results[i] = e  # 把异常对象暂存，稍后在主线程统一重新抛出

    if len(modules) > 1:  # 多卡：为每个副本开一个线程并行计算
        threads = [threading.Thread(target=_worker,  # 为每个 (副本, 输入, 标签...) 组合创建一个线程
                                    args=(i, module, input, target, kwargs, device))  # 线程参数：卡号、损失模块副本、该卡的输入/标签/关键字参数/设备
                   for i, (module, input, target, kwargs, device) in  # 逐卡解包五元组并编号
                   enumerate(zip(modules, inputs, targets, kwargs_tup, devices))]  # 逐卡打包参数创建线程（CUDA 算子会释放 GIL，可近似真并行）

        for thread in threads:  # 遍历线程列表
            thread.start()  # 启动线程
        for thread in threads:  # 再次遍历
            thread.join()  # 阻塞等待所有线程结束
    else:  # 单卡：直接在当前线程执行，省去线程开销
        _worker(0, modules[0], inputs[0], targets[0], kwargs_tup[0], devices[0])  # 就地调用 worker

    outputs = []  # 汇总输出列表
    for i in range(len(inputs)):  # 按卡号顺序遍历
        output = results[i]  # 取出该卡结果（正常输出或异常）
        if isinstance(output, Exception):  # 该卡线程当时失败了
            raise output  # 在主线程重新抛出，暴露真实错误
        outputs.append(output)  # 正常输出加入列表
    return outputs  # 返回各卡损失列表（顺序与输入一致，供 Reduce 求和）
