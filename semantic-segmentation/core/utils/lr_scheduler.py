# ============================================================================
# 【文件说明】lr_scheduler.py —— 语义分割训练常用的学习率调度器：函数式的
#             LRScheduler（支持 constant/step/linear/poly/cosine 五种衰减模式并可
#             叠加 warmup 预热）与基于 torch 调度器框架的 WarmupMultiStepLR、
#             WarmupPolyLR（训练脚本实际使用的两个类）。
# 【核心思想】
#   warmup（预热）：训练初期权重随机、梯度噪声大，直接用大学习率易震荡发散，
#   因此前 warmup_iters 步让学习率从极小值按系数 warmup_factor（如 1/3）线性爬升
#   到基准值后再进入正常衰减；预热期 lr = target + (base - target) * warmup_factor，
#   其中 warmup_factor 由 warmup_factor*(1-alpha)+alpha 随 alpha=t/warmup_iters
#   从起始系数线性升至 1。
#   poly（多项式）衰减：lr = target_lr + (base_lr - target_lr) * (1 - t/T)^power，
#   power 常取 0.9，曲线前段下降快、末段平缓逼近 target_lr，是分割训练标配。
#   step（阶梯）衰减：每越过一个里程碑 epoch/iter，学习率乘以 gamma（或 step_factor）。
# ============================================================================
"""Popular Learning Rate Schedulers"""  # 原有 docstring：常用学习率调度器
from __future__ import division  # Python2 的真除法兼容导入；【注意】Python3 下已无实际作用（保留原样）
import math  # 数学库：cosine 模式用到 math.cos 与 math.pi
import torch  # 导入 PyTorch 主库（下方两个类基于 torch.optim 的调度器基类）

from bisect import bisect_right  # 二分查找：返回插入点右侧下标，用于统计 last_epoch 已越过（<=）的里程碑个数

__all__ = ['LRScheduler', 'WarmupMultiStepLR', 'WarmupPolyLR']  # 对外导出符号白名单


class LRScheduler(object):  # 函数式学习率调度器：训练循环中每步手动调用 __call__ 计算并写入优化器 lr
    # 类 docstring：五种衰减模式与各参数（基准/目标学习率、迭代与轮数、预热等）的英文说明
    r"""Learning Rate Scheduler

    Parameters
    ----------
    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'constant', 'step', 'linear', 'poly' and 'cosine'.
    base_lr : float
        Base learning rate, i.e. the starting learning rate.
    target_lr : float
        Target learning rate, i.e. the ending learning rate.
        With constant mode target_lr is ignored.
    niters : int
        Number of iterations to be scheduled.
    nepochs : int
        Number of epochs to be scheduled.
    iters_per_epoch : int
        Number of iterations in each epoch.
    offset : int
        Number of iterations before this scheduler.
    power : float
        Power parameter of poly scheduler.
    step_iter : list
        A list of iterations to decay the learning rate.
    step_epoch : list
        A list of epochs to decay the learning rate.
    step_factor : float
        Learning rate decay factor.
    """  # 原有 docstring 整体保留：各参数的英文说明

    # 形参说明: mode 衰减模式('constant'/'step'/'linear'/'poly'/'cosine'); base_lr 基准学习率(默认0.01)
    # 形参说明: target_lr 结束学习率(默认0); niters 总迭代数; nepochs 总轮数; iters_per_epoch 每轮迭代数
    # 形参说明: offset 迭代偏移(接续训练用); power poly 幂次(默认0.9); step_iter/step_epoch 衰减节点; step_factor 阶梯衰减因子(默认0.1); warmup_epochs 预热轮数
    def __init__(self, mode, base_lr=0.01, target_lr=0, niters=0, nepochs=0, iters_per_epoch=0,
                 offset=0, power=0.9, step_iter=None, step_epoch=None, step_factor=0.1, warmup_epochs=0):  # （构造器参数续行）迭代偏移/poly 幂次/衰减节点/阶梯因子/预热轮数
        super(LRScheduler, self).__init__()  # 初始化父类 object
        assert (mode in ['constant', 'step', 'linear', 'poly', 'cosine'])  # 校验调度模式必须合法

        if mode == 'step':  # 阶梯模式必须提供衰减节点
            assert (step_iter is not None or step_epoch is not None)  # 迭代粒度或轮数粒度的节点至少提供其一
        self.niters = niters  # 总迭代数
        self.step = step_iter  # 迭代粒度的衰减节点列表
        epoch_iters = nepochs * iters_per_epoch  # 由"轮数 × 每轮迭代数"换算出的总迭代数
        if epoch_iters > 0:  # 若提供了轮数信息，则以换算结果为准
            self.niters = epoch_iters  # 覆盖总迭代数
            if step_epoch is not None:  # 轮数粒度的节点需要换算为迭代粒度
                self.step = [s * iters_per_epoch for s in step_epoch]  # 每个节点乘以每轮迭代数

        self.step_factor = step_factor  # 阶梯衰减因子：每过一个节点 lr 乘以该因子（默认 0.1）
        self.base_lr = base_lr  # 基准（起始）学习率
        self.target_lr = base_lr if mode == 'constant' else target_lr  # constant 模式忽略 target_lr（恒为 base_lr）
        self.offset = offset  # 迭代偏移量：调度从 offset 之后开始计数
        self.power = power  # poly 衰减的幂指数
        self.warmup_iters = warmup_epochs * iters_per_epoch  # 预热总迭代数 = 预热轮数 × 每轮迭代数
        self.mode = mode  # 保存调度模式

    def __call__(self, optimizer, num_update):  # 每步调用入口：按当前迭代数计算并写入优化器 lr
        self.update(num_update)  # 先根据迭代数计算 self.learning_rate
        assert self.learning_rate >= 0  # 断言学习率非负
        self._adjust_learning_rate(optimizer, self.learning_rate)  # 将计算结果写入优化器各参数组

    def update(self, num_update):  # 核心逻辑：由迭代数 num_update 计算当前学习率并保存到 self.learning_rate
        N = self.niters - 1  # 有效总步数（减 1 使最后一次迭代 factor 恰为 0）
        T = num_update - self.offset  # 扣除偏移量，得到调度器自身的迭代计数
        T = min(max(0, T), N)  # 截断到 [0, N] 区间，防止越界外推

        if self.mode == 'constant':  # 常数模式
            factor = 0  # 因子恒 0 → lr 恒等于 target_lr（即 base_lr）
        elif self.mode == 'linear':  # 线性衰减
            factor = 1 - T / N  # 因子从 1 线性降到 0：lr 匀速下降
        elif self.mode == 'poly':  # 多项式衰减（分割训练常用）
            factor = pow(1 - T / N, self.power)  # 因子 = (1 - T/N)^power，power 默认 0.9，先快后慢
        elif self.mode == 'cosine':  # 余弦退火
            factor = (1 + math.cos(math.pi * T / N)) / 2  # 因子 = (1 + cos(pi*T/N))/2，从 1 平滑降到 0
        elif self.mode == 'step':  # 阶梯衰减
            if self.step is not None:  # 已提供衰减节点列表
                count = sum([1 for s in self.step if s <= T])  # 统计当前已越过（<= T）的节点个数
                factor = pow(self.step_factor, count)  # 因子 = step_factor^count，每过一个节点乘一次衰减因子
            else:  # 未提供衰减节点列表的分支
                factor = 1  # 未提供节点则不衰减
        else:  # 未知调度模式分支
            raise NotImplementedError  # 未知模式直接抛错

        # warm up lr schedule  # 原注释保留：学习率预热调度
        if self.warmup_iters > 0 and T < self.warmup_iters:  # 处于预热区间（前 warmup_iters 步）
            factor = factor * 1.0 * T / self.warmup_iters  # 在衰减因子上再乘线性预热系数 T/warmup_iters（从 0 升到 1）

        if self.mode == 'step':  # step 模式：lr 直接以 base_lr 缩放（不做 target_lr 插值）
            self.learning_rate = self.base_lr * factor  # lr = base_lr * step_factor^count（可含预热系数）
        else:  # 非 step 模式分支：按 factor 在 target_lr 与 base_lr 之间插值
            self.learning_rate = self.target_lr + (self.base_lr - self.target_lr) * factor  # 其余模式：lr 在 target_lr 与 base_lr 之间按 factor 插值

    def _adjust_learning_rate(self, optimizer, lr):  # 把计算好的 lr 写入优化器的各参数组
        optimizer.param_groups[0]['lr'] = lr  # 第 0 个参数组（骨干网络参数）使用基准 lr
        # enlarge the lr at the head  # 原注释保留：放大分割头的学习率
        for i in range(1, len(optimizer.param_groups)):  # 遍历其余参数组（通常是分割头/解码头）
            optimizer.param_groups[i]['lr'] = lr * 10  # 【注意】魔法数字 10：分割头多为新初始化的层，lr 放大 10 倍以加速收敛


# separating MultiStepLR with WarmupLR  # 原注释保留：作者原意想把 MultiStepLR 与 WarmupLR 拆分组合
# but the current LRScheduler design doesn't allow it  # 原注释保留：但上面的 LRScheduler 设计不允许这样做
# reference: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/solver/lr_scheduler.py  # 原注释保留：参考 maskrcnn-benchmark 的实现
# （中文补充）因此单独实现下面这个"带预热的多阶梯衰减"调度器
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):  # 带预热的阶梯衰减调度器；【注意】_LRScheduler 是 PyTorch 内部私有类（新版已公开为 LRScheduler），依赖私有 API 存在兼容风险
    # 形参说明: optimizer 优化器; milestones 依次衰减的 epoch 节点列表; gamma 每节点衰减系数(默认0.1)
    # 形参说明: warmup_factor 预热起始系数(默认1/3, 预热从 base_lr/3 起步); warmup_iters 预热迭代数(默认500)
    # 形参说明: warmup_method 预热方式('constant'/'linear'); last_epoch 上次迭代号(恢复训练用)
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, warmup_method="linear", last_epoch=-1):  # （构造器参数续行）预热迭代数/预热方式/上次迭代号
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)  # 初始化父类（构造过程中会触发一次 get_lr，要求属性访问安全）
        if not list(milestones) == sorted(milestones):  # 【注意】写法易混淆（解析为 not (a == b)，语义碰巧正确）：校验里程碑列表是否递增
            raise ValueError(  # 里程碑列表非递增时抛出参数错误
                "Milestones should be a list of" " increasing integers. Got {}", milestones)  # 【注意】字符串中的 {} 并未调用 .format 填充，milestones 是作为 ValueError 的第二个参数传入的，报错信息不会真正替换占位符
        if warmup_method not in ("constant", "linear"):  # 校验预热方式必须合法
            raise ValueError(  # 预热方式非法时抛出参数错误
                "Only 'constant' or 'linear' warmup_method accepted got {}".format(warmup_method))  # 非法预热方式报错（此处正确使用了 .format）

        self.milestones = milestones  # 衰减节点列表（epoch 粒度）
        self.gamma = gamma  # 每个节点处的衰减系数
        self.warmup_factor = warmup_factor  # 预热起始系数
        self.warmup_iters = warmup_iters  # 预热迭代步数
        self.warmup_method = warmup_method  # 预热方式

    def get_lr(self):  # 由 last_epoch（框架每步自动自增的计数）计算各参数组的学习率，返回 lr 列表
        warmup_factor = 1  # 预热系数默认 1（等效于不预热）
        if self.last_epoch < self.warmup_iters:  # 仍处于预热阶段
            if self.warmup_method == 'constant':  # 常数预热
                warmup_factor = self.warmup_factor  # 整个预热期保持固定的小系数
            elif self.warmup_factor == 'linear':  # 【注意】bug：误将 warmup_factor（float）与字符串 'linear' 比较，恒为 False，正确应为 self.warmup_method == 'linear'——导致线性预热分支永不执行，linear 预热实际失效（等效 factor 恒 1）
                alpha = float(self.last_epoch) / self.warmup_iters  # 预热进度 alpha ∈ [0,1)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha  # 系数由 warmup_factor 线性升至 1
        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)  # lr = base_lr * warmup_factor * gamma^(已越过节点数)；bisect_right 返回 milestones 中 <= last_epoch 的元素个数
                for base_lr in self.base_lrs]  # 对优化器中每个参数组的 base_lr 分别计算（列表推导）


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):  # 带预热的 poly 多项式衰减调度器（分割训练最常用）；【注意】同样基于私有基类 _LRScheduler
    # 形参说明: optimizer 优化器; target_lr 最终学习率(默认0); max_iters 最大迭代数; power poly 幂次(默认0.9)
    # 形参说明: warmup_factor 预热起始系数(默认1/3); warmup_iters 预热迭代数(默认500); warmup_method 预热方式; last_epoch 上次迭代号
    def __init__(self, optimizer, target_lr=0, max_iters=0, power=0.9, warmup_factor=1.0 / 3,
                 warmup_iters=500, warmup_method='linear', last_epoch=-1):  # （构造器参数续行）预热迭代数/预热方式/上次迭代号
        if warmup_method not in ("constant", "linear"):  # 先校验预热方式合法
            raise ValueError(  # 预热方式非法时抛出参数错误
                "Only 'constant' or 'linear' warmup_method accepted "  # 报错信息前半段（与下一行字符串自动拼接）
                "got {}".format(warmup_method))  # 非法预热方式报错

        self.target_lr = target_lr  # 最终学习率（poly 衰减的渐近值）
        self.max_iters = max_iters  # 总迭代数上限
        self.power = power  # poly 幂次
        self.warmup_factor = warmup_factor  # 预热起始系数
        self.warmup_iters = warmup_iters  # 预热迭代数
        self.warmup_method = warmup_method  # 预热方式

        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)  # 最后才初始化父类——顺序关键：父类构造会立即调用 get_lr，必须保证以上属性全部就绪（对比 WarmupMultiStepLR 的先 super 后校验，两者顺序相反）

    def get_lr(self):  # 计算当前迭代各参数组的学习率（返回 lr 列表）
        N = self.max_iters - self.warmup_iters  # 衰减阶段总步数（扣除预热部分）；【注意】若 max_iters <= warmup_iters 则 N <= 0，存在除零风险
        T = self.last_epoch - self.warmup_iters  # 衰减阶段已走过的步数
        if self.last_epoch < self.warmup_iters:  # 预热阶段：lr 从 target_lr 侧向 base_lr 爬升
            if self.warmup_method == 'constant':  # 常数预热
                warmup_factor = self.warmup_factor  # 预热期保持固定小系数
            elif self.warmup_method == 'linear':  # 线性预热（此处判断的是 warmup_method，写法正确，对比 WarmupMultiStepLR 中的笔误）
                alpha = float(self.last_epoch) / self.warmup_iters  # 预热进度 alpha ∈ [0,1)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha  # 系数由 warmup_factor 线性升至 1
            else:  # 未知预热方式分支
                raise ValueError("Unknown warmup type.")  # 未知预热方式报错
            return [self.target_lr + (base_lr - self.target_lr) * warmup_factor for base_lr in self.base_lrs]  # 预热期 lr = target_lr + (base_lr - target_lr) * warmup_factor（alpha=0 时约 target_lr + (base_lr-target_lr)/3，随 alpha 增大逼近 base_lr）
        factor = pow(1 - T / N, self.power)  # poly 衰减因子 = (1 - T/N)^power；【注意】若 T > N（迭代超出 max_iters）则底数为负，非整数幂会得到 nan
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]  # lr = target_lr + (base_lr - target_lr) * factor，随迭代衰减并渐近 target_lr


if __name__ == '__main__':  # 直接运行本文件时执行的简易自测代码
    import torch  # 导入 torch 用于构造优化器（与文件顶部导入重复但无害）
    import torch.nn as nn  # 导入 nn 用于构造示例模型

    model = nn.Conv2d(16, 16, 3, 1, 1)  # 示例模型：16→16 通道、stride=1、padding=1 的 3x3 卷积
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam 优化器，初始学习率 0.01
    lr_scheduler = WarmupPolyLR(optimizer, niters=1000)  # 【注意】bug：WarmupPolyLR 没有 niters 形参（应为 max_iters=1000），运行此自测代码会抛出 TypeError
