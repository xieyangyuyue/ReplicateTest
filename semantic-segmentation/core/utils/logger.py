# ============================================================================
# 【文件说明】logger.py —— 训练/评估日志初始化工具（移植自 maskrcnn-benchmark）。
# 核心逻辑：setup_logger 为指定 name 获取全局唯一的 logging.Logger：
#   1) logger 级别设为 DEBUG，尽量不丢信息，具体过滤交给各 handler；
#   2) 分布式训练时仅 rank==0 的主进程挂载 handler，其余进程返回“空”
#      logger（没有 handler，调用 info/warn 无处输出），实现非主进程静默；
#   3) 主进程同时输出到 stdout（StreamHandler）与 save_dir/filename
#      （FileHandler），二者共用同一个 Formatter。
# 本仓库在 scripts/train.py、scripts/eval.py 中调用：
#   setup_logger("semantic_segmentation", args.log_dir, get_rank(), ...)。
# 【注意】同一 name 重复调用 setup_logger 会不断叠加 handler，导致日志
#         重复打印多份（本仓库只调用一次，未触发该问题）。
# ============================================================================
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging  # Python 标准日志库
import os  # 拼接日志文件路径、创建日志目录
import sys  # 把日志流定向到标准输出 stdout

__all__ = ['setup_logger']  # 模块对外仅导出 setup_logger 一个接口


# reference from: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/logger.py
# 参数：name——logger 名称（同名复用全局注册表中的同一实例）；
#      save_dir——日志文件保存目录（空值表示只打印不落盘）；
#      distributed_rank——当前进程的分布式 rank（>0 为非主进程，不挂 handler）；
#      filename——日志文件名（默认 log.txt）；mode——写文件模式（'w' 覆盖 / 'a+' 追加）
def setup_logger(name, save_dir, distributed_rank, filename="log.txt", mode='w'):
    logger = logging.getLogger(name)  # 按名称获取（或首次创建）全局 logger 实例
    logger.setLevel(logging.DEBUG)  # logger 自身阈值设为最低级 DEBUG，过滤交给 handler 决定
    # don't log results for the non-master process
    if distributed_rank > 0:  # 非主进程（rank>0）
        return logger  # 直接返回未挂任何 handler 的 logger：后续 log 调用因无 handler 而静默
    ch = logging.StreamHandler(stream=sys.stdout)  # 控制台 handler，输出流绑定到 stdout
    ch.setLevel(logging.DEBUG)  # 控制台输出 DEBUG 及以上所有级别
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")  # 日志格式：时间 名称 级别: 消息
    ch.setFormatter(formatter)  # 控制台 handler 套用该格式器
    logger.addHandler(ch)  # 把控制台 handler 挂载到 logger

    if save_dir:  # 指定了保存目录才写日志文件
        if not os.path.exists(save_dir):  # 目录尚不存在
            os.makedirs(save_dir)  # 递归创建日志目录
        # 文件 handler：写入 save_dir/filename，mode 决定覆盖('w')还是追加('a+')
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)  # 文件同样记录 DEBUG 及以上级别
        fh.setFormatter(formatter)  # 文件 handler 复用同一个格式器
        logger.addHandler(fh)  # 挂载文件 handler

    return logger  # 返回配置好的 logger（主进程：控制台+文件双输出；非主进程：无输出）
