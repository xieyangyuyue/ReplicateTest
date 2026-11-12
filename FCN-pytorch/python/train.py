# -*- coding: utf-8 -*-
# ============================================================================
# 【文件说明】train.py v2 —— FCN 语义分割网络的训练与验证脚本
# 【运行方式】在仓库根目录（与 CamVid/、python/ 同级）执行：
#   python python/train.py CamVid                     # 默认超参与原注释版完全相同
#   python python/train.py CamVid --amp -e 100        # 开混合精度、只训 100 轮
#   python python/train.py CamVid -b 4 -l 1e-4 --clip 1.0   # 改批大小/学习率/开梯度裁剪
#   python python/train.py Cityscapes --workers 4     # 训练 Cityscapes、4 个读取进程
# 【v2 升级说明】参考你提供的 U-Net 训练脚本（milesial/Pytorch-UNet 风格）吸收的
#   优点，均以【v2】标注；不带任何参数时超参与原版一致：
#   1. 命令行参数（argparse）：轮数/批大小/学习率等不用再改源码（原版硬编码）
#   2. 进度条（tqdm）：训练与验证实时显示进度、速度和当前 loss（未装 tqdm 也能跑）
#   3. 混合精度（--amp）：autocast + GradScaler，GPU 上一般提速 1.5~2 倍、省显存
#   4. 【重要·修复原仓库 bug】训练循环补上 model.train()：原代码只在 val() 里调
#      model.eval()，从头到尾没调过 model.train()，导致 FCNs 的 5 层 BatchNorm
#      训练时也用"从未更新过的全局统计量"（BN 形同虚设）。修复后 BN 用批次统计
#      量归一化并正常更新，通常收敛更快、最终精度更高（训练曲线与原版会不同）
#   5. 验证改在 torch.no_grad() 下进行（原版验证时也建计算图，白费显存和时间）
#   6. scheduler.step() 移到每轮训练结束后调用（原版在轮首调用，新版会警告）
#   7. zero_grad(set_to_none=True)、pin_memory、non_blocking 等提速细节
#   8. 新增最优模型另存：验证 meanIoU 创新高时另存一份 <模型名>_best
#   9. 新增人类可读日志：scores/<配置名>/log.csv，每轮一行（含 32 类完整 IoU），
#      可用 Excel/pandas 直接打开；配合 view_scores.py 可画训练曲线
# 【整体流程】解析命令行 → 读 csv 构建 DataLoader → 搭 VGG+FCNs 模型 → 先跑一次
#             验证记录"未训练基线" → 逐轮训练（进度条 + AMP）→ 每轮验证 IoU 与
#             像素精度 → 保存模型与指标（模型文件 + meanIU.npy + meanPixel.npy + log.csv）
# ============================================================================
from __future__ import print_function  # 兼容 Python2/3 的 print 函数写法

import argparse   # 【v2】命令行参数解析：数据集、轮数、批大小、学习率等均从命令行传入
import os         # 操作系统库：路径拼接、目录创建
import time       # 时间库：统计每轮训练/验证耗时

import numpy as np                    # 数值计算库：保存指标数组（.npy）
import torch                          # PyTorch 主库
import torch.nn as nn                 # 神经网络模块（DataParallel 多卡并行）
import torch.optim as optim           # 优化器模块（RMSprop）
from torch.optim import lr_scheduler  # 学习率调度器（StepLR 按步长衰减）
from torch.utils.data import DataLoader  # 数据加载器：批量读取 + 多进程加速

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs  # 从 fcn.py 导入模型结构（实际用到 VGGNet 与 FCNs，其余保留原仓库的导入）
from CamVid_loader import CamVidDataset              # CamVid 数据集类
from Cityscapes_loader import CityScapesDataset      # Cityscapes 数据集类

# 【v2】进度条：优先使用 tqdm；未安装时退化为下面的"静默替身"（接口兼容、什么都不
# 显示），脚本照样能跑。想要进度条请安装：pip install tqdm
try:  # 尝试导入 tqdm
    from tqdm import tqdm  # 导入成功：使用真正的进度条
except ImportError:  # 导入失败（环境里没有 tqdm）
    class tqdm:  # 静默进度条：与 tqdm 常用接口兼容的空实现
        def __init__(self, iterable=None, **kwargs):  # 构造函数：只记住要迭代的对象
            self.iterable = iterable  # 保存可迭代对象，其余参数忽略
        def __iter__(self):  # 迭代协议
            return iter(self.iterable)  # 直接转发给内部对象
        def set_postfix(self, **kwargs):  # 更新进度条尾部信息（空操作）
            pass  # 什么都不做
        @staticmethod  # 静态方法（不依赖实例）
        def write(msg, **kwargs):  # tqdm.write 的替身：在进度条上方打印一行
            print(msg)  # 退化为普通 print


def get_args():  # 【v2】解析命令行参数（原版的超参全部硬编码在文件开头，改参要改源码）
    parser = argparse.ArgumentParser(description='FCN 语义分割训练（v2：命令行超参 + 进度条 + AMP 混合精度）')  # 参数解析器
    parser.add_argument('dataset', choices=['CamVid', 'Cityscapes'],  # 位置参数：数据集名（原 sys.argv[1]）
                        help='数据集名称：CamVid 或 Cityscapes')  # 说明
    parser.add_argument('-e', '--epochs', type=int, default=500,  # 训练轮数（默认与原版相同）
                        help='总训练轮数（默认 500）')  # 说明
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=6,  # 批大小
                        help='每批图片数（默认 6，与原版相同）')  # 说明
    parser.add_argument('-l', '--learning-rate', dest='lr', type=float, default=1e-4,  # 学习率
                        help='初始学习率（默认 1e-4）')  # 说明
    parser.add_argument('--momentum', type=float, default=0,  # RMSprop 动量
                        help='RMSprop 动量系数（默认 0 与原版相同；参考的 U-Net 脚本用 0.999）')  # 说明
    parser.add_argument('--w-decay', dest='w_decay', type=float, default=1e-5,  # 权重衰减
                        help='权重衰减系数（默认 1e-5）')  # 说明
    parser.add_argument('--step-size', dest='step_size', type=int, default=50,  # 衰减间隔
                        help='学习率每多少轮衰减一次（默认 50）')  # 说明
    parser.add_argument('--gamma', type=float, default=0.5,  # 衰减倍率
                        help='学习率衰减倍率（默认 0.5）')  # 说明
    parser.add_argument('--workers', type=int, default=8,  # 读取进程数
                        help='DataLoader 读取进程数（默认 8；内存不足可调小）')  # 说明
    parser.add_argument('--amp', action='store_true',  # AMP 开关（默认关）
                        help='开启 GPU 混合精度训练（推荐：提速省显存，精度基本无损）')  # 说明
    parser.add_argument('--clip', type=float, default=0.0,  # 梯度裁剪阈值
                        help='梯度裁剪的 L2 范数阈值，0=关闭（参考的 U-Net 脚本默认 1.0）')  # 说明
    return parser.parse_args()  # 解析并返回参数


args = get_args()  # 【v2】解析命令行参数

# ---------- 基本超参（默认值与原注释版完全相同，不加参数时训练行为不变） ----------
dataset    = args.dataset    # 数据集名称：'CamVid' 或 'Cityscapes'
n_class    = 32 if dataset == 'CamVid' else 20  # 类别数：CamVid 32 类；Cityscapes 处理后 20 类（19 有效类 + 兜底类）
batch_size = args.batch_size  # 每批训练图片数量
epochs     = args.epochs     # 总训练轮数
lr         = args.lr         # 初始学习率
momentum   = args.momentum   # RMSprop 动量系数
w_decay    = args.w_decay    # 权重衰减系数（L2 正则，防过拟合）
step_size  = args.step_size  # 学习率衰减间隔：每 step_size 轮衰减一次
gamma      = args.gamma      # 学习率衰减倍率：每次衰减为原来的 gamma 倍
# 把全部超参拼成一个字符串，用作模型文件名和评分目录名，便于区分不同实验配置
configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)  # 打印当前实验配置

# 根据数据集选择数据根目录（原 sys.argv[1] 判断改为上面的 args.dataset）
root_dir   = "CamVid/" if dataset == 'CamVid' else "CityScapes/"  # 数据目录（Cityscapes 目录名沿用原仓库的 CityScapes 拼写）
train_file = os.path.join(root_dir, "train.csv")  # 训练集列表文件（每行：图片路径,标签路径）
val_file   = os.path.join(root_dir, "val.csv")    # 验证集列表文件

# create dir for model
model_dir = "models"               # 模型保存目录
if not os.path.exists(model_dir):  # 若目录不存在
    os.makedirs(model_dir)         # 递归创建（含多级父目录）
model_path = os.path.join(model_dir, configs)  # 模型完整保存路径（以配置串命名，无后缀）

use_gpu = torch.cuda.is_available()         # 检测 GPU 是否可用
num_gpu = list(range(torch.cuda.device_count()))  # 所有可用 GPU 的编号列表，如 [0, 1]

# ---------- 根据数据集构建训练集 Dataset 与 DataLoader ----------
if dataset == 'CamVid':   # CamVid：训练阶段在 Dataset 内部开启随机裁剪 + 随机翻转增广
    train_data = CamVidDataset(csv_file=train_file, phase='train')  # 构建 CamVid 训练集对象
else:                     # Cityscapes：train 阶段同样在类内部强制开启裁剪与翻转
    train_data = CityScapesDataset(csv_file=train_file, phase='train')  # 构建 Cityscapes 训练集对象
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  # 训练加载器：打乱顺序
                          num_workers=args.workers,  # 读取进程数（原版硬编码 8，现可 --workers 调整）
                          pin_memory=use_gpu)  # 【v2】锁页内存：配合 non_blocking 异步搬运提速（仅 GPU 时开启，CPU 上开了也无意义）

# ---------- 构建验证集 Dataset 与 DataLoader ----------
if dataset == 'CamVid':   # CamVid 验证集：不翻转（flip_rate=0）
    val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)  # 构建 CamVid 验证集对象
else:                     # Cityscapes 验证集
    val_data = CityScapesDataset(csv_file=val_file, phase='val', flip_rate=0)  # 构建 Cityscapes 验证集对象
val_loader = DataLoader(val_data, batch_size=1, num_workers=args.workers,  # 验证加载器：批大小 1、不打乱
                        pin_memory=use_gpu)  # 【v2】锁页内存（同上）

vgg_model = VGGNet(requires_grad=True, remove_fc=True)  # VGG 编码器：参数可训练（端到端微调）、删除冗余全连接层
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)  # FCNs 分割网络：以该 VGG 为编码器

if use_gpu:  # 若 GPU 可用：模型搬上显卡并包裹为多卡并行
    ts = time.time()                       # 记录搬运开始时间
    vgg_model = vgg_model.cuda()           # 编码器移入 GPU（vgg_model 已嵌入 fcn_model 内）
    fcn_model = fcn_model.cuda()           # 分割网络整体移入 GPU
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)  # DataParallel 包裹：多卡数据并行
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))  # 打印搬运耗时

criterion = nn.BCEWithLogitsLoss()  # 带 logits 的二值交叉熵损失：内部自带 sigmoid，无需手动激活；
                                    # 把每个像素的每一类都视为独立二分类（多标签分割方式）。
                                    # 【v2】autocast 下该损失会自动用 FP32 计算，数值安全
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)  # RMSprop 优化器
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # 每 step_size 轮把学习率乘 gamma（【v2】改为每轮训练结束后调用，见 train() 内说明）

# ---------- 【v2】混合精度（AMP）配置：写法与参考的 U-Net 脚本一致，兼容新旧版本 ----------
amp_on = args.amp and use_gpu  # AMP 实际开关：命令行开启且 GPU 可用时才生效
if hasattr(torch.amp, 'GradScaler'):  # 新版 PyTorch：GradScaler 移到了 torch.amp 下（需传设备类型）
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp_on)  # 新 API：梯度缩放器（防 FP16 梯度下溢）
else:  # 旧版 PyTorch：在 torch.cuda.amp 下
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp_on)  # 旧 API（enabled=False 时两者都是空操作）
print("AMP 混合精度: {}（加 --amp 可开启）".format("已开启" if amp_on else "关闭"))  # 打印 AMP 状态


def autocast_ctx():  # 【v2】返回当前环境可用的 autocast 上下文管理器（新旧 PyTorch 兼容）
    if hasattr(torch, 'autocast'):  # PyTorch >= 1.10：标准写法
        return torch.autocast(device_type='cuda', enabled=amp_on)  # enabled=False 时为空操作（纯 CPU 环境也安全）
    return torch.cuda.amp.autocast(enabled=amp_on)  # 更旧版本的回退写法


# create dir for score
score_dir = os.path.join("scores", configs)  # 验证指标保存目录（以配置串命名）
if not os.path.exists(score_dir):            # 若目录不存在
    os.makedirs(score_dir)                   # 创建
IU_scores    = np.zeros((epochs, n_class))   # 记录表：每轮各类别的 IoU（即 meanIU.npy 的内容）
pixel_scores = np.zeros(epochs)              # 记录表：每轮的像素精度（即 meanPixel.npy 的内容）
lr_hist   = [''] * epochs                    # 【v2】每轮结束时的学习率（写 log.csv 用；训练前基线轮为空）
loss_hist = [float('nan')] * epochs          # 【v2】每轮平均训练损失（基线轮记 nan）
csv_file  = os.path.join(score_dir, "log.csv")  # 【v2】人类可读的逐轮指标日志（Excel 可直接打开）
best_miou = -1.0  # 【v2】历史最优 meanIoU（初值 -1 保证第一次验证必然刷新，从而另存 _best 模型）

if os.path.exists(os.path.join(score_dir, "meanIU.npy")):  # 【v2】该配置目录下已有旧指标
    print("[提示] {} 下已有历史指标：重新训练会从零开始覆盖（meanIU.npy / meanPixel.npy / log.csv），".format(score_dir))  # 覆盖警告
    print("       如需保留旧结果请先备份或改名该目录（不同超参会自动生成不同的目录名，互不冲突）")  # 给出建议


def train():  # 训练主函数
    for epoch in range(epochs):  # 外层循环：逐个 epoch 训练
        fcn_model.train()  # 【v2·修复原仓库 bug】切回训练模式！原版主流程先跑 val(0)（内部
                           # model.eval()），而训练循环从未调过 model.train()，导致 FCNs 的
                           # 5 层 BatchNorm 全程处于评估模式：永远用初始的 running_mean=0 /
                           # running_var=1 归一化、统计量从不更新（BN 形同虚设）。补上这行后
                           # BN 恢复正常：训练用批次统计量，验证用更新过的运行统计量
        ts = time.time()  # 记录本 epoch 开始时间
        epoch_loss, n_batch = 0.0, 0  # 【v2】本轮损失累计值与批次数（用于算平均损失）
        pbar = tqdm(train_loader, total=len(train_loader),  # 【v2】训练进度条（total=总批数）
                    desc='Epoch %d/%d' % (epoch + 1, epochs), unit='batch')  # 描述与单位
        for iter, batch in enumerate(pbar):  # 内层循环：逐批读取训练数据
            optimizer.zero_grad(set_to_none=True)  # 梯度清零（【v2】set_to_none=True 比原版写法更快）

            if use_gpu:  # GPU 可用：把数据也搬到显卡
                inputs = batch['X'].to('cuda', non_blocking=True)  # 输入图像（异步搬运；原版 Variable 包装自 PyTorch 0.4 起已冗余，故移除）
                labels = batch['Y'].to('cuda', non_blocking=True)   # one-hot 编码的分割标签（异步搬运）
            else:        # CPU 模式：直接使用
                inputs, labels = batch['X'], batch['Y']  # 取出输入与标签

            with autocast_ctx():  # 【v2】混合精度前向（--amp 开启时部分算子用 FP16 计算）
                outputs = fcn_model(inputs)        # 前向传播：得到 (N, n_class, H, W) 的得分图
                loss = criterion(outputs, labels)  # 计算损失（BCEWithLogitsLoss）

            grad_scaler.scale(loss).backward()  # 【v2】缩放损失并反向传播（未开 AMP 时等价于 loss.backward()）
            if args.clip > 0:  # 【v2】梯度裁剪（默认 0 关闭；--clip 1.0 开启可防梯度爆炸）
                grad_scaler.unscale_(optimizer)  # 先把梯度反缩放回真实尺度，裁剪阈值才有意义
                torch.nn.utils.clip_grad_norm_(fcn_model.parameters(), args.clip)  # 按全局 L2 范数裁剪
            grad_scaler.step(optimizer)  # 【v2】优化器更新参数（AMP 下会先检查梯度有无 inf/nan）
            grad_scaler.update()  # 【v2】更新损失缩放因子（未开 AMP 时为空操作）

            epoch_loss += loss.item()  # 【v2】累计本轮损失
            n_batch += 1  # 【v2】批次数 +1
            pbar.set_postfix(loss='%.4f' % loss.item())  # 【v2】进度条右侧实时显示当前批次损失（替代原"每 10 批 print 一次"）

        # ---------- 本轮训练摘要（【v2】替代原版零散的 print，一行看完关键信息） ----------
        avg_loss = epoch_loss / max(n_batch, 1)  # 本轮平均损失（防除零）
        lr_hist[epoch] = optimizer.param_groups[0]['lr']  # 记录本轮使用的学习率
        loss_hist[epoch] = avg_loss  # 记录本轮平均损失
        print('[训练] epoch {} 完成 | 平均 loss {:.4f} | 学习率 {:.2e} | 用时 {:.1f}s'.format(  # 【v2】本轮训练一行摘要
            epoch, avg_loss, optimizer.param_groups[0]['lr'], time.time() - ts))  # 各格式化实参（轮号/平均损失/学习率/耗时）

        torch.save(fcn_model, model_path)  # 保存整个模型（结构+权重）——每轮覆盖，始终是最新
                                          # 【注意】PyTorch>=2.6 加载此类"整个模型"文件需显式
                                          # torch.load(model_path, weights_only=False)，
                                          # predict.py 已处理好该兼容
        val(epoch)  # 在验证集上评估本 epoch 训练效果

        scheduler.step()  # 【v2】学习率衰减移到"本轮训练结束后"调用：新版 PyTorch 要求
                          # scheduler.step() 在 optimizer.step() 之后调用（原版放在轮首会触发
                          # UserWarning，且衰减时机整体提前了一轮，现按正确顺序执行）


def val(epoch):  # 验证函数：在验证集上计算各类 IoU 与像素精度
    global best_miou  # 【v2】引用模块级的最优指标变量（刷新纪录时要更新它）
    fcn_model.eval()  # 切换到评估模式（BN 用运行统计量、关闭 Dropout 等训练行为）
    total_ious = []   # 收集每张验证图的各类 IoU 列表
    pixel_accs = []   # 收集每张验证图的像素精度
    tv = time.time()  # 【v2】验证计时开始
    with torch.no_grad():  # 【v2】验证阶段关闭梯度（原版验证时也在建计算图，浪费显存和时间）
        for batch in tqdm(val_loader, total=len(val_loader),  # 【v2】验证进度条
                          desc='验证 epoch %d' % epoch, unit='img',  # 描述与单位（验证批大小为 1，按图计）
                          leave=False):  # leave=False：验证条走完自动消失，不占屏
            if use_gpu:  # GPU：输入搬上显卡
                inputs = batch['X'].to('cuda', non_blocking=True)  # 异步搬运（【v2】替代原 Variable 包装）
            else:        # CPU：直接使用
                inputs = batch['X']  # 直接使用 CPU 张量

            with autocast_ctx():  # 【v2】验证同样可享混合精度加速
                output = fcn_model(inputs)  # 前向传播得到得分图
            output = output.float().data.cpu().numpy()  # 【v2】先统一转 FP32 再回 CPU 转 numpy（AMP 下输出是 FP16）

            N, _, h, w = output.shape  # 解包批大小 N 与空间尺寸 h×w
            pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)  # 通道维换到最后并展平，逐像素取得分最大的类别，再还原空间形状

            target = batch['l'].cpu().numpy().reshape(N, h, w)  # 取类别索引标签（'l' = label，非 one-hot 版本）
            for p, t in zip(pred, target):  # 逐张图计算指标
                total_ious.append(iou(p, t))        # 各类 IoU
                pixel_accs.append(pixel_acc(p, t))  # 像素精度

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len  # 列表转为数组并转置成 (类别数 × 图片数)
    ious = np.nanmean(total_ious, axis=1)  # 沿图片维求均值（自动忽略 NaN，即忽略验证集中未出现的类）
    pixel_accs = np.array(pixel_accs).mean()  # 所有验证图的平均像素精度
    mean_iou = float(np.nanmean(ious))  # 【v2】本轮 meanIoU 单独存变量（用于最优模型判断）
    print('[验证] epoch {} | 像素精度 pix_acc {:.4f} | 平均IoU meanIoU {:.4f} | 用时 {:.1f}s'.format(  # 【v2】本轮验证一行摘要
        epoch, pixel_accs, mean_iou, time.time() - tv))  # 各格式化实参（轮号/像素精度/平均IoU/耗时）
    print('       完整 {} 类 IoU 见 {}/log.csv；画训练曲线用 view_scores.py'.format(n_class, score_dir))  # 【v2】指路完整数据

    IU_scores[epoch] = ious  # 记录本 epoch 各类 IoU
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)      # 把 IoU 记录数组存盘（.npy）
    pixel_scores[epoch] = pixel_accs  # 记录本 epoch 像素精度
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)  # 把像素精度记录存盘

    # ---------- 【v2】同步写人类可读的 CSV 日志（每轮一行；整体重写，与 npy 内容严格一致） ----------
    with open(csv_file, 'w') as f:  # 覆写式重写（内容始终来自下方两个数组，不会残留旧轮次）
        f.write('epoch,lr,train_loss,pix_acc,meanIoU')  # 表头：汇总列
        for c in range(n_class):  # 再补每类一列
            f.write(',iou_%d' % c)  # 列名 iou_0 ~ iou_31
        f.write('\n')  # 表头行结束
        for e in range(epochs):  # 扫描全部轮次
            if not (pixel_scores[e] != 0 or np.any(IU_scores[e] != 0) or np.any(np.isnan(IU_scores[e]))):  # 全零且无 nan = 该轮尚未验证
                continue  # 跳过未完成的轮
            f.write('%d,%s,%s,%.6f,%.6f' % (e, lr_hist[e], loss_hist[e], pixel_scores[e], np.nanmean(IU_scores[e])))  # 汇总列（lr/loss 在基线轮为空/nan）
            for c in range(n_class):  # 逐类 IoU
                v = IU_scores[e][c]  # 该类该轮的值
                f.write(',%s' % ('nan' if np.isnan(v) else '%.6f' % v))  # nan 原样写出，便于 Excel 筛选
            f.write('\n')  # 该轮一行结束

    # ---------- 【v2】meanIoU 创新高时，另存一份最优模型 ----------
    if mean_iou > best_miou:  # 刷新历史纪录
        best_miou = mean_iou  # 更新纪录值
        torch.save(fcn_model, model_path + '_best')  # 另存最优模型（predict.py 可直接加载）
        print('[保存] meanIoU 创新高 {:.4f} → 已另存最优模型: {}_best'.format(mean_iou, model_path))  # 提示


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
# 借用并修改自 Kaixhin/FCN-semantic-segmentation：计算各类别的交并比（IoU）
def iou(pred, target):  # pred：预测类别图 (H,W)；target：真值类别图 (H,W)
    ious = []  # 保存每个类别的 IoU
    for cls in range(n_class):  # 遍历全部类别
        pred_inds = pred == cls      # 布尔掩码：预测为该类的像素
        target_inds = target == cls  # 布尔掩码：真值为该类的像素
        intersection = pred_inds[target_inds].sum()  # 交集：两者都为该类的像素数
        union = pred_inds.sum() + target_inds.sum() - intersection  # 并集：两者像素数之和减去交集
        if union == 0:  # 并集为 0：该类在预测和真值中均未出现
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
                                     # 记为 NaN，评估时会被 nanmean 忽略
        else:
            ious.append(float(intersection) / max(union, 1))  # IoU = 交集 / 并集（max 防除零）
    return ious  # 返回长度为 n_class 的各类 IoU 列表


def pixel_acc(pred, target):  # 像素精度：预测正确的像素数占总像素数的比例
    correct = (pred == target).sum()  # 预测与真值类别相同的像素数
    total   = (target == target).sum()  # 总像素数（利用 target==target 恒真来计数）
    return correct / total  # 返回精度比值


if __name__ == "__main__":  # 作为主程序运行：先验证初始模型、再开始训练
    val(0)  # show the accuracy before training  # 训练前先验证一次：记录未训练模型的"基线"精度
    train()  # 正式开始训练
