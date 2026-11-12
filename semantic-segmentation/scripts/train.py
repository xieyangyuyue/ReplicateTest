# ============================================================================
# 【文件说明】train.py —— PyTorch 语义分割训练主脚本（训练入口）。
#               支持 fcn/psp/deeplabv3+/danet/bisenet 等 20 余种分割网络与
#               vgg/resnet/densenet 等 backbone 的组合，覆盖 pascal_voc、
#               pascal_aug、ade20k、citys、sbu 五个数据集；包含 warmup+poly
#               学习率调度、OHEM 与辅助损失、SyncBN/DDP 分布式训练、
#               checkpoint 保存与日志管理。
# 【运行方式】单卡：python train.py --model fcn --backbone resnet50 --dataset pascal_voc
#               多卡：python -m torch.distributed.launch --nproc_per_node=N train.py ...
#               （脚本会自动把项目根目录加入 sys.path，工作目录不限）
# 【整体流程】parse_args() 解析命令行参数 → __main__ 检测 CUDA 与环境变量
#               WORLD_SIZE，初始化 cudnn / NCCL 分布式，并按 GPU 数线性放大
#               学习率 → Trainer() 构建数据增强与 DataLoader、分割网络、损失
#               函数、SGD 优化器与 WarmupPolyLR 调度器 → train() 逐迭代训练，
#               定期打印日志、保存 checkpoint、执行验证 → validation() 累积
#               计算 pixAcc/mIoU，创新高时另存 best_model → 训练结束打印总
#               耗时并清空显存缓存。
# ============================================================================
import argparse  # 命令行参数解析库，用于构建训练脚本的参数解析器 parse_args()
import time  # 时间库，用于统计训练耗时并估算剩余时间（ETA）
import datetime  # 日期时间库，将耗时秒数格式化为 "时:分:秒" 字符串
import os  # 操作系统接口，用于路径处理、目录创建与权重文件读写
import shutil  # 高级文件操作库，用于把最优模型 checkpoint 复制为 best_model 文件
import sys  # 系统功能，用于把项目根目录追加进模块搜索路径

cur_path = os.path.abspath(os.path.dirname(__file__))  # 获取当前脚本所在目录（scripts/）的绝对路径
root_path = os.path.split(cur_path)[0]  # 取其上一级目录，即项目根目录
sys.path.append(root_path)  # 将项目根目录加入 sys.path，保证 core 包可被导入

import torch  # PyTorch 主库：张量计算、自动求导与 GPU 训练
import torch.nn as nn  # 神经网络模块库：SyncBatchNorm、DistributedDataParallel 等
import torch.utils.data as data  # 数据工具库：主要使用 data.DataLoader 构建训练/验证加载器
import torch.backends.cudnn as cudnn  # cuDNN 后端接口：开启 benchmark 自动调优以加速训练

from torchvision import transforms  # 图像变换库：构建 ToTensor + Normalize 输入预处理流水线
from core.data.dataloader import get_segmentation_dataset  # 数据集工厂函数：按名称创建训练/验证 Dataset
from core.models.model_zoo import get_segmentation_model  # 模型工厂函数：按名称创建分割网络并装配 backbone
from core.utils.loss import get_segmentation_loss  # 损失工厂函数：创建含可选 OHEM/辅助损失的 criterion
from core.utils.distributed import *  # 分布式工具集：synchronize、get_rank、reduce_loss_dict、采样器构建等
from core.utils.logger import setup_logger  # 日志初始化函数：训练日志同时输出到控制台与文件
from core.utils.lr_scheduler import WarmupPolyLR  # 学习率调度器：先 warmup 再按 poly(power=0.9) 衰减
from core.utils.score import SegmentationMetric  # 分割指标类：基于混淆矩阵累积计算 pixAcc 与 mIoU


def parse_args():  # 定义命令行参数解析函数（eval.py 也会 import 复用本函数）
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')  # 创建解析器并写明脚本用途
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',  # 分割网络名称，默认 'fcn'
                        choices=['fcn32s', 'fcn16s', 'fcn8s', 'fcn', 'psp', 'deeplabv3',  # 可选项：经典 FCN 系列
                            'deeplabv3_plus', 'danet', 'denseaspp', 'bisenet', 'encnet',  # 可选项：DeepLabV3+、DANet、DenseASPP、BiSENet、EncNet
                            'dunet', 'icnet', 'enet', 'ocnet', 'psanet', 'cgnet', 'espnet',  # 可选项：DUNet、ICNet、ENet、OCNet、PSANet、CGNet、ESPNet
                            'lednet', 'dfanet','swnet'],  # 可选项：LEDNet、DFANet、SWNet
                        help='model name (default: fcn32s)')  # 【注意】help 写的默认值 fcn32s 与实际 default='fcn' 不一致
    parser.add_argument('--backbone', type=str, default='resnet50',  # backbone（骨干特征提取网络）名称，默认 'resnet50'
                        choices=['vgg16', 'resnet18', 'resnet50', 'resnet101', 'resnet152',  # 可选项：VGG16 与 ResNet 系列
                            'densenet121', 'densenet161', 'densenet169', 'densenet201'],  # 可选项：DenseNet 系列
                        help='backbone name (default: vgg16)')  # 【注意】help 写的默认值 vgg16 与实际 default='resnet50' 不一致
    parser.add_argument('--dataset', type=str, default='pascal_voc',  # 训练/验证所用数据集名称，默认 'pascal_voc'
                        choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys', 'sbu'],  # 可选项：VOC2012、SBD(VOC-Aug)、ADE20K、Cityscapes、SBU
                        help='dataset name (default: pascal_voc)')  # 帮助文本：数据集名称（与实际默认值一致）
    parser.add_argument('--base-size', type=int, default=520,  # 数据增强基准尺寸（像素）：训练前先把图像缩放到该尺寸附近再随机裁剪
                        help='base image size')  # 帮助文本：基准图像尺寸
    parser.add_argument('--crop-size', type=int, default=480,  # 随机裁剪尺寸（像素）：训练时裁出该大小的图块送入网络
                        help='crop image size')  # 帮助文本：裁剪图像尺寸
    parser.add_argument('--workers', '-j', type=int, default=4,  # DataLoader 数据加载进程数，默认 4
                        metavar='N', help='dataloader threads')  # 帮助文本：数据加载线程数
    # training hyper params
    parser.add_argument('--jpu', action='store_true', default=False,  # 是否启用 JPU（Joint Pyramid Upsampling，联合金字塔上采样）模块，默认关闭
                        help='JPU')  # 帮助文本：JPU 开关（高效融合多尺度特征，常配合 DeepLabV3+ 等）
    parser.add_argument('--use-ohem', type=bool, default=False,  # 是否使用 OHEM（在线难例挖掘）损失，主要针对 cityscapes，默认关闭
                        help='OHEM Loss for cityscapes dataset')  # 【注意】type=bool 是 argparse 经典误用：任意非空字符串（含 "False"）都会解析为 True，命令行无法传 False
    parser.add_argument('--aux', action='store_true', default=False,  # 是否启用辅助分类头（auxiliary loss），默认关闭
                        help='Auxiliary loss')  # 帮助文本：辅助损失开关（对 backbone 中间层加监督，助收敛）
    parser.add_argument('--aux-weight', type=float, default=0.4,  # 辅助损失加权系数，默认 0.4
                        help='auxiliary loss weight')  # 帮助文本：辅助损失权重
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',  # 训练批大小（每 GPU 样本数），默认 4
                        help='input batch size for training (default: 8)')  # 【注意】help 写的默认值 8 与实际 default=4 不一致
    parser.add_argument('--start_epoch', type=int, default=0,  # 起始 epoch 编号，默认 0
                        metavar='N', help='start epochs (default:0)')  # 【注意】该参数定义后从未被使用（训练按 iteration 计数，断点续训也未用到）
    parser.add_argument('--epochs', type=int, default=50, metavar='N',  # 总训练轮数（epoch），默认 50
                        help='number of epochs to train (default: 50)')  # 帮助文本：训练轮数
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',  # 初始学习率，默认 1e-4（多卡时还会按 GPU 数线性放大）
                        help='learning rate (default: 1e-4)')  # 帮助文本：学习率
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',  # SGD 动量系数，默认 0.9
                        help='momentum (default: 0.9)')  # 帮助文本：动量
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',  # 权重衰减（L2 正则）系数，默认 1e-4
                        help='w-decay (default: 5e-4)')  # 【注意】help 写的默认值 5e-4 与实际 default=1e-4 不一致
    parser.add_argument('--warmup-iters', type=int, default=0,  # 学习率 warmup 的迭代数，默认 0（即不启用 warmup）
                        help='warmup iters')  # 帮助文本：warmup 迭代数
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,  # warmup 起始学习率系数：warmup 阶段 lr = warmup_factor * lr
                        help='lr = warmup_factor * lr')  # 帮助文本：warmup 起始系数（默认 1/3）
    parser.add_argument('--warmup-method', type=str, default='linear',  # warmup 策略：linear（线性升至基准 lr）或 constant（常数）
                        help='method of warmup')  # 帮助文本：warmup 方法
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,  # 禁用 CUDA 的开关，默认 False（默认尽量用 GPU）
                        help='disables CUDA training')  # 帮助文本：禁用 CUDA 训练
    parser.add_argument('--local_rank', type=int, default=0)  # 当前进程使用的 GPU 序号（由 torch.distributed.launch 自动注入），单卡默认 0
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,  # 断点续训权重文件路径，默认不恢复
                        help='put the path to resuming file if needed')  # 帮助文本：续训文件路径
    parser.add_argument('--save-dir', default='~/.torch/models',  # checkpoint 保存目录（支持 ~ 展开），默认 ~/.torch/models
                        help='Directory for saving checkpoint models')  # 帮助文本：模型保存目录
    parser.add_argument('--save-epoch', type=int, default=10,  # 每隔多少 epoch 保存一次 checkpoint，默认 10
                        help='save model every checkpoint-epoch')  # 帮助文本：保存间隔
    parser.add_argument('--log-dir', default='../runs/logs/',  # 日志文件保存目录，默认 ../runs/logs/
                        help='Directory for saving checkpoint models')  # 【注意】help 文本复制自 --save-dir，该目录实际保存的是日志而非模型
    parser.add_argument('--log-iter', type=int, default=10,  # 每隔多少次迭代打印一次训练日志，默认 10
                        help='print log every log-iter')  # 帮助文本：日志打印间隔
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,  # 每隔多少 epoch 验证一次，默认 1（每轮都验证）
                        help='run validation every val-epoch')  # 帮助文本：验证间隔
    parser.add_argument('--skip-val', action='store_true', default=False,  # 跳过训练期间验证的开关，默认 False
                        help='skip validation during training')  # 帮助文本：跳过验证
    args = parser.parse_args()  # 解析命令行参数，返回命名空间对象

    # default settings for epochs, batch_size and lr
    # 【注意】以下两个 if 分支均为死代码：--epochs 与 --lr 已有非 None 默认值，条件永假，永不执行
    if args.epochs is None:  # 若未指定 epoch 数则按数据集查表自动选择（因默认值非 None，实际永不触发）
        epoches = {  # 各数据集推荐训练轮数查找表（原代码变量名 epoches 为 epochs 拼写笔误，保持原样）
            'coco': 30,  # COCO 数据集：30 轮
            'pascal_aug': 80,  # VOC 增强集（SBD）：80 轮
            'pascal_voc': 50,  # VOC2012：50 轮
            'pcontext': 80,  # PASCAL Context：80 轮
            'ade20k': 160,  # ADE20K：160 轮
            'citys': 120,  # Cityscapes：120 轮
            'sbu': 160,  # SBU：160 轮
        }  # epoch 查找表定义结束
        args.epochs = epoches[args.dataset.lower()]  # 按数据集名（转小写）查表设置总轮数
    if args.lr is None:  # 若未指定学习率则按数据集查表并做 batch 缩放（因默认值非 None，实际永不触发）
        lrs = {  # 各数据集参考学习率查找表（基准为 batch_size=8）
            'coco': 0.004,  # COCO：0.004
            'pascal_aug': 0.001,  # VOC 增强集：0.001
            'pascal_voc': 0.0001,  # VOC2012：0.0001
            'pcontext': 0.001,  # PASCAL Context：0.001
            'ade20k': 0.01,  # ADE20K：0.01
            'citys': 0.01,  # Cityscapes：0.01
            'sbu': 0.001,  # SBU：0.001
        }  # 学习率查找表定义结束
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size  # 线性缩放规则：以 batch=8 为基准按实际 batch 大小调整学习率
    return args  # 返回解析完成的参数对象


class Trainer(object):  # 训练器类：封装数据/模型/优化器构建以及训练、验证全流程
    def __init__(self, args):  # 构造函数：完成训练前全部组件的初始化
        self.args = args  # 保存命令行参数（后续会向其中补充 iters_per_epoch 等派生量）
        self.device = torch.device(args.device)  # 根据参数选定计算设备（"cuda" 或 "cpu"）

        # image transform
        input_transform = transforms.Compose([  # 组合输入图像预处理流水线
            transforms.ToTensor(),  # PIL 图像转张量，像素值由 [0,255] 归一化到 [0,1]
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # 按 ImageNet 训练集均值/标准差做标准化
        ])  # 预处理流水线定义结束
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}  # 数据集公共构造参数：预处理、基准尺寸、裁剪尺寸
        train_dataset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)  # 创建训练集：train 模式含随机翻转/缩放/裁剪等增强
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)  # 创建验证集：val 模式仅缩放与中心裁剪，不做随机增强
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)  # 每个 epoch 的迭代数 = 样本数整除（GPU 数 × batch size）
        args.max_iters = args.epochs * args.iters_per_epoch  # 总迭代数 = 轮数 × 每轮迭代数（学习率按迭代而非 epoch 调度）

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)  # 训练采样器：打乱样本顺序，分布式时按 rank 划分数据
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)  # 训练批采样器：组 batch 并保证总迭代数恰为 max_iters
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)  # 验证采样器：顺序取样不打乱，分布式时按 rank 划分
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.batch_size)  # 验证批采样器：不设总步数，遍历整个验证集

        self.train_loader = data.DataLoader(dataset=train_dataset,  # 训练数据加载器：绑定训练数据集
                                            batch_sampler=train_batch_sampler,  # 用批采样器控制取样顺序与组批
                                            num_workers=args.workers,  # 多进程并行加载与预处理数据
                                            pin_memory=True)  # 锁页内存，加速 CPU→GPU 数据搬运
        self.val_loader = data.DataLoader(dataset=val_dataset,  # 验证数据加载器：绑定验证数据集
                                          batch_sampler=val_batch_sampler,  # 用验证批采样器顺序取样
                                          num_workers=args.workers,  # 多进程并行加载
                                          pin_memory=True)  # 锁页内存加速传输

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d  # 分布式用跨卡同步 BN（统计量全局一致），单机用普通 BN
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,  # 模型工厂创建分割网络（按数据集自动确定输出类别数）
                                            aux=args.aux, jpu=args.jpu, norm_layer=BatchNorm2d).to(self.device)  # 传入辅助头/JPU/BN 层类型，并将模型搬到计算设备

        # resume checkpoint if needed
        if args.resume:  # 指定了 --resume 时恢复权重
            if os.path.isfile(args.resume):  # 先校验路径确为存在的文件
                name, ext = os.path.splitext(args.resume)  # 拆分文件路径的名称与扩展名
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'  # 【注意】断言逻辑有 bug：应为 ext in ('.pkl', '.pth')；此处非空字符串 '.pth' 恒为真，任意扩展名都能通过校验
                print('Resuming training, loading {}...'.format(args.resume))  # 打印恢复训练提示信息
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))  # 加载权重到当前设备；【注意】仅恢复模型权重，未恢复优化器状态与训练进度，并非完整断点续训

        # create criterion
        self.criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,  # 损失工厂创建 criterion：可含 OHEM 损失与辅助损失
                                               aux_weight=args.aux_weight, ignore_index=-1).to(self.device)  # 辅助损失权重；忽略标签为 -1 的像素（边界/未标注区域）

        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()  # 优化器参数组列表：为不同模块配置不同学习率
        if hasattr(self.model, 'pretrained'):  # 模型含 pretrained 属性（即 backbone 子模块）时
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': args.lr})  # backbone 参数组：使用基准学习率（预训练微调）
        if hasattr(self.model, 'exclusive'):  # 模型声明 exclusive 属性（需独立设置学习率的模块名列表）时
            for module in self.model.exclusive:  # 遍历这些模块名（如分类头、辅助头）
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': args.lr * 10})  # 新增模块参数组：学习率为基准的 10 倍（新层从零训练需更大学习率）
        self.optimizer = torch.optim.SGD(params_list,  # 使用 SGD 优化器并传入多个参数组
                                         lr=args.lr,  # 各参数组未单独指定 lr 时使用的基准学习率
                                         momentum=args.momentum,  # 动量系数 0.9，加速收敛
                                         weight_decay=args.weight_decay)  # 权重衰减 1e-4，抑制过拟合

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,  # 学习率调度器绑定优化器
                                         max_iters=args.max_iters,  # 调度总迭代数（poly 衰减的时间基准 T）
                                         power=0.9,  # poly 衰减指数：lr = base_lr * (1 - t/T)^0.9
                                         warmup_factor=args.warmup_factor,  # warmup 起始学习率系数（默认 1/3）
                                         warmup_iters=args.warmup_iters,  # warmup 迭代数（默认 0 即不做 warmup）
                                         warmup_method=args.warmup_method)  # warmup 方式：linear 或 constant

        if args.distributed:  # 分布式训练时包装模型
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],  # 包装为 DDP：反向传播时自动跨卡 all-reduce 梯度
                                                             output_device=args.local_rank)  # 指定输出设备为本进程的 GPU

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)  # 创建分割指标：类别数取自训练集的 num_class 属性

        self.best_pred = 0.0  # 历史最优验证得分（pixAcc 与 mIoU 的均值），初始为 0

    def train(self):  # 训练主循环：按迭代（iteration）而非 epoch 驱动
        save_to_disk = get_rank() == 0  # 仅 0 号进程写日志/保存 checkpoint，避免多进程重复写文件
        epochs, max_iters = self.args.epochs, self.args.max_iters  # 取出总轮数与总迭代数
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch  # 日志打印间隔；验证间隔（epoch 数换算为迭代数）
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch  # checkpoint 保存间隔（epoch 数换算为迭代数）
        start_time = time.time()  # 记录训练开始时刻，用于统计耗时
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))  # 打印训练总规模（轮数与总迭代数）

        self.model.train()  # 切换训练模式：BN 使用批统计量，Dropout 生效
        for iteration, (images, targets, _) in enumerate(self.train_loader):  # 遍历训练数据；第三个返回值（文件名列表）以下划线丢弃
            iteration = iteration + 1  # 迭代号从 1 开始计数，便于取模判断打印/保存/验证时机
            self.lr_scheduler.step()  # 每个迭代更新一次学习率（warmup + poly 按迭代步进）；【注意】首轮先于 optimizer.step() 调用，新版 PyTorch 可能触发调用顺序警告

            images = images.to(self.device)  # 图像批次搬运到计算设备
            targets = targets.to(self.device)  # 标签批次搬运到计算设备

            outputs = self.model(images)  # 前向传播：返回输出列表，outputs[0] 为主预测，开启 aux 时含 outputs[1] 辅助输出
            loss_dict = self.criterion(outputs, targets)  # 计算损失，返回形如 {'loss': …, 'aux_loss': …} 的字典

            losses = sum(loss for loss in loss_dict.values())  # 汇总各损失项得到总损失（用于反向传播）

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)  # 分布式下跨 GPU 归约（求平均）各项损失，仅用于日志显示
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())  # 汇总归约后的总损失（仅日志显示用，不参与反传）

            self.optimizer.zero_grad()  # 清空上一迭代残留的梯度
            losses.backward()  # 对总损失反向传播求梯度
            self.optimizer.step()  # 优化器按各参数组的学习率更新参数

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)  # 按当前平均迭代耗时估算剩余训练秒数
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))  # 格式化为 "时:分:秒" 形式的 ETA 字符串

            if iteration % log_per_iters == 0 and save_to_disk:  # 到达日志间隔且为主进程时打印训练日志
                logger.info(  # 记录迭代进度/当前学习率/归约后损失/已用时间/预计剩余时间
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(  # 训练日志格式串：迭代进度/学习率/损失/已用时间/预计剩余时间
                        iteration, max_iters, self.optimizer.param_groups[0]['lr'], losses_reduced.item(),  # 打印第 0 个参数组（backbone）的学习率与标量化后的损失
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))  # 已用时间字符串与预计剩余时间

            if iteration % save_per_iters == 0 and save_to_disk:  # 到达保存间隔且为主进程时保存 checkpoint
                save_checkpoint(self.model, self.args, is_best=False)  # 常规保存（不标记为最优）

            if not self.args.skip_val and iteration % val_per_iters == 0:  # 未跳过验证且到达验证间隔时执行验证
                self.validation()  # 在验证集上评估 pixAcc/mIoU，若创新高则保存 best_model
                self.model.train()  # 验证用的 eval 模式会改变 BN/Dropout 行为，验证后须切回训练模式

        save_checkpoint(self.model, self.args, is_best=False)  # 训练循环结束后再保存一次最终权重
        total_training_time = time.time() - start_time  # 统计训练总耗时（秒）
        total_training_str = str(datetime.timedelta(seconds=total_training_time))  # 格式化为 "时:分:秒" 字符串
        logger.info(  # 打印训练总耗时与平均每次迭代耗时
            "Total training time: {} ({:.4f}s / it)".format(  # 训练总耗时与平均单次迭代耗时的格式串
                total_training_str, total_training_time / max_iters))  # 平均耗时 = 总耗时 / 总迭代数

    def validation(self):  # 验证函数：在验证集上累积计算 pixAcc 与 mIoU
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False  # 是否为最优模型的标记，默认 False
        self.metric.reset()  # 清空指标内部累积的混淆矩阵
        if self.args.distributed:  # 分布式时模型被 DDP 包装
            model = self.model.module  # 取出 DDP 内部的原始模型用于前向（省去 DDP 包装层的开销）
        else:  # 单机（非分布式）分支：直接使用原始模型
            model = self.model  # 单机直接使用模型本身
        # 清空 CUDA 缓存显存，为验证阶段的推理腾出空间（原注释自带 TODO：待确认是否真有帮助）
        torch.cuda.empty_cache()  # TODO check if helps
        model.eval()  # 切换评估模式：BN 使用运行统计量，Dropout 关闭
        for i, (image, target, filename) in enumerate(self.val_loader):  # 逐批遍历验证集（filename 为文件名列表，此处未使用）
            image = image.to(self.device)  # 图像搬运到计算设备
            target = target.to(self.device)  # 标签搬运到计算设备

            with torch.no_grad():  # 关闭梯度追踪，节省显存并加速推理
                outputs = model(image)  # 前向推理得到输出列表
            self.metric.update(outputs[0], target)  # 用主输出（outputs[0]）与标签更新混淆矩阵
            pixAcc, mIoU = self.metric.get()  # 获取到目前为止累积的像素精度与平均 IoU
            logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))  # 逐批打印当前累计指标（0~1 小数，非百分比）

        new_pred = (pixAcc + mIoU) / 2  # 【注意】以 pixAcc 与 mIoU 的简单平均作为选优标准；若验证集为空，pixAcc/mIoU 未定义会抛 NameError
        if new_pred > self.best_pred:  # 打破历史最优记录时
            is_best = True  # 标记本次保存为最优
            self.best_pred = new_pred  # 更新历史最优得分
        save_checkpoint(self.model, self.args, is_best)  # 保存 checkpoint（is_best=True 时额外复制出 best_model 文件）
        synchronize()  # 分布式屏障：等待所有进程完成本轮流验证后再继续训练


def save_checkpoint(model, args, is_best=False):  # 将模型权重保存到磁盘；is_best=True 时另存最优副本
    """Save Checkpoint"""  # 原英文 docstring：保存断点
    directory = os.path.expanduser(args.save_dir)  # 展开保存目录中的 "~" 为用户主目录
    if not os.path.exists(directory):  # 保存目录不存在时
        os.makedirs(directory)  # 递归创建目录（含多级父目录）
    filename = '{}_{}_{}.pth'.format(args.model, args.backbone, args.dataset)  # 常规权重文件名：模型_骨干_数据集.pth
    filename = os.path.join(directory, filename)  # 拼出完整保存路径

    if args.distributed:  # 分布式时模型被 DDP 包装
        model = model.module  # 取出原始模型再保存，避免 state_dict 键名带 "module." 前缀
    torch.save(model.state_dict(), filename)  # 序列化保存模型参数（仅权重，不含优化器/调度器状态）
    if is_best:  # 若为历史最优模型
        best_filename = '{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.dataset)  # 最优权重文件名：…_best_model.pth
        best_filename = os.path.join(directory, best_filename)  # 拼出最优模型完整路径
        shutil.copyfile(filename, best_filename)  # 将刚保存的权重复制一份作为 best_model 文件


if __name__ == '__main__':  # 脚本直接执行时的入口（被 import 时不运行）
    args = parse_args()  # 解析命令行参数

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1  # 从环境变量读进程总数（分布式启动器注入），未注入则视为单机单卡
    args.num_gpus = num_gpus  # 存入参数对象，供 Trainer 计算每轮迭代数使用
    args.distributed = num_gpus > 1  # 进程数大于 1 即启用分布式训练
    if not args.no_cuda and torch.cuda.is_available():  # 未禁用 CUDA 且 GPU 可用时
        cudnn.benchmark = True  # 开启 cuDNN 自动调优：输入尺寸固定时可显著提速（尺寸多变时可能适得其反）
        args.device = "cuda"  # 使用 GPU 训练
    else:  # 无 GPU 或禁用 CUDA 分支：改用 CPU 训练并关闭分布式
        args.distributed = False  # CPU 环境强制关闭分布式
        args.device = "cpu"  # 使用 CPU 训练
    if args.distributed:  # 初始化分布式训练环境
        torch.cuda.set_device(args.local_rank)  # 绑定当前进程使用的 GPU
        torch.distributed.init_process_group(backend="nccl", init_method="env://")  # 以 NCCL 后端、环境变量寻址方式初始化进程组
        synchronize()  # 等待全部进程完成初始化
    args.lr = args.lr * num_gpus  # 【注意】学习率按 GPU 数线性放大（linear scaling rule）：多卡实际学习率 = 单卡 lr × N

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_log.txt'.format(  # 初始化全局日志器：输出到 log_dir，文件名含模型/骨干/数据集；train()/validation() 中引用的 logger 即此全局变量
        args.model, args.backbone, args.dataset))  # 日志文件名格式：模型_骨干_数据集_log.txt
    logger.info("Using {} GPUs".format(num_gpus))  # 记录使用的 GPU 数量
    logger.info(args)  # 记录完整参数配置，便于复现实验

    trainer = Trainer(args)  # 创建训练器实例（完成数据/模型/优化器等全部初始化）
    trainer.train()  # 启动训练主循环
    torch.cuda.empty_cache()  # 训练结束后清空 CUDA 缓存显存
