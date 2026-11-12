# ============================================================================
# 【文件说明】eval.py —— 语义分割模型评估脚本：在验证集上测试已训练模型，
#               逐样本累积计算 pixAcc（像素精度）与 mIoU（平均交并比）指标，
#               并将预测结果按数据集调色板着色后保存为 PNG 图片。
# 【运行方式】python eval.py --model fcn --backbone resnet50 --dataset pascal_voc
#               （参数解析复用 train.py 的 parse_args，训练/评估参数保持一致；
#                分布式评估方式与 train.py 相同：torch.distributed.launch 启动）
# 【整体流程】parse_args() 解析参数（来自 train.py）→ __main__ 初始化 CUDA/
#               分布式环境并创建预测图输出目录 → Evaluator() 构建验证集
#               DataLoader（batch=1、不做随机增强）与分割网络 → eval() 以
#               no_grad 前向推理，SegmentationMetric 累积混淆矩阵并逐样本打印
#               指标 → 对预测 logits 做 argmax 得类别图，get_color_pallete
#               着色后保存到 ../runs/pred_pic/ 目录 → synchronize 等待全部
#               进程结束后清空显存缓存。
# ============================================================================
from __future__ import print_function  # 兼容 Python2 的 print 函数语法（历史遗留，Python3 下无实际影响）

import os  # 操作系统接口：路径拼接、输出目录创建等
import sys  # 系统功能：把项目根目录加入模块搜索路径

cur_path = os.path.abspath(os.path.dirname(__file__))  # 获取当前脚本所在目录（scripts/）的绝对路径
root_path = os.path.split(cur_path)[0]  # 取其上一级目录，即项目根目录
sys.path.append(root_path)  # 将项目根目录加入 sys.path，保证 core 包可被导入

import torch  # PyTorch 主库：张量计算与 GPU 推理
import torch.nn as nn  # 神经网络模块库：SyncBatchNorm、DistributedDataParallel 等
import torch.utils.data as data  # 数据工具库：使用 data.DataLoader 构建验证数据加载器
import torch.backends.cudnn as cudnn  # cuDNN 后端接口：开启 benchmark 自动调优加速推理

from torchvision import transforms  # 图像变换库：构建 ToTensor + Normalize 输入预处理
from core.data.dataloader import get_segmentation_dataset  # 数据集工厂函数：按名称创建验证集 Dataset
from core.models.model_zoo import get_segmentation_model  # 模型工厂函数：按名称创建分割网络并加载训练好的权重
from core.utils.score import SegmentationMetric  # 分割指标类：基于混淆矩阵累积计算 pixAcc 与 mIoU
from core.utils.visualize import get_color_pallete  # 可视化工具：将类别索引图按数据集调色板着色为 PIL 彩色图
from core.utils.logger import setup_logger  # 日志初始化函数：评估日志同时输出到控制台与文件
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler  # 分布式工具：进程同步、rank 获取与采样器构建

from train import parse_args  # 复用 train.py 的命令行参数解析，保证训练/评估参数口径一致


class Evaluator(object):  # 评估器类：封装验证数据加载、模型构建与评估流程
    def __init__(self, args):  # 构造函数：初始化评估所需的全部组件
        self.args = args  # 保存命令行参数
        self.device = torch.device(args.device)  # 根据参数选定计算设备（"cuda" 或 "cpu"）

        # image transform
        input_transform = transforms.Compose([  # 组合输入图像预处理流水线
            transforms.ToTensor(),  # PIL 图像转张量，像素值由 [0,255] 归一化到 [0,1]
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # 按 ImageNet 训练集均值/标准差做标准化
        ])  # 预处理流水线定义结束

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval', transform=input_transform)  # 创建验证集：testval 模式仅做缩放对齐尺寸，不做随机增强
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)  # 顺序采样器（不打乱），分布式时按 rank 划分数据
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)  # 批采样器：每批 1 张图，便于逐样本保存预测图
        self.val_loader = data.DataLoader(dataset=val_dataset,  # 验证数据加载器：绑定验证数据集
                                          batch_sampler=val_batch_sampler,  # 用批采样器控制取样顺序
                                          num_workers=args.workers,  # 多进程并行加载数据
                                          pin_memory=True)  # 锁页内存，加速 CPU→GPU 数据搬运

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d  # 分布式用跨卡同步 BN，单机用普通 BN
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,  # 模型工厂创建分割网络（按数据集自动确定类别数）
                                            aux=args.aux, pretrained=True, pretrained_base=False,  # pretrained=True 加载训练好的分割头权重；pretrained_base=False 不再下载 backbone 的 ImageNet 预训练权重
                                            local_rank=args.local_rank,  # 【注意】评估并不需要 local_rank，仅为兼容部分模型构造函数的传参而保留
                                            norm_layer=BatchNorm2d).to(self.device)  # 指定 BN 层类型，并将模型搬到计算设备
        if args.distributed:  # 分布式评估时包装模型
            self.model = nn.parallel.DistributedDataParallel(self.model,  # 包装为 DDP：推理虽不需要梯度同步，但配合分布式采样器切分数据
                device_ids=[args.local_rank], output_device=args.local_rank)  # 指定当前进程的输入/输出 GPU
        self.model.to(self.device)  # 【注意】DDP 包装后又执行一次 .to(device)，属冗余写法（无害）

        self.metric = SegmentationMetric(val_dataset.num_class)  # 按验证集类别数创建分割评价指标

    def eval(self):  # 评估主流程：遍历验证集，累积指标并保存预测图
        self.metric.reset()  # 重置混淆矩阵等累积状态
        self.model.eval()  # 切换评估模式：BN 使用运行统计量，Dropout 关闭
        if self.args.distributed:  # 分布式时模型被 DDP 包装
            model = self.model.module  # 取出 DDP 内部原始模型用于前向
        else:  # 单机（非分布式）分支：直接使用原始模型
            model = self.model  # 单机直接使用模型本身
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))  # 打印验证集批次数（batch=1 时即样本数）
        for i, (image, target, filename) in enumerate(self.val_loader):  # 逐样本遍历验证集（filename 为文件名列表，供保存预测图用）
            image = image.to(self.device)  # 图像搬运到计算设备
            target = target.to(self.device)  # 标签搬运到计算设备

            with torch.no_grad():  # 关闭梯度追踪，节省显存并加速推理
                outputs = model(image)  # 前向推理得到输出列表
            self.metric.update(outputs[0], target)  # 用主输出（outputs[0]）与标签更新混淆矩阵
            pixAcc, mIoU = self.metric.get()  # 获取到目前为止累积的像素精度与平均 IoU
            logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(  # 逐样本打印当前累计指标
                i + 1, pixAcc * 100, mIoU * 100))  # 【注意】此处乘 100 转为百分比显示，而 train.py 的 validation() 未乘，两处口径不一致

            if self.args.save_pred:  # 若开启预测图保存（当前在 __main__ 中被强制开启）
                pred = torch.argmax(outputs[0], 1)  # 沿通道维取最大值索引，得到每像素的类别预测
                pred = pred.cpu().data.numpy()  # 搬回 CPU 并转为 numpy 数组

                predict = pred.squeeze(0)  # 去掉 batch 维，得到 (H, W) 的类别索引图
                mask = get_color_pallete(predict, self.args.dataset)  # 按数据集调色板将索引图着色为 PIL 彩色图
                mask.save(os.path.join(outdir, os.path.splitext(filename[0])[0] + '.png'))  # 以原文件名（去扩展名）保存为 PNG；【注意】outdir 是 __main__ 中的全局变量，类方法内直接引用，写法不规范（应作为参数传入）
        synchronize()  # 分布式屏障：等待所有进程完成评估，保证日志与预测图全部落盘


if __name__ == '__main__':  # 脚本直接执行时的入口（被 import 时不运行）
    args = parse_args()  # 复用 train.py 的参数解析
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1  # 从环境变量读进程总数（分布式启动器注入），未注入则视为单机单卡
    args.distributed = num_gpus > 1  # 进程数大于 1 即启用分布式评估
    if not args.no_cuda and torch.cuda.is_available():  # 未禁用 CUDA 且 GPU 可用时
        cudnn.benchmark = True  # 开启 cuDNN 自动调优，加速推理
        args.device = "cuda"  # 使用 GPU 评估
    else:  # 无 GPU 或禁用 CUDA 分支：改用 CPU 并关闭分布式
        args.distributed = False  # CPU 环境强制关闭分布式
        args.device = "cpu"  # 使用 CPU 评估
    if args.distributed:  # 初始化分布式评估环境
        torch.cuda.set_device(args.local_rank)  # 绑定当前进程使用的 GPU
        torch.distributed.init_process_group(backend="nccl", init_method="env://")  # 以 NCCL 后端、环境变量寻址方式初始化进程组
        synchronize()  # 等待全部进程完成初始化

    # TODO: optim code
    args.save_pred = True  # 【注意】强制开启预测图保存，上方原 TODO 注释表示此处待优化；如需关闭保存需修改此行
    if args.save_pred:  # 创建预测图输出目录
        outdir = '../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)  # 输出目录：../runs/pred_pic/模型_骨干_数据集
        if not os.path.exists(outdir):  # 目录不存在时
            os.makedirs(outdir)  # 递归创建目录

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),  # 初始化全局日志器：输出到 log_dir；eval() 方法中引用的 logger 即此全局变量
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')  # 日志文件名含模型信息；a+ 追加模式，与训练日志共用同一文件

    evaluator = Evaluator(args)  # 创建评估器实例（完成数据/模型等初始化）
    evaluator.eval()  # 执行评估
    torch.cuda.empty_cache()  # 评估结束后清空 CUDA 缓存显存
