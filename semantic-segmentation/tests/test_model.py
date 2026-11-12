# ============================================================================
# 【文件说明】test_model.py —— 模型"过拟合"冒烟测试脚本（非断言式单元测试）
# 测试思路：在仓库根目录的单张样例图（test_img.jpg / test_mask.png）上，用同一份数据
#   反复训练（过拟合）指定模型 100 个 epoch，全链路验证：前向推理 → 损失计算 →
#   反向传播 → 参数更新 → 预测可视化 → 指标统计（pixAcc / mIoU）；实现有误时
#   （形状不匹配、梯度为 None 等）训练循环会直接报错，loss 正常时应持续下降。
# 覆盖模型（--model）：fcn32s/fcn16s/fcn8s/fcn/psp/deeplabv3/danet/denseaspp/
#   bisenet/encnet/dunet/icnet/enet/ocnet；骨干（--backbone）：vgg16、resnet 系列等；
# 数据集（--dataset）：pascal_voc/pascal_aug/ade20k/citys/sbu（决定类别数与调色板）。
# 输入：单张 504x368 图片（batch=1，ImageNet 归一化）；输出：每 epoch 打印
#   loss/pixAcc/mIoU，并把彩色预测图保存到 runs/<模型名>/ 目录。
# 【注意】需在仓库根目录运行（图片为相对路径 test_img.jpg/test_mask.png），
#   且该脚本会真正执行训练（GPU 优先，无 GPU 回退 CPU）。
# ============================================================================
"""Model overfitting test"""  # 原有模块 docstring 保留：模型过拟合测试
import argparse  # 标准库：命令行参数解析
import time  # 标准库：计时（统计训练累计耗时）
import os  # 标准库：创建预测结果保存目录
import sys  # 标准库：把仓库根目录加入模块搜索路径

import torch  # PyTorch 主库：张量运算与自动求导
import torch.nn as nn  # 神经网络模块（此处用 BatchNorm2d 作为 norm_layer）
import torch.backends.cudnn as cudnn  # cuDNN 开关：benchmark 模式加速固定尺寸输入
import numpy as np  # NumPy：掩码转数组、指标计算

cur_path = os.path.abspath(os.path.dirname(__file__))  # tests/ 目录的绝对路径
root_path = os.path.split(cur_path)[0]  # tests 的上一级目录即仓库根目录
sys.path.append(root_path)  # 加入模块搜索路径以便导入 core 包

from torchvision import transforms  # 图像预处理：ToTensor、Normalize
from core.models.model_zoo import get_segmentation_model  # 模型工厂：按 model+dataset+backbone 创建分割网络
from core.utils.loss import MixSoftmaxCrossEntropyLoss, EncNetLoss, ICNetLoss  # 损失函数：主用 MixSoftmaxCrossEntropyLoss，EncNetLoss/ICNetLoss 为备选（见下方注释行）
from core.utils.lr_scheduler import LRScheduler  # 函数式 poly 学习率调度器
from core.utils.score import hist_info, compute_score  # 混淆直方图统计、pixAcc/mIoU 计算
from core.utils.visualize import get_color_pallete  # 类别预测图 → 按数据集调色板着色的 PIL 彩图
from PIL import Image  # PIL：读取样例图片与掩码


# 形参说明: 无参数，配置全部来自命令行
def parse_args():  # 解析命令行参数并选择计算设备
    parser = argparse.ArgumentParser(description='Semantic Segmentation Overfitting Test')  # 创建解析器：语义分割过拟合测试
    # model
    parser.add_argument('--model', type=str, default='fcn32s',  # 待测模型名（默认 fcn32s）
                        choices=['fcn32s', 'fcn16s', 'fcn8s', 'fcn', 'psp',   # 可选模型（前半段）
                        'deeplabv3', 'danet', 'denseaspp', 'bisenet', 'encnet',   # 可选模型（中段）
                        'dunet', 'icnet', 'enet', 'ocnet'],  # 可选模型（后段）
                        help='model name (default: fcn32s)')  # 帮助文本
    parser.add_argument('--backbone', type=str, default='vgg16',  # 骨干网络名（默认 vgg16）
                        choices=['vgg16', 'resnet18', 'resnet50', 'resnet101',   # 可选骨干（前半段）
                        'resnet152', 'densenet121', '161', '169', '201'],  # 【注意】'161'/'169'/'201' 疑为 'densenet161/169/201' 的漏写，传入模型会因无法识别骨干而报错
                        help='backbone name (default: vgg16)')  # 帮助文本
    parser.add_argument('--dataset', type=str, default='pascal_voc',  # 数据集名（决定类别数与调色板，默认 pascal_voc）
                        choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys',   # 可选数据集（前半段）
                        'sbu'],  # 可选数据集（后段）：SBU Shadow
                        help='dataset name (default: pascal_voc)')  # 帮助文本
    parser.add_argument('--epochs', type=int, default=100, metavar='N',  # 过拟合训练轮数（默认 100）
                        help='number of epochs to train (default: 100)')  # 帮助文本
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',  # 初始学习率（默认 1e-3）
                        help='learning rate (default: 1e-3)')  # 帮助文本
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',  # 动量参数（默认 0.9）；【注意】本脚本用 Adam 优化器，该参数实际未传入优化器，属于无效参数
                        help='momentum (default: 0.9)')  # 帮助文本
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',  # 权重衰减（代码默认 1e-4）
                        help='w-decay (default: 5e-4)')  # 【注意】帮助文本写的默认值是 5e-4，与代码实际 1e-4 不一致
    args = parser.parse_args()  # 解析命令行参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 自动选择设备：有 GPU 用 cuda:0，否则回退 CPU
    cudnn.benchmark = True  # 开启 cuDNN 自动调优：输入尺寸固定时可显著加速（尺寸多变时反而变慢）
    args.device = device  # 把设备对象挂到 args 上，便于后续访问
    print(args)  # 打印完整配置，便于复现实验
    return args  # 返回参数命名空间


class VOCSegmentation(object):  # 迷你"数据集"：仅加载单张样例图与掩码（模拟 VOC 单样本）
    # 形参说明: 无参数；图片路径写死为相对路径，必须在仓库根目录下运行
    def __init__(self):  # 构造函数：读取并缩放样例图/掩码
        super(VOCSegmentation, self).__init__()  # 调用父类 object 构造（习惯性写法）
        self.img = Image.open('test_img.jpg').convert('RGB')  # 读取样例图并统一转为 RGB 三通道
        self.mask = Image.open('test_mask.png')  # 读取对应的分割掩码（P 模式调色板图）

        self.img = self.img.resize((504, 368), Image.BILINEAR)  # 原图用双线性插值缩放到 504(宽)x368(高)
        self.mask = self.mask.resize((504, 368), Image.NEAREST)  # 掩码用最近邻缩放，避免类别标签被插值混叠

    def get(self):  # 对外接口：返回预处理完成的 (输入张量, 目标张量)
        img, mask = self._img_transform(self.img), self._mask_transform(self.mask)  # 分别执行图像/掩码变换
        return img, mask  # 返回 (img, mask) 二元组

    def _img_transform(self, img):  # 图像预处理：归一化并补充 batch 维
        input_transform = transforms.Compose([  # 顺序组合多个变换
            transforms.ToTensor(),  # PIL 图 → [0,1] 浮点张量，且 HWC → CHW
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])  # 按 ImageNet 三通道均值/标准差标准化
        img = input_transform(img)  # 应用变换，得到 3x368x504 张量
        img = img.unsqueeze(0)  # 增加 batch 维 → 1x3x368x504

        # For adaptive pooling
        # img = torch.cat([img, img], dim=0)
        return img  # 返回带 batch 维的输入张量

    def _mask_transform(self, mask):  # 掩码预处理：转类别索引张量并补充 batch 维
        target = np.array(mask).astype('int32')  # PIL 掩码 → int32 NumPy 数组
        target[target == 255] = -1  # VOC 忽略标签 255 → -1（与损失函数 ignore_label=-1 对应）
        target = torch.from_numpy(target).long()  # NumPy → int64 张量（CrossEntropy 的目标要求 long 型）
        target = target.unsqueeze(0)  # 增加 batch 维 → 1x368x504

        # For adaptive pooling
        # target = torch.cat([target, target], dim=0)
        return target  # 返回目标张量


class Trainer(object):  # 过拟合训练器：组装数据、模型、损失、优化器、调度器并执行训练
    # 形参说明: args parse_args() 返回的参数命名空间
    def __init__(self, args):  # 构造：准备单图数据与全套训练组件
        self.args = args  # 保存参数

        self.img, self.target = VOCSegmentation().get()  # 获取单张样例的输入与目标

        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,  # 按名字从模型工厂创建分割网络（不加载预训练权重）
                                            aux=False, norm_layer=nn.BatchNorm2d).to(args.device)  # 关闭辅助损失分支，BN 作为归一化层；模型移至所选设备

        self.criterion = MixSoftmaxCrossEntropyLoss(False, 0., ignore_label=-1).to(args.device)  # 混合 softmax 交叉熵损失：不开 aux、权重为 0、忽略 -1 标签；移至设备

        # for EncNet
        # self.criterion = EncNetLoss(nclass=21, ignore_label=-1).to(args.device)
        # for ICNet
        # self.criterion = ICNetLoss(nclass=21, ignore_index=-1).to(args.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),  # Adam 优化器（【注意】命令行 --momentum 未被使用）
                                          lr=args.lr,  # 学习率
                                          weight_decay=args.weight_decay)  # 权重衰减
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr, nepochs=args.epochs,  # poly 学习率调度器
                                        iters_per_epoch=1, power=0.9)  # 每个 epoch 仅 1 次迭代；poly 幂次 0.9

    def train(self):  # 在同一张图上反复训练（过拟合）并逐轮输出指标
        self.model.train()  # 切换到训练模式（BN 使用批统计并更新 running 值，Dropout 生效）
        start_time = time.time()  # 记录训练起始时间
        for epoch in range(self.args.epochs):  # 逐 epoch 训练
            self.lr_scheduler(self.optimizer, epoch)  # 按当前 epoch 计算学习率并写入优化器参数组
            cur_lr = self.lr_scheduler.learning_rate  # 读回刚计算出的当前学习率
            # self.lr_scheduler(self.optimizer, epoch)
            for param_group in self.optimizer.param_groups:  # 遍历优化器的所有参数组
                param_group['lr'] = cur_lr  # 【注意】冗余写法：上面对调度器的调用已把 lr 写入参数组，这里又手动覆盖一遍（本脚本只有一个参数组，无实际影响）

            images = self.img.to(self.args.device)  # 输入张量移至设备
            targets = self.target.to(self.args.device)  # 目标张量移至设备

            outputs = self.model(images)  # 前向推理，得到 logits 列表 [主输出, (可选辅助输出)]
            loss = self.criterion(outputs, targets)  # 计算损失，返回含 'loss' 键的字典

            self.optimizer.zero_grad()  # 清空上一步残留的梯度
            loss['loss'].backward()  # 对总损失反向传播
            self.optimizer.step()  # 优化器更新模型参数

            pred = torch.argmax(outputs[0], 1).cpu().data.numpy()  # 取主输出沿通道维 argmax 得类别图（1xHxW）；【注意】.data 取值已不推荐，建议改用 .detach()
            mask = get_color_pallete(pred.squeeze(0), self.args.dataset)  # 去掉 batch 维后按数据集调色板转为彩色 PIL 图
            save_pred(self.args, epoch, mask)  # 保存本 epoch 的彩色预测图
            hist, labeled, correct = hist_info(pred, targets.cpu().numpy(), 21)  # 统计混淆直方图/有效像素/正确像素；【注意】类别数硬编码 21（仅适配 VOC），选 ade20k/citys 等数据集时指标计算不正确
            _, mIoU, _, pixAcc = compute_score(hist, correct, labeled)  # 由直方图计算像素精度 pixAcc 与平均 IoU

            print('Epoch: [%2d/%2d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f || pixAcc: %.3f || mIoU: %.3f' % (  # 打印训练日志（格式化字符串）
                epoch, self.args.epochs, time.time() - start_time, cur_lr, loss['loss'].item(), pixAcc, mIoU))  # 各字段：当前轮/总轮数、累计耗时、当前 lr、损失、像素精度、平均 IoU


# 形参说明: args 参数命名空间; epoch 当前轮次（从 0 计）; mask 已着色的预测 PIL 图
def save_pred(args, epoch, mask):  # 将某轮的预测图保存到 runs/<模型名>/ 目录
    directory = "runs/%s/" % (args.model)  # 保存目录：runs/<模型名>/（相对当前工作目录）
    if not os.path.exists(directory):  # 目录不存在时
        os.makedirs(directory)  # 递归创建目录
    filename = directory + '{}_epoch_{}.png'.format(args.model, epoch + 1)  # 文件名：<模型>_epoch_<从 1 计的轮次>.png
    mask.save(filename)  # 将彩色预测图保存为 PNG


if __name__ == '__main__':  # 作为脚本直接运行时的主入口
    args = parse_args()  # 解析参数
    trainer = Trainer(args)  # 构造过拟合训练器
    print('Test model: ', args.model)  # 打印被测模型名
    trainer.train()  # 开始过拟合训练
