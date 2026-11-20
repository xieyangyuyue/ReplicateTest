# ============================================================================
# 【文件说明】score.py —— 语义分割评价指标：核心类 SegmentationMetric 在验证过程中
#             逐 batch 累计一个 (nclass, nclass) 混淆矩阵（GPU 上完成），
#             最终给出 pixAcc（像素准确率）、mIoU（平均交并比）、mPA（平均像素准确率）、
#             FWIoU（频率加权 IoU）、mF1（平均 F1）与逐类 IoU；支持多卡 all_reduce。
#             另附 PyTorch 版批量函数
#             batch_pix_accuracy / batch_intersection_union、numpy 版单图函数
#             pixelAccuracy / intersectionAndUnion、混淆矩阵构建 hist_info 与
#             指标计算 compute_score。
# 【核心思想】指标必须在"整个数据集的累计口径"下计算（而非逐 batch 求平均再平均，
#             否则结果有偏）。实现技巧：验证过程中只在 GPU 上累计一个 (nclass, nclass)
#             混淆矩阵（行=真实类别，列=预测类别），每个 batch 用一次 torch.bincount 完成
#             统计，不做任何 .cpu()/.item() 同步——这正是修复"验证时双卡利用率低"的关键。
#             所有指标（pixAcc / mIoU / mPA / FWIoU / mF1）只在需要时由混淆矩阵一次性推导：
#             TP=对角线，真实面积=行和，预测面积=列和，并集=真实+预测-TP（容斥原理），
#             pixAcc = ΣTP / Σ所有像素，IoU_c = TP_c / 并集_c，mIoU = 出现类别 IoU 的均值。
# ============================================================================
"""Evaluation Metrics for Semantic Segmentation"""  # 原有 docstring：语义分割评价指标
import torch  # 导入 PyTorch 主库
import numpy as np  # 导入 numpy（下方单图版指标与混淆矩阵计算使用）

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union',  # 对外导出符号白名单（首行，续于下一行）
           'pixelAccuracy', 'intersectionAndUnion', 'hist_info', 'compute_score']  # 对外导出符号白名单


class SegmentationMetric(object):  # 分割指标累计器：跨 batch 累计混淆统计后统一计算 pixAcc 与 mIoU
    # 类 docstring：计算像素准确率 pixAcc 与平均交并比 mIoU 两种指标
    """Computes pixAcc and mIoU metric scores
    """  # 原有 docstring：计算像素准确率与平均交并比两种指标

    # 形参说明: nclass 类别数（决定各累计向量的维度）
    def __init__(self, nclass, device=None):
        super(SegmentationMetric, self).__init__()  # 初始化父类 object
        self.nclass = nclass  # 保存类别数
        # 指标累计张量所在设备：训练脚本传入 cuda 后，整个验证过程都在 GPU 上累计，避免逐 batch 同步
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.reset()  # 调用 reset 将所有累计器清零

    def update(self, preds, labels):  # 用一个 batch（或多张图）的预测与标签更新累计器
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : Tensor or list of Tensor
            Ground-truth label maps (ignore label < 0).
        preds : Tensor or list of Tensor
            Logits (N, C, H, W) or class-index maps (N, H, W).
        """
        if isinstance(preds, torch.Tensor):  # 输入是单个张量：整个 batch 一次统计
            self._update_one(preds, labels)  # 直接处理该 batch
        elif isinstance(preds, (list, tuple)):  # 输入是列表/元组：逐张图分别统计
            for (pred, label) in zip(preds, labels):  # 成对遍历预测与标签
                self._update_one(pred, label)  # 逐张累计结果
        else:  # 不支持的类型
            raise TypeError("preds should be a torch.Tensor or a list/tuple of tensors")  # 明确报错，避免静默漏统计

    def _update_one(self, pred, label):  # 内部工作函数：在 GPU 上累计 (nclass, nclass) 混淆矩阵
        # 关键优化：原来每个 batch 要用 3 次 torch.histc(...).cpu() 和多次 .item()，
        # 这些操作都会强制 GPU 同步，导致验证时 GPU 等 CPU、双卡利用率低。
        # 现在直接在整个 batch 上做一次 bincount 得到混淆矩阵，累计过程不触发同步。
        if pred.dim() == 4:  # 输入是 logits (N, C, H, W)
            pred = torch.argmax(pred, dim=1)  # 取类别维 argmax 得到 (N, H, W) 类别图（在 GPU 上完成）
        pred = pred.long()  # 统一为 int64，bincount 需要整型索引
        label = label.long()  # 标签统一为 int64

        if self.confusion.device != pred.device:  # 指标对象设备与输入不一致（例如初始化在 CPU、数据在 GPU）
            self.confusion = self.confusion.to(pred.device)  # 把累计矩阵搬到输入设备
            self.device = pred.device  # 记录当前设备，reset 时保持一致

        valid = (label >= 0) & (label < self.nclass) & (pred >= 0) & (pred < self.nclass)  # 有效像素：忽略 -1 与其他越界值
        pred = pred[valid]  # 只保留有效像素的预测
        label = label[valid]  # 只保留有效像素的标签
        if label.numel() == 0:  # 整个 batch 全部是忽略像素（边界情况）
            return  # 直接跳过，避免 bincount 空输入
        index = label * self.nclass + pred  # 展平为一维混淆矩阵索引：row=真实类别, col=预测类别
        try:  # 优先在 GPU 上 bincount（现代 PyTorch 支持 CUDA）
            hist = torch.bincount(index, minlength=self.nclass * self.nclass)  # 本 batch 的混淆矩阵计数
        except RuntimeError:  # 极旧版本/特殊后端不支持 CUDA bincount 时回退 CPU
            hist = torch.bincount(index.cpu(), minlength=self.nclass * self.nclass).to(index.device)  # CPU 回退
        self.confusion += hist.reshape(self.nclass, self.nclass).to(self.confusion.dtype)  # 累计到全局混淆矩阵

    def _stats(self):  # 从混淆矩阵一次性推导所有统计量（TP / 真实面积 / 预测面积 / 并集）
        tp = self.confusion.diag()  # 逐类 TP：对角线元素
        gt = self.confusion.sum(dim=1)  # 逐类真实像素数（TP + FN），即行和
        pred = self.confusion.sum(dim=0)  # 逐类预测像素数（TP + FP），即列和
        union = gt + pred - tp  # 逐类并集：TP + FP + FN
        return tp, gt, pred, union  # 返回四个 (nclass,) 张量

    def get(self):  # 获取累计口径下的最终指标
        # docstring：获取当前累计的评估结果（返回 pixAcc 与 mIoU）
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """  # 原有 docstring 原样保留：返回 (pixAcc, mIoU)
        return self.pixel_accuracy(), self.mean_intersection_over_union()  # 保持旧接口不变，供 eval.py / train.py 直接复用

    def pixel_accuracy(self):  # 像素准确率（全局口径）
        total = self.confusion.sum()  # 有效像素总数（混淆矩阵所有元素之和）
        if total <= 0:  # 没有任何有效像素
            return 0.0  # 返回 0，避免除零
        return (self.confusion.diag().sum() / total).item()  # 对角线之和 / 总像素 = 正确率

    def class_pixel_accuracy(self):  # 逐类别像素准确率（即召回率 recall = TP / (TP + FN)）
        tp, gt, _, _ = self._stats()  # TP 与真实像素数
        return tp / (gt + 2.220446049250313e-16)  # 逐类正确数 / 该类真实像素数；eps 防除零

    def mean_pixel_accuracy(self):  # 平均像素准确率 mPA：只对真实标注中出现过的类别求平均
        _, gt, _, _ = self._stats()  # 逐类真实像素数
        class_acc = self.class_pixel_accuracy()  # 逐类召回率
        valid = gt > 0  # 忽略验证集中从未出现的类别，避免其 0 值拉低均值
        if valid.any():  # 至少存在一个有效类别
            return class_acc[valid].mean().item()  # 有效类别的平均召回率
        return 0.0  # 空验证集或无有效类别时返回 0

    def intersection_over_union(self):  # 逐类别 IoU
        tp, _, _, union = self._stats()  # TP 与并集
        return tp / (union + 2.220446049250313e-16)  # TP / (TP + FP + FN)

    def mean_intersection_over_union(self):  # 平均交并比 mIoU：只对验证集中出现过的类别求平均
        _, _, _, union = self._stats()  # 逐类并集
        iou = self.intersection_over_union()  # 逐类 IoU
        valid = union > 0  # 并集为 0 说明该类在预测和标签中都没出现
        if valid.any():  # 至少存在一个有效类别
            return iou[valid].mean().item()  # 有效类别的平均 IoU
        return 0.0  # 空验证集时返回 0

    def frequency_weighted_iou(self):  # 频率加权交并比 FWIoU：按各类真实像素频率对 IoU 加权
        _, gt, _, _ = self._stats()  # 逐类真实像素数
        total = gt.sum()  # 全部有效像素数
        if total <= 0:  # 没有有效像素
            return 0.0  # 返回 0，避免除零
        freq = gt / total  # 各类别在真实标签中的出现频率
        iou = self.intersection_over_union()  # 逐类 IoU
        valid = freq > 0  # 只统计出现过的类别
        return (freq[valid] * iou[valid]).sum().item()  # 频率加权求和

    def class_f1_score(self):  # 逐类别 F1 = 2TP / (2TP + FP + FN)
        tp, gt, pred, _ = self._stats()  # TP / 真实面积 / 预测面积
        return 2 * tp / (pred + gt + 2.220446049250313e-16)  # 分母 (TP+FP) + (TP+FN) = 2TP+FP+FN，加 eps 防除零

    def mean_f1_score(self):  # 平均 F1：只对出现过的类别求平均
        _, gt, pred, _ = self._stats()  # 真实面积与预测面积
        f1 = self.class_f1_score()  # 逐类 F1
        valid = (pred + gt) > 0  # 预测与标签都为空的类别不参与平均
        if valid.any():  # 至少存在一个有效类别
            return f1[valid].mean().item()  # 有效类别的平均 F1
        return 0.0  # 空验证集时返回 0

    def get_scores(self):  # 一次性返回全部指标，便于训练日志与模型选优统一读取
        # 只在验证结束时调用一次，因此这里的 .item()/.tolist() 同步开销可以忽略
        return {
            'pixel_accuracy': float(self.pixel_accuracy()),  # 像素准确率 pixAcc
            'mean_pixel_accuracy': self.mean_pixel_accuracy(),  # 平均像素准确率 mPA
            'mean_iou': self.mean_intersection_over_union(),  # 平均交并比 mIoU
            'frequency_weighted_iou': self.frequency_weighted_iou(),  # 频率加权交并比 FWIoU
            'mean_f1': self.mean_f1_score(),  # 平均 F1
            'class_iou': self.intersection_over_union().tolist(),  # 逐类 IoU（保留原始数值，便于离线分析）
        }

    def all_reduce(self):  # 分布式验证：把各 rank 的累计统计量求和，得到全局指标
        """All-reduce the accumulated statistics across all processes."""
        import torch.distributed as dist  # 延迟导入，单卡/非分布式环境无需关心
        if not dist.is_available() or not dist.is_initialized():  # 分布式未初始化
            return self  # 单进程直接返回
        if dist.get_world_size() <= 1:  # 只有 1 个进程
            return self  # 无需通信
        if dist.get_backend() == 'nccl':  # NCCL 只支持 CUDA 张量
            comm_device = torch.device('cuda', torch.cuda.current_device())  # 使用当前进程绑定的 GPU
        else:  # gloo 等后端
            comm_device = torch.device('cpu')  # 使用 CPU 张量通信
        # 只通信一个 (nclass, nclass) 混淆矩阵，替代原来 6 次小 all_reduce
        confusion = self.confusion.to(comm_device)  # 搬到通信设备（NCCL 支持 float64）
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)  # 各 rank 求和 = 全局混淆矩阵
        self.confusion = confusion.to(self.device)  # 写回指标所在设备，后续 get() 无需再搬运
        return self  # 返回自身便于链式调用

    def reset(self):  # 重置所有累计器到初始状态
        """Resets the internal evaluation result to initial state."""  # 原有 docstring：重置内部评估结果
        # 混淆矩阵：行 = 真实类别，列 = 预测类别；float64 防止大验证集累加精度损失
        self.confusion = torch.zeros(self.nclass, self.nclass, dtype=torch.float64,  # (nclass, nclass) 混淆矩阵
                                     device=self.device)  # 初始化在指标设备上（GPU 上累计，避免逐 batch 同步）


# pytorch version  # 原注释保留：以下为 PyTorch 版本实现
def batch_pix_accuracy(output, target):  # 计算一个 batch 的 (正确像素数, 有效标记像素数)
    """PixAcc"""  # 原有 docstring：像素准确率
    # inputs are numpy array, output 4D, target 3D  # 原注释保留：输入说明（output 为 4D logits，target 为 3D 标签）
    predict = torch.argmax(output, 1) + 1  # 类别维 argmax 得预测标签后整体 +1（直接对浮点 logits 取 argmax，避免先 .long() 截断造成并列）
    target = target.long() + 1  # 标签 +1：使忽略值 -1 变为 0，从而可用 target > 0 识别有效像素

    pixel_labeled = torch.sum(target > 0).item()  # 有效（被标注）像素总数
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()  # 预测正确且落在有效区域内的像素数
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"  # 断言：正确数不会超过标记数
    return pixel_correct, pixel_labeled  # 返回 (正确像素数, 有效标记数)


def batch_intersection_union(output, target, nclass, return_counts=False):  # 计算各类别的交集与并集像素面积
    """mIoU"""  # 原有 docstring：平均交并比
    # inputs are numpy array, output 4D, target 3D  # 原注释保留：输入说明
    # return_counts=True 时额外返回逐类预测面积 area_pred 与逐类标签面积 area_lab，
    # 用于在 SegmentationMetric 中计算 mPA / F1 / FWIoU；默认 False 保持旧接口兼容。
    mini = 1  # 直方图统计下界：类别 +1 后的最小有效类别值 1
    maxi = nclass  # 直方图统计上界：类别 +1 后的最大类别值 nclass
    nbins = nclass  # 直方图 bin 数：每个类别一个 bin
    predict = torch.argmax(output, 1) + 1  # 类别维 argmax 后 +1: (N,H,W)，有效取值范围 [1, nclass]
    target = target.float() + 1  # 标签 +1 并转 float（histc 要求 float 输入）: (N,H,W)

    predict = predict.float() * (target > 0).float()  # 无效位置（标签 +1 后 <= 0，即原忽略标签）的预测置 0，不参与直方图统计
    intersection = predict * (predict == target).float()  # 仅预测正确的像素保留其类别值、错误位置置 0：作为交集面积的统计源
    # areas of intersection and union  # 原注释保留：交集与并集面积
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.  # 原注释保留：说明 0 号元素与 np.bincount 的差异及边界处理
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)  # 交集直方图：各类别的预测正确像素数 (nclass,)；.cpu() 因旧版 histc 仅支持 CPU 张量（存在设备往返开销）
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)  # 预测直方图：各类别的预测像素数 (nclass,)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)  # 标签直方图：各类别的真实像素数 (nclass,)
    area_union = area_pred + area_lab - area_inter  # 容斥原理：并集 = 预测面积 + 标签面积 - 交集面积
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"  # 断言：任一类别的交集不超过并集
    if return_counts:  # 额外返回逐类预测/标签面积
        return area_inter.float(), area_union.float(), area_pred.float(), area_lab.float()  # (交集, 并集, 预测面积, 标签面积)
    return area_inter.float(), area_union.float()  # 返回各类别 (交集面积, 并集面积)，均转 float


def pixelAccuracy(imPred, imLab):  # numpy 版单图像素准确率（返回准确率、正确数、标记数三元组）
    # docstring：单图像素级准确率计算（含多图聚合的用法示例）
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    To compute over many images do:
    for i = range(Nimages):
         (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
            pixelAccuracy(imPred[i], imLab[i])
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """  # 原有 docstring 原样保留：说明单图用法与多图聚合方式
    # Remove classes from unlabeled pixels in gt image.  # 原注释保留：剔除真值图中未标注像素的类别
    # We should not penalize detections in unlabeled portions of the image.  # 原注释保留：不应惩罚未标注区域的检测
    pixel_labeled = np.sum(imLab >= 0)  # 有效标记像素数（标签 >= 0 视为有效）
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))  # 有效区域内预测正确的像素数
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled  # 像素准确率 = 正确 / 标记；【注意】无 eps 保护，全图无标注时会除零（docstring 中的多图聚合公式反而有 np.spacing(1)）
    return (pixel_accuracy, pixel_correct, pixel_labeled)  # 返回 (准确率, 正确像素数, 标记像素数)


def intersectionAndUnion(imPred, imLab, numClass):  # numpy 版单图的各类别交集/并集面积
    # docstring：由单图预测与标签计算各类别交集/并集面积（含多图聚合示例）
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """  # 原有 docstring 原样保留：说明单图用法与多图 IoU 聚合方式
    # Remove classes from unlabeled pixels in gt image.  # 原注释保留：剔除真值图中未标注像素的类别
    # We should not penalize detections in unlabeled portions of the image.  # 原注释保留：不应惩罚未标注区域的检测
    imPred = imPred * (imLab >= 0)  # 未标注区域的预测置 0，使其不参与直方图统计

    # Compute area intersection:  # 原注释保留：计算交集面积
    intersection = imPred * (imPred == imLab)  # 预测正确的像素保留类别值，错误位置置 0
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))  # 交集直方图：各类别的预测正确像素数 (numClass,)

    # Compute area union:  # 原注释保留：计算并集面积
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))  # 预测直方图：各类别的预测像素数
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))  # 标签直方图：各类别的真实像素数
    area_union = area_pred + area_lab - area_intersection  # 容斥原理：并集 = 预测面积 + 标签面积 - 交集面积
    return (area_intersection, area_union)  # 返回各类别 (交集面积, 并集面积)


def hist_info(pred, label, num_cls):  # 由单图预测与标签构建混淆矩阵（行=真实类别，列=预测类别）
    assert pred.shape == label.shape  # 断言：预测与标签形状必须一致
    k = (label >= 0) & (label < num_cls)  # 有效像素掩码：剔除负值（忽略标签）与越界标签
    labeled = np.sum(k)  # 有效像素数
    correct = np.sum((pred[k] == label[k]))  # 有效像素中预测正确的数量

    return np.bincount(num_cls * label[k].astype(int) + pred[k], minlength=num_cls ** 2).reshape(num_cls,  # 混淆矩阵的一维索引 = num_cls * 真实类别 + 预测类别，bincount 计数后 reshape 成 (num_cls, num_cls)
                                                                                                 num_cls), labeled, correct  # 返回 (混淆矩阵, 有效像素数, 正确像素数)


def compute_score(hist, correct, labeled):  # 由混淆矩阵与像素统计计算各项指标
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # 各类 IoU = 对角线(正确数) / (行和(真实面积) + 列和(预测面积) - 对角线)，即交并比
    mean_IU = np.nanmean(iu)  # mIoU：忽略 nan（某类从未出现时分母为 0）后取平均
    mean_IU_no_back = np.nanmean(iu[1:])  # 去除背景类（索引 0）后的 mIoU
    freq = hist.sum(1) / hist.sum()  # 各类别真实像素的出现频率
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()  # 频率加权 IoU（FWIoU）：只统计出现过的类别并按频率加权求和
    mean_pixel_acc = correct / labeled  # 像素准确率 = 正确像素数 / 有效像素数

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc  # 返回 (各类 IoU, mIoU, 去背景 mIoU, 像素准确率)
