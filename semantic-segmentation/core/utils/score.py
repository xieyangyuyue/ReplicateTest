# ============================================================================
# 【文件说明】score.py —— 语义分割评价指标：核心类 SegmentationMetric 在验证过程中
#             逐 batch 累计"像素正确数/标记数"与"各类别交集/并集面积"，最终给出
#             pixAcc（像素准确率）与 mIoU（平均交并比）；另附 PyTorch 版批量函数
#             batch_pix_accuracy / batch_intersection_union、numpy 版单图函数
#             pixelAccuracy / intersectionAndUnion、混淆矩阵构建 hist_info 与
#             指标计算 compute_score。
# 【核心思想】指标必须在"整个数据集的累计口径"下计算（而非逐 batch 求平均再平均，
#             否则结果有偏）。实现技巧：把预测与标签整体 +1，使忽略标签 -1 变为 0，
#             用 target > 0 作为有效像素掩码；各类面积用 torch.histc 直方图统计——
#             交集面积 = "预测正确的像素"按类别的直方图，并集 = 预测面积 + 标签面积
#             - 交集面积（容斥原理）。最终：pixAcc = Σ正确像素 / Σ有效像素，
#             IoU_c = 交集_c / 并集_c，mIoU = 各类 IoU 的均值。
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
    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()  # 初始化父类 object
        self.nclass = nclass  # 保存类别数
        self.reset()  # 调用 reset 将所有累计器清零

    def update(self, preds, labels):  # 用一个 batch（或多张图）的预测与标签更新累计器
        # docstring：用新一批预测与标签更新内部累计的评估结果
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """  # 原有 docstring 原样保留：说明输入可为数组或数组列表

        def evaluate_worker(self, pred, label):  # 内部工作函数：处理一对 (预测, 标签) 并完成累计（self 由外部闭包显式传入）
            correct, labeled = batch_pix_accuracy(pred, label)  # 本对数据的 (正确像素数, 有效标记像素数)
            inter, union = batch_intersection_union(pred, label, self.nclass)  # 各类别交集/并集面积向量 (nclass,)

            self.total_correct += correct  # 累加正确像素数
            self.total_label += labeled  # 累加有效标记像素数
            if self.total_inter.device != inter.device:  # 若累计向量与当前 batch 所在设备不一致
                self.total_inter = self.total_inter.to(inter.device)  # 把累计交集向量搬到当前设备
                self.total_union = self.total_union.to(union.device)  # 把累计并集向量搬到当前设备
            self.total_inter += inter  # 累加各类别交集面积
            self.total_union += union  # 累加各类别并集面积

        if isinstance(preds, torch.Tensor):  # 输入是单个张量：整个 batch 一次统计
            evaluate_worker(self, preds, labels)  # 直接处理该 batch
        elif isinstance(preds, (list, tuple)):  # 输入是列表/元组：逐张图分别统计
            for (pred, label) in zip(preds, labels):  # 成对遍历预测与标签
                evaluate_worker(self, pred, label)  # 逐张累计结果

    def get(self):  # 获取累计口径下的最终指标
        # docstring：获取当前累计的评估结果（返回 pixAcc 与 mIoU）
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """  # 原有 docstring 原样保留：返回 (pixAcc, mIoU)
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)  # pixAcc = 累计正确像素 / 累计有效像素；2.220446049250313e-16 即双精度机器精度 np.spacing(1)，用于防除零（原英文注释保留）
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)  # 各类 IoU = 累计交集 / 累计并集（同样加 eps 防除零）
        mIoU = IoU.mean().item()  # mIoU = 各类 IoU 的均值（.item() 转为 Python 标量）
        return pixAcc, mIoU  # 返回 (像素准确率, 平均交并比)

    def reset(self):  # 重置所有累计器到初始状态
        """Resets the internal evaluation result to initial state."""  # 原有 docstring：重置内部评估结果
        self.total_inter = torch.zeros(self.nclass)  # 各类别交集面积累计向量清零 (nclass,)
        self.total_union = torch.zeros(self.nclass)  # 各类别并集面积累计向量清零 (nclass,)
        self.total_correct = 0  # 正确像素数清零
        self.total_label = 0  # 有效标记像素数清零


# pytorch version  # 原注释保留：以下为 PyTorch 版本实现
def batch_pix_accuracy(output, target):  # 计算一个 batch 的 (正确像素数, 有效标记像素数)
    """PixAcc"""  # 原有 docstring：像素准确率
    # inputs are numpy array, output 4D, target 3D  # 原注释保留：输入说明（output 为 4D logits，target 为 3D 标签）
    predict = torch.argmax(output.long(), 1) + 1  # 类别维 argmax 得预测标签后整体 +1；【注意】可疑写法：先 .long() 截断 logits 小数再 argmax，可能因截断后并列而改变 argmax 结果（如 0.9 与 0.95 均截为 0），且与 batch_intersection_union 中不带 .long() 的写法不一致
    target = target.long() + 1  # 标签 +1：使忽略值 -1 变为 0，从而可用 target > 0 识别有效像素

    pixel_labeled = torch.sum(target > 0).item()  # 有效（被标注）像素总数
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()  # 预测正确且落在有效区域内的像素数
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"  # 断言：正确数不会超过标记数
    return pixel_correct, pixel_labeled  # 返回 (正确像素数, 有效标记数)


def batch_intersection_union(output, target, nclass):  # 计算各类别的交集与并集像素面积
    """mIoU"""  # 原有 docstring：平均交并比
    # inputs are numpy array, output 4D, target 3D  # 原注释保留：输入说明
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
