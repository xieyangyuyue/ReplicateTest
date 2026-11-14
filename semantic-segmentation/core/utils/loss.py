# ============================================================================
# 【文件说明】loss.py —— 语义分割的自定义损失函数集合：通用带辅助输出的交叉熵
#             MixSoftmaxCrossEntropyLoss、EncNet 专用 EncNetLoss（交叉熵 + SE 损失）、
#             ICNet 专用 ICNetLoss（多分辨率标签交叉熵）、OHEM 难例挖掘交叉熵
#             OhemCrossEntropy2d / MixSoftmaxCrossEntropyOHEMLoss，
#             并由工厂函数 get_segmentation_loss 按模型名统一创建对应损失。
# 【核心思想】
#   1) 辅助损失（深监督）：主输出与各辅助输出共享同一标签分别计算交叉熵后加权求和
#      （loss = loss_main + aux_weight * Σ loss_aux），让中间层获得更直接的梯度信号；
#   2) OHEM（Online Hard Example Mining，在线难例挖掘）：对每像素取 softmax 后其
#      真实类别的概率 p，p 越小说明越难分；仅保留 p <= thresh（默认 0.7）的像素计算
#      损失，且保证至少保留 min_kept 个最难像素——把梯度集中在难样本上，缓解
#      类别不均衡下"简单像素淹没梯度"的问题；
#   3) SE 损失（EncNet）：把每张图真实出现的类别编码为 0/1 向量，用 BCE 监督通道
#      注意力（SEBlock）的预测，迫使网络学会"图里有哪些类"的全局语义。
# ============================================================================
"""Custom losses."""  # 原有 docstring：自定义损失函数
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块（BCELoss、CrossEntropyLoss 等）
import torch.nn.functional as F  # 导入函数式接口（softmax、interpolate 等）

from torch.autograd import Variable  # 【注意】Variable 是 PyTorch 0.3 时代的废弃 API，现代版本直接使用 Tensor 即可

__all__ = ['MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss',  # 对外导出符号白名单（首行，续于下一行）
           'EncNetLoss', 'ICNetLoss', 'get_segmentation_loss']  # 对外导出符号白名单（OhemCrossEntropy2d 未导出，仅作基类使用）


# TODO: optim function  # 原注释保留：待优化
class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):  # 混合交叉熵损失：继承 nn.CrossEntropyLoss，扩展支持"主输出 + 多个辅助输出"的联合监督
    # 形参说明: aux 是否启用辅助损失(默认True); aux_weight 辅助损失权重(默认0.2); ignore_index 被忽略的标签值(默认-1)
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)  # 初始化父类交叉熵（内部等价于 log_softmax + NLLLoss，输入须为未归一化的 logits）
        self.aux = aux  # 是否使用辅助分支损失
        self.aux_weight = aux_weight  # 辅助损失加权系数

    def _aux_forward(self, *inputs, **kwargs):  # 辅助分支前向：inputs = (pred_main, pred_aux1, ..., target)
        *preds, target = tuple(inputs)  # 星号解包：前面的都归 preds（各预测输出），最后一个为 target 标签 (N,H,W)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)  # 主输出的交叉熵损失（调用父类 forward）
        for i in range(1, len(preds)):  # 遍历其余辅助输出
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)  # 第 i 个辅助输出对同一标签的交叉熵
            loss += self.aux_weight * aux_loss  # 加权累加: loss = loss_main + aux_weight * Σ loss_aux
        return loss  # 返回标量总损失

    def forward(self, *inputs, **kwargs):  # 统一入口：inputs = (pred 或 pred, ..., target)
        preds, target = tuple(inputs)  # 拆分：preds 取除最后一个元素外的全部，target 取最后一个元素
        inputs = tuple(list(preds) + [target])  # 重新拼回 (pred..., target) 元组（值不变，供下方分支解包使用）
        if self.aux:  # 启用辅助损失
            return dict(loss=self._aux_forward(*inputs))  # 走多输出分支，返回 {'loss': 标量张量}
        else:  # 未启用辅助损失的分支
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))  # 单输出直接用父类交叉熵，返回 {'loss': ...}


# reference: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/loss.py  # 原注释保留：参考 PyTorch-Encoding 的实现
class EncNetLoss(nn.CrossEntropyLoss):  # EncNet 损失：交叉熵 + SE（通道注意力）二元交叉熵，可选辅助损失
    """2D Cross Entropy Loss with SE Loss"""  # 原有 docstring：带 SE 损失的 2D 交叉熵

    # 形参说明: se_loss 是否启用 SE 损失(默认True); se_weight SE 损失权重(默认0.2); nclass 类别数(默认19, 对应 Cityscapes)
    # 形参说明: aux 是否启用辅助损失(默认False); aux_weight 辅助损失权重(默认0.4); weight CE 类别权重; ignore_index 忽略标签(默认-1)
    def __init__(self, se_loss=True, se_weight=0.2, nclass=19, aux=False,
                 aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):  # （构造器参数续行）辅助损失权重/CE 类别权重/忽略标签等
        super(EncNetLoss, self).__init__(weight, None, ignore_index)  # 初始化父类 CE：位置参数依次为 weight、size_average(None)、ignore_index
        self.se_loss = se_loss  # SE 损失开关
        self.aux = aux  # 辅助损失开关
        self.nclass = nclass  # 类别数，用于生成 0/1 类别出现向量
        self.se_weight = se_weight  # SE 损失权重
        self.aux_weight = aux_weight  # 辅助损失权重
        self.bceloss = nn.BCELoss(weight)  # 二元交叉熵，用于 SE 损失；【注意】可疑写法：把 CE 的逐类别权重张量直接传给 BCELoss 的 weight，二者语义并不对应（BCE 的 weight 应与输入逐元素同形）

    def forward(self, *inputs):  # 前向：inputs 依配置为 (pred, target) / (pred1, pred2, target) / (pred, se_pred, target) / (pred1, se_pred, pred2, target)
        preds, target = tuple(inputs)  # preds 为预测组，target 为最后一个标签
        inputs = tuple(list(preds) + [target])  # 原样重组元组（值不变）
        if not self.se_loss and not self.aux:  # 既无 SE 也无辅助损失
            return super(EncNetLoss, self).forward(*inputs)  # 退化为纯交叉熵；【注意】此分支直接返回张量而非 dict，与其他分支返回 dict 不一致
        elif not self.se_loss:  # 仅辅助损失：inputs = (pred1, pred2, target)
            pred1, pred2, target = tuple(inputs)  # 解包主/辅预测与标签
            loss1 = super(EncNetLoss, self).forward(pred1, target)  # 主输出交叉熵
            loss2 = super(EncNetLoss, self).forward(pred2, target)  # 辅助输出交叉熵
            return dict(loss=loss1 + self.aux_weight * loss2)  # 返回 {'loss': loss1 + aux_weight * loss2}
        elif not self.aux:  # 仅 SE 损失：inputs = (pred, se_pred, target)
            pred, se_pred, target = tuple(inputs)  # 解包分割预测、SE 预测与标签
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)  # 由标签生成 0/1 类别出现向量 (B, nclass)，并转到 pred 的设备与类型
            loss1 = super(EncNetLoss, self).forward(pred, target)  # 主分割交叉熵
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)  # SE 损失：sigmoid 后的通道注意力预测 vs 类别出现向量的 BCE
            return dict(loss=loss1 + self.se_weight * loss2)  # 返回 {'loss': loss1 + se_weight * loss2}
        else:  # SE 与辅助损失同时启用：inputs = (pred1, se_pred, pred2, target)
            pred1, se_pred, pred2, target = tuple(inputs)  # 解包主预测、SE 预测、辅助预测与标签
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)  # 0/1 类别出现向量 (B, nclass)
            loss1 = super(EncNetLoss, self).forward(pred1, target)  # 主输出交叉熵
            loss2 = super(EncNetLoss, self).forward(pred2, target)  # 辅助输出交叉熵
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)  # SE 损失（BCE）
            return dict(loss=loss1 + self.aux_weight * loss2 + self.se_weight * loss3)  # 返回三项加权和

    @staticmethod  # 静态方法装饰器（无需实例即可调用）
    def _get_batch_label_vector(target, nclass):  # 静态方法：把每张图的标签图转为"出现了哪些类别"的 0/1 向量
        # target is a 3D Variable BxHxW, output is 2D BxnClass  # 原注释保留：输入 (B,H,W)，输出 (B,nClass)
        batch = target.size(0)  # batch 大小
        tvect = Variable(torch.zeros(batch, nclass))  # 初始化全 0 目标矩阵 (B, nclass)；【注意】Variable 已废弃，直接用 torch.zeros 即可
        for i in range(batch):  # 逐张图统计类别出现情况
            hist = torch.histc(target[i].cpu().data.float(),  # 对单张标签做直方图统计（.cpu() 因 histc 仅支持 CPU；【注意】.data 是取消梯度追踪的旧式写法，现代推荐 .detach()）
                               bins=nclass, min=0,  # 直方图 bin 数 = 类别数，统计下界 0
                               max=nclass - 1)  # 统计上界 nclass-1，使每个 bin 恰对应一个类别 id
            vect = hist > 0  # 出现次数 > 0 的类别置 True，得到 0/1 出现向量
            tvect[i] = vect  # 写入该样本的目标向量
        return tvect  # 返回 (B, nclass) 的 0/1 类别出现矩阵


# TODO: optim function  # 原注释保留：待优化
class ICNetLoss(nn.CrossEntropyLoss):  # ICNet 损失：对 1/4、1/8、1/16 三个分辨率的辅助输出，分别将标签下采样后计算 CE 并加权求和
    """Cross Entropy Loss for ICNet"""  # 原有 docstring：ICNet 的交叉熵损失

    # 形参说明: nclass 类别数; aux_weight 低分辨率分支损失权重(默认0.4); ignore_index 忽略标签(默认-1)
    def __init__(self, nclass, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)  # 初始化父类交叉熵
        self.nclass = nclass  # 保存类别数（当前实现中未直接使用）
        self.aux_weight = aux_weight  # 低分辨率分支的损失权重

    def forward(self, *inputs):  # 前向：inputs = (pred, pred_sub4, pred_sub8, pred_sub16, target)
        preds, target = tuple(inputs)  # preds 为四个预测输出，target 为标签
        inputs = tuple(list(preds) + [target])  # 原样重组元组（值不变）

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)  # 解包：全分辨率 pred 与 1/4、1/8、1/16 三个分辨率的辅助输出
        # [batch, W, H] -> [batch, 1, W, H]  # 原注释保留：形状变化说明
        target = target.unsqueeze(1).float()  # 标签增加 channel 维并转 float 以便插值: (N,H,W) -> (N,1,H,W)
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()  # 标签双线性下采样到 1/4 分支尺寸，去掉 channel 维后转 long: -> (N,H/4,W/4)；【注意】双线性插值产生非整数标签值后直接 .long() 截断，通常 nearest 插值对标签更稳妥
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()  # 同理下采样到 1/8 尺寸: -> (N,H/8,W/8)
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(  # 同理下采样到 1/16 尺寸（链式调用换行）
            1).long()  # squeeze(1).long() 之后得到 (N,H/16,W/16)
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)  # 1/4 分支交叉熵（主监督，权重为 1）
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)  # 1/8 分支交叉熵
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)  # 1/16 分支交叉熵
        return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)  # 总损失 = loss1 + 0.4*loss2 + 0.4*loss3；【注意】解包出的全分辨率 pred 未参与损失（符合 ICNet 论文仅监督三个降采样分支的设定，但变量本身冗余）


class OhemCrossEntropy2d(nn.Module):  # OHEM 在线难例挖掘交叉熵（2D 分割版）：只对困难像素计算 CE 损失
    # 形参说明: ignore_index 忽略标签(默认-1); thresh 概率阈值(默认0.7, 真实类概率低于它的像素视为难例)
    # 形参说明: min_kept 最少保留像素数(默认100000, 防止难例太少导致损失噪声过大); use_weight 是否使用 Cityscapes 类别频率权重
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=True, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()  # 初始化父类 nn.Module
        self.ignore_index = ignore_index  # 保存忽略标签值
        self.thresh = float(thresh)  # 难例概率阈值（转 float 便于比较）
        self.min_kept = int(min_kept)  # 最少保留像素数（转 int）
        if use_weight:  # 使用类别权重
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,  # 19 个权重对应 Cityscapes 19 类，由类别出现频率折算（频率低的类权重高，缓解不均衡）
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,  # （接上行权重列表）
                                        1.0865, 1.1529, 1.0507])  # （权重列表结尾）
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)  # 带类别权重与 ignore 的 CE 作为最终损失器
        else:  # 不使用类别权重的分支
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)  # 不带权重的 CE

    def forward(self, pred, target):  # OHEM 前向：pred 为 logits (N,C,H,W)，target 为标签 (N,H,W)
        n, c, h, w = pred.size()  # 读取 batch 数、类别数、高、宽
        target = target.view(-1)  # 标签展平为一维 (N*H*W,)
        valid_mask = target.ne(self.ignore_index)  # 有效像素掩码：标签 != ignore_index 的位置为 True
        target = target * valid_mask.long()  # 将忽略位置的标签置 0，避免后续按类别索引时越界
        num_valid = valid_mask.sum()  # 有效像素总数

        prob = F.softmax(pred, dim=1)  # 沿类别维 softmax 得到概率图 (N,C,H,W)
        prob = prob.transpose(0, 1).reshape(c, -1)  # 维度重排: (N,C,H,W) -> (C,N,H,W) -> (C, N*H*W)，每列对应一个像素的各类概率

        if self.min_kept > num_valid:  # 有效像素数少于最少保留数
            print("Lables: {}".format(num_valid))  # 仅打印提示，跳过难例筛选；【注意】"Lables" 拼写错误（应为 Labels），且用 print 而非 logger
        elif num_valid > 0:  # 存在有效像素才进行难例筛选
            prob = prob.masked_fill_(~valid_mask, 1)  # 无效像素概率填 1（最"简单"）以确保不被选为难例；用 ~ 取反，兼容新版 PyTorch 禁止 bool 减法的限制
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long, device=target.device)]  # 花式索引取每个像素在其真实类别上的预测概率: (N*H*W,)，索引张量必须与 target 同设备
            threshold = self.thresh  # 初始阈值取 0.7
            if self.min_kept > 0:  # 需要保证最少保留 min_kept 个像素
                index = mask_prob.argsort()  # 按概率升序排序（最难的排最前）
                threshold_index = index[min(len(index), self.min_kept) - 1]  # 第 min_kept 难像素的下标（min 防止越界）
                if mask_prob[threshold_index] > self.thresh:  # 若第 min_kept 难的像素概率已超过 0.7（说明难例不足）
                    threshold = mask_prob[threshold_index]  # 抬高阈值到该像素的概率，保证仍保留 min_kept 个像素
            kept_mask = mask_prob.le(threshold)  # 保留掩码：概率 <= 阈值（即难例）的位置为 True
            valid_mask = valid_mask * kept_mask  # 最终参与损失的像素 = 有效 且 属于难例
            target = target * kept_mask.long()  # 未保留像素的标签置 0

        target = target.masked_fill_(~valid_mask, self.ignore_index)  # 未保留/无效像素的标签统一改回 ignore_index，使 CE 彻底忽略它们
        target = target.view(n, h, w)  # 标签还原为 (N,H,W) 形状

        return self.criterion(pred, target)  # 用（可选类别加权的）CE 在保留下来的难例像素上计算损失


class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):  # 带辅助损失支持的 OHEM 混合损失：继承 OhemCrossEntropy2d 并扩展多输出联合监督
    # 形参说明: aux 是否启用辅助损失(默认False); aux_weight 辅助损失权重(默认0.4); weight 占位参数(见下方 bceloss 说明); ignore_index 忽略标签
    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_index=ignore_index)  # 初始化父类 OHEM 损失（thresh/min_kept 等均用默认值）
        self.aux = aux  # 辅助损失开关
        self.aux_weight = aux_weight  # 辅助损失权重
        self.bceloss = nn.BCELoss(weight)  # 【注意】可疑写法：此处创建的 bceloss 在本类中从未被使用，疑似从 EncNetLoss 复制粘贴的残留代码

    def _aux_forward(self, *inputs, **kwargs):  # 辅助分支前向：inputs = (pred_main, pred_aux..., target)
        *preds, target = tuple(inputs)  # 星号解包：前面为各预测输出，最后一个为标签

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)  # 主输出的 OHEM 损失
        for i in range(1, len(preds)):  # 遍历各辅助输出
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)  # 辅助输出的 OHEM 损失（对同一标签）
            loss += self.aux_weight * aux_loss  # 加权累加
        return loss  # 返回总损失

    def forward(self, *inputs):  # 统一入口：inputs = (pred..., target)
        preds, target = tuple(inputs)  # 拆分预测组与标签
        inputs = tuple(list(preds) + [target])  # 原样重组元组
        if self.aux:  # 启用辅助损失
            return dict(loss=self._aux_forward(*inputs))  # 走多输出 OHEM 分支
        else:  # 未启用辅助损失的分支
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs))  # 单输出直接调用 OHEM 前向


def get_segmentation_loss(model, use_ohem=False, **kwargs):  # 损失工厂函数：按模型名与是否启用 OHEM 创建对应损失实例
    if use_ohem:  # 指定使用 OHEM 难例挖掘
        return MixSoftmaxCrossEntropyOHEMLoss(**kwargs)  # 返回 OHEM 版混合损失（此分支忽略 model 参数）

    model = model.lower()  # 模型名转小写以便匹配
    if model == 'encnet':  # EncNet：需要额外的通道注意力 SE 损失
        return EncNetLoss(**kwargs)  # 创建 EncNet 损失并返回
    elif model == 'icnet':  # ICNet：需要对多分辨率输出分别下采样标签
        return ICNetLoss(**kwargs)  # 创建 ICNet 损失并返回
    else:  # 其余模型：通用带辅助输出的交叉熵
        return MixSoftmaxCrossEntropyLoss(**kwargs)  # 创建通用混合交叉熵并返回
