# ============================================================================
# 【文件说明】dfanet.py —— 深度特征聚合网络 DFANet（Deep Feature Aggregation Network）
#              的 PyTorch 实现。以轻量 XceptionA（深度可分离卷积骨干）为编码器，
#              将编码器级联执行三遍（stage1→stage2→stage3）：后一遍的各级输入都拼接
#              前一遍对应层级的输出（子网级聚合）；解码端把三遍的 enc2 层特征与三遍
#              FCA 注意力输出分别 1x1 降维、对齐到 1/4 分辨率后逐元素相加融合，
#              最后 1x1 卷积分类并上采样回原图尺寸。
# 【核心思想】论文《DFANet: Deep Feature Aggregation for Real-Time Semantic
#            Segmentation》(ICCV 2019, Henghui Ding et al.)。
#            思想：对单个轻量骨干反复精炼（refine）优于简单堆宽堆深——把前一轮编码
#            的高层语义（经 FCAttention 全局注意力加权、上采样对齐）重新注入下一轮
#            编码器同级入口，实现“同一骨干多遍聚合”；解码端不用复杂上采样结构，
#            直接多尺度对齐后逐元素相加完成特征级聚合。
# ============================================================================
""" Deep Feature Aggregation"""  # 原英文 docstring 保留：深度特征聚合
import torch  # PyTorch 主库：张量运算与权重加载
import torch.nn as nn  # 神经网络模块：nn.Module、卷积/BN/激活等网络层
import torch.nn.functional as F  # 函数式接口：双线性上采样 F.interpolate 等

from core.models.base_models import Enc, FCAttention, get_xception_a  # 导入 XceptionA 相关组件：Enc 编码块、FCAttention 全局注意力、骨干构造函数（base_models/xception.py）
from core.nn import _ConvBNReLU  # 导入通用“卷积+BN+ReLU”基础模块（core/nn/basic.py）

__all__ = ['DFANet', 'get_dfanet', 'get_dfanet_citys']  # 模块公开接口：模型类 + 两个工厂函数


class DFANet(nn.Module):  # DFANet 主网络：XceptionA 编码器级联三遍 + 多尺度解码融合
    # ----------------------- __init__ 参数说明 -----------------------
    # nclass：分割类别数（输出通道数）
    # backbone / jpu：接口占位参数，本模型未使用【注意】
    # aux：接口占位参数——本实现既未保存 self.aux 也未构建辅助分支【注意】forward 中无任何 aux 逻辑
    # pretrained_base：是否加载 XceptionA 的 ImageNet 预训练权重
    # **kwargs：透传给各子模块（如 norm_layer）
    def __init__(self, nclass, backbone='', aux=False, jpu=False, pretrained_base=False, **kwargs):  # 构造函数
        super(DFANet, self).__init__()  # 调用父类构造函数
        self.pretrained = get_xception_a(pretrained_base, **kwargs)  # 第一遍编码器：XceptionA（conv1→enc2→enc3→enc4→fca，内部为深度可分离卷积）

        self.enc2_2 = Enc(240, 48, 4, **kwargs)  # 第二遍 enc2：输入 240=48(第一遍 enc2)+192(第一遍 fca 上采样输出)，输出 48，4 个 BlockA，首块 stride=2
        self.enc3_2 = Enc(144, 96, 6, **kwargs)  # 第二遍 enc3：输入 144=96(第一遍 enc3)+48(第二遍 enc2)，输出 96，6 个 BlockA
        self.enc4_2 = Enc(288, 192, 4, **kwargs)  # 第二遍 enc4：输入 288=192(第一遍 enc4)+96(第二遍 enc3)，输出 192，4 个 BlockA
        self.fca_2 = FCAttention(192, **kwargs)  # 第二遍全局注意力：GAP→FC(192→1000)→1x1 卷积回 192，通道加权

        self.enc2_3 = Enc(240, 48, 4, **kwargs)  # 第三遍 enc2：输入 240=48(第二遍 enc2)+192(第二遍 fca 上采样输出)，输出 48
        self.enc3_3 = Enc(144, 96, 6, **kwargs)  # 第三遍 enc3：输入 144=96(第二遍 enc3)+48(第三遍 enc2)，输出 96
        self.enc3_4 = Enc(288, 192, 4, **kwargs)  # 第三遍 enc4：【注意】命名疑误——按本文件 encX_Y 约定（X=层级、Y=第几遍）应为 enc4_3，此处写成 enc3_4；功能不受影响，但可读性差
        self.fca_3 = FCAttention(192, **kwargs)  # 第三遍全局注意力（结构同 fca_2）

        self.enc2_1_reduce = _ConvBNReLU(48, 32, 1, **kwargs)  # 解码端 1x1 卷积：第一遍 enc2 特征 48→32
        self.enc2_2_reduce = _ConvBNReLU(48, 32, 1, **kwargs)  # 解码端 1x1 卷积：第二遍 enc2 特征 48→32
        self.enc2_3_reduce = _ConvBNReLU(48, 32, 1, **kwargs)  # 解码端 1x1 卷积：第三遍 enc2 特征 48→32
        self.conv_fusion = _ConvBNReLU(32, 32, 1, **kwargs)  # 1x1 卷积：三遍 enc2 特征相加融合后再精炼（32→32）

        self.fca_1_reduce = _ConvBNReLU(192, 32, 1, **kwargs)  # 解码端 1x1 卷积：第一遍 fca 特征 192→32
        self.fca_2_reduce = _ConvBNReLU(192, 32, 1, **kwargs)  # 解码端 1x1 卷积：第二遍 fca 特征 192→32
        self.fca_3_reduce = _ConvBNReLU(192, 32, 1, **kwargs)  # 解码端 1x1 卷积：第三遍 fca 特征 192→32
        self.conv_out = nn.Conv2d(32, nclass, 1)  # 最终 1x1 分类卷积：32→nclass

        self.__setattr__('exclusive', ['enc2_2', 'enc3_2', 'enc4_2', 'fca_2', 'enc2_3', 'enc3_3', 'enc3_4', 'fca_3',  # 登记除骨干外的自有模块名单（骨干 self.pretrained 另列）
                                       'enc2_1_reduce', 'enc2_2_reduce', 'enc2_3_reduce', 'conv_fusion', 'fca_1_reduce',  # 续行：解码端 enc2 分支相关模块
                                       'fca_2_reduce', 'fca_3_reduce', 'conv_out'])  # 续行：解码端 fca 分支与输出卷积

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像 (N, 3, H, W)；返回单元素元组 ((N, nclass, H, W),)
    def forward(self, x):  # 前向：三遍编码级联聚合 → enc2 层特征融合 + fca 层特征融合 → 分类 → 上采样
        # backbone
        stage1_conv1 = self.pretrained.conv1(x)  # 第一遍 stem：3x3 stride=2 卷积+BN+ReLU → (N, 8, H/2, W/2)
        stage1_enc2 = self.pretrained.enc2(stage1_conv1)  # 第一遍 enc2（首块 stride=2）→ (N, 48, H/4, W/4)
        stage1_enc3 = self.pretrained.enc3(stage1_enc2)  # 第一遍 enc3 → (N, 96, H/8, W/8)
        stage1_enc4 = self.pretrained.enc4(stage1_enc3)  # 第一遍 enc4 → (N, 192, H/16, W/16)
        stage1_fca = self.pretrained.fca(stage1_enc4)  # 第一遍 FCA 通道注意力加权 → (N, 192, H/16, W/16)
        stage1_out = F.interpolate(stage1_fca, scale_factor=4, mode='bilinear', align_corners=True)  # ×4 上采样到 1/4 分辨率 → (N, 192, H/4, W/4)，与 enc2 对齐后注入第二遍

        # stage2
        stage2_enc2 = self.enc2_2(torch.cat([stage1_enc2, stage1_out], dim=1))  # 第一遍 enc2(48 通道,1/4) + 第一遍 fca 上采样(192 通道,1/4) 拼接 240 → Enc（首块 stride=2）→ (N, 48, H/8, W/8)
        stage2_enc3 = self.enc3_2(torch.cat([stage1_enc3, stage2_enc2], dim=1))  # 第一遍 enc3(96,1/8) + 第二遍 enc2(48,1/8) 拼接 144 → (N, 96, H/16, W/16)
        stage2_enc4 = self.enc4_2(torch.cat([stage1_enc4, stage2_enc3], dim=1))  # 第一遍 enc4(192,1/16) + 第二遍 enc3(96,1/16) 拼接 288 → (N, 192, H/32, W/32)
        stage2_fca = self.fca_2(stage2_enc4)  # 第二遍 FCA → (N, 192, H/32, W/32)
        stage2_out = F.interpolate(stage2_fca, scale_factor=4, mode='bilinear', align_corners=True)  # ×4 上采样到 1/8 分辨率 → (N, 192, H/8, W/8)，与第二遍 enc2 对齐后注入第三遍

        # stage3
        stage3_enc2 = self.enc2_3(torch.cat([stage2_enc2, stage2_out], dim=1))  # 第二遍 enc2(48,1/8) + 第二遍 fca 上采样(192,1/8) 拼接 240 → (N, 48, H/16, W/16)
        stage3_enc3 = self.enc3_3(torch.cat([stage2_enc3, stage3_enc2], dim=1))  # 第二遍 enc3(96,1/16) + 第三遍 enc2(48,1/16) 拼接 144 → (N, 96, H/32, W/32)
        stage3_enc4 = self.enc3_4(torch.cat([stage2_enc4, stage3_enc3], dim=1))  # 第二遍 enc4(192,1/32) + 第三遍 enc3(96,1/32) 拼接 288 → (N, 192, H/64, W/64)
        stage3_fca = self.fca_3(stage3_enc4)  # 第三遍 FCA → (N, 192, H/64, W/64)

        stage1_enc2_decoder = self.enc2_1_reduce(stage1_enc2)  # 解码：第一遍 enc2(1/4) 1x1 降维 → (N, 32, H/4, W/4)
        stage2_enc2_docoder = F.interpolate(self.enc2_2_reduce(stage2_enc2), scale_factor=2,  # 解码：第二遍 enc2(1/8) 降维后 ×2 上采样（【注意】变量名 docoder 系原代码拼写错误，应为 decoder）
                                            mode='bilinear', align_corners=True)  # → (N, 32, H/4, W/4)
        stage3_enc2_decoder = F.interpolate(self.enc2_3_reduce(stage3_enc2), scale_factor=4,  # 解码：第三遍 enc2(1/16) 降维后 ×4 上采样
                                            mode='bilinear', align_corners=True)  # → (N, 32, H/4, W/4)
        fusion = stage1_enc2_decoder + stage2_enc2_docoder + stage3_enc2_decoder  # 三遍 enc2 特征逐元素相加（均已对齐到 1/4 分辨率、32 通道）→ (N, 32, H/4, W/4)
        fusion = self.conv_fusion(fusion)  # 1x1 精炼卷积 → (N, 32, H/4, W/4)

        stage1_fca_decoder = F.interpolate(self.fca_1_reduce(stage1_fca), scale_factor=4,  # 解码：第一遍 fca(1/16) 降维后 ×4 上采样
                                           mode='bilinear', align_corners=True)  # → (N, 32, H/4, W/4)
        stage2_fca_decoder = F.interpolate(self.fca_2_reduce(stage2_fca), scale_factor=8,  # 解码：第二遍 fca(1/32) 降维后 ×8 上采样
                                           mode='bilinear', align_corners=True)  # → (N, 32, H/4, W/4)
        stage3_fca_decoder = F.interpolate(self.fca_3_reduce(stage3_fca), scale_factor=16,  # 解码：第三遍 fca(1/64) 降维后 ×16 上采样
                                           mode='bilinear', align_corners=True)  # → (N, 32, H/4, W/4)
        fusion = fusion + stage1_fca_decoder + stage2_fca_decoder + stage3_fca_decoder  # enc2 层融合结果再与三遍 fca 解码特征逐元素相加（特征级聚合）→ (N, 32, H/4, W/4)

        outputs = list()  # 输出收集列表
        out = self.conv_out(fusion)  # 1x1 分类卷积 → (N, nclass, H/4, W/4)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)  # ×4 上采样回原尺寸 → (N, nclass, H, W)
        outputs.append(out)  # 收集主输出

        return tuple(outputs)  # 返回单元素元组（与其他模型输出接口一致）


# ----------------------- get_dfanet 参数说明 -----------------------
# dataset：数据集名称（决定类别数）
# backbone：占位参数（本模型未使用）
# pretrained：是否加载本模型的分割预训练权重
# root：权重文件查找根目录
# pretrained_base：是否加载 XceptionA 骨干预训练权重
# **kwargs：透传；pretrained=True 时须包含 local_rank 用于指定加载设备
def get_dfanet(dataset='citys', backbone='', pretrained=False, root='~/.torch/models',  # 工厂函数：按数据集构建 DFANet 并可选加载权重
               pretrained_base=True, **kwargs):  # 续行：【注意】此处默认 True，与 DFANet 类构造函数默认 False 不一致（走工厂路径会自动加载骨干预训练）
    acronyms = {  # 数据集名 → 权重文件名缩写映射
        'pascal_voc': 'pascal_voc',  # Pascal VOC 2012
        'pascal_aug': 'pascal_aug',  # VOC 增广集（SBD）
        'ade20k': 'ade',  # ADE20K 场景解析数据集
        'coco': 'coco',  # COCO Stuff
        'citys': 'citys',  # Cityscapes 城市街景（19 类）
    }  # 缩写映射表结束
    from ..data.dataloader import datasets  # 延迟导入数据集注册表（相对导入写法）
    model = DFANet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)  # 按数据集类别数构建模型
    if pretrained:  # 若需加载分割预训练权重
        from .model_store import get_model_file  # 延迟导入权重文件路径解析工具
        device = torch.device(kwargs['local_rank'])  # 【注意】可疑写法：kwargs 未包含 local_rank 时会 KeyError；且以 int 直接构造 torch.device 的行为依赖 PyTorch 版本（更稳妥写法为 torch.device('cuda', rank)）
        model.load_state_dict(torch.load(get_model_file('dfanet_%s' % (acronyms[dataset]), root=root),  # 拼出权重文件名（如 dfanet_citys.pth）并读取
                              map_location=device))  # 将权重映射到指定设备
    return model  # 返回构建/加载完成的模型


def get_dfanet_citys(**kwargs):  # 便捷构造函数：Cityscapes 数据集（19 类）DFANet
    return get_dfanet('citys', **kwargs)  # 委托给 get_dfanet


if __name__ == '__main__':  # 直接运行本文件时的小型自测（不触发训练）
    model = get_dfanet_citys()  # 构建 Cityscapes 版 DFANet（工厂默认 pretrained_base=True，会尝试加载 XceptionA 预训练权重）
