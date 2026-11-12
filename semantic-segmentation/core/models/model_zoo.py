# ============================================================================
# 【文件说明】model_zoo.py —— 模型动物园/注册表：集中导入全部分割模型的工厂函数，
#              维护"模型名 → 工厂函数"映射，对外提供 get_model / get_model_list /
#              get_segmentation_model 三个创建入口。
#              工程位置：core/models/model_zoo.py，是 core/models/__init__.py 的实际实现，
#              训练/推理脚本通过它按名称创建任意分割模型。
# 【核心思想】工厂模式 + 注册表：各模型子模块通过 import * 暴露 get_xxx 工厂函数；
#            _models 字典登记"预训练权重级"名称（含骨干与数据集信息）；
#            get_segmentation_model 按算法短名（fcn/psp/deeplabv3...）调用通用工厂，
#            数据集与骨干由调用方通过 kwargs 指定。
# ============================================================================
"""Model store which handles pretrained models """  # 原模块 docstring 保留（实际职责更接近模型注册表）
from .fcn import *  # 导入 FCN32s/16s/8s 系列（VGG 骨干）的全部公开工厂函数
from .fcnv2 import *  # 导入 FCN v2（ResNet 骨干）的工厂函数
from .pspnet import *  # 导入 PSPNet（金字塔场景解析）工厂函数
from .deeplabv3 import *  # 导入 DeepLabV3 工厂函数
from .deeplabv3_plus import *  # 导入 DeepLabV3+ 工厂函数
from .danet import *  # 导入 DANet（双注意力网络）工厂函数
from .denseaspp import *  # 导入 DenseASPP 工厂函数
from .bisenet import *  # 导入 BiSeNet（双边分割网络）工厂函数
from .encnet import *  # 导入 EncNet（上下文编码网络）工厂函数
from .dunet import *  # 导入 DU-Net 工厂函数
from .icnet import *  # 导入 ICNet（级联网络，面向实时）工厂函数
from .enet import *  # 导入 ENet（轻量级实时网络）工厂函数
from .ocnet import *  # 导入 OCNet（目标上下文网络）工厂函数
from .psanet import *  # 导入 PSANet（点空间注意力网络）工厂函数
from .cgnet import *  # 导入 CGNet（上下文引导网络）工厂函数
from .espnet import *  # 导入 ESPNet（轻量级实时网络）工厂函数
from .lednet import *  # 导入 LEDNet（轻量级编码-解码网络）工厂函数
from .dfanet import *  # 导入 DFANet（深度特征聚合网络）工厂函数

__all__ = ['get_model', 'get_model_list', 'get_segmentation_model']  # 模块公开接口：三个模型获取入口

_models = {  # 注册表：预训练权重级模型名（算法_骨干_数据集）→ 对应预设工厂函数
    'fcn32s_vgg16_voc': get_fcn32s_vgg16_voc,  # FCN32s · VGG16 · Pascal VOC
    'fcn16s_vgg16_voc': get_fcn16s_vgg16_voc,  # FCN16s · VGG16 · Pascal VOC
    'fcn8s_vgg16_voc': get_fcn8s_vgg16_voc,  # FCN8s · VGG16 · Pascal VOC
    'fcn_resnet50_voc': get_fcn_resnet50_voc,  # FCN(v2) · ResNet50 · Pascal VOC
    'fcn_resnet101_voc': get_fcn_resnet101_voc,  # FCN(v2) · ResNet101 · Pascal VOC
    'fcn_resnet152_voc': get_fcn_resnet152_voc,  # FCN(v2) · ResNet152 · Pascal VOC
    'psp_resnet50_voc': get_psp_resnet50_voc,  # PSPNet · ResNet50 · Pascal VOC
    'psp_resnet50_ade': get_psp_resnet50_ade,  # PSPNet · ResNet50 · ADE20K
    'psp_resnet101_voc': get_psp_resnet101_voc,  # PSPNet · ResNet101 · Pascal VOC
    'psp_resnet101_ade': get_psp_resnet101_ade,  # PSPNet · ResNet101 · ADE20K
    'psp_resnet101_citys': get_psp_resnet101_citys,  # PSPNet · ResNet101 · Cityscapes
    'psp_resnet101_coco': get_psp_resnet101_coco,  # PSPNet · ResNet101 · COCO
    'deeplabv3_resnet50_voc': get_deeplabv3_resnet50_voc,  # DeepLabV3 · ResNet50 · Pascal VOC
    'deeplabv3_resnet101_voc': get_deeplabv3_resnet101_voc,  # DeepLabV3 · ResNet101 · Pascal VOC
    'deeplabv3_resnet152_voc': get_deeplabv3_resnet152_voc,  # DeepLabV3 · ResNet152 · Pascal VOC
    'deeplabv3_resnet50_ade': get_deeplabv3_resnet50_ade,  # DeepLabV3 · ResNet50 · ADE20K
    'deeplabv3_resnet101_ade': get_deeplabv3_resnet101_ade,  # DeepLabV3 · ResNet101 · ADE20K
    'deeplabv3_resnet152_ade': get_deeplabv3_resnet152_ade,  # DeepLabV3 · ResNet152 · ADE20K
    'deeplabv3_plus_xception_voc': get_deeplabv3_plus_xception_voc,  # DeepLabV3+ · Xception · Pascal VOC
    'danet_resnet50_ciyts': get_danet_resnet50_citys,  # 【注意】键名拼写错误 'ciyts'（应为 citys），用正确拼写查找会 KeyError
    'danet_resnet101_citys': get_danet_resnet101_citys,  # DANet · ResNet101 · Cityscapes
    'danet_resnet152_citys': get_danet_resnet152_citys,  # DANet · ResNet152 · Cityscapes
    'denseaspp_densenet121_citys': get_denseaspp_densenet121_citys,  # DenseASPP · DenseNet121 · Cityscapes
    'denseaspp_densenet161_citys': get_denseaspp_densenet161_citys,  # DenseASPP · DenseNet161 · Cityscapes
    'denseaspp_densenet169_citys': get_denseaspp_densenet169_citys,  # DenseASPP · DenseNet169 · Cityscapes
    'denseaspp_densenet201_citys': get_denseaspp_densenet201_citys,  # DenseASPP · DenseNet201 · Cityscapes
    'bisenet_resnet18_citys': get_bisenet_resnet18_citys,  # BiSeNet · ResNet18 · Cityscapes
    'encnet_resnet50_ade': get_encnet_resnet50_ade,  # EncNet · ResNet50 · ADE20K
    'encnet_resnet101_ade': get_encnet_resnet101_ade,  # EncNet · ResNet101 · ADE20K
    'encnet_resnet152_ade': get_encnet_resnet152_ade,  # EncNet · ResNet152 · ADE20K
    'dunet_resnet50_pascal_voc': get_dunet_resnet50_pascal_voc,  # DU-Net · ResNet50 · Pascal VOC
    'dunet_resnet101_pascal_voc': get_dunet_resnet101_pascal_voc,  # DU-Net · ResNet101 · Pascal VOC
    'dunet_resnet152_pascal_voc': get_dunet_resnet152_pascal_voc,  # DU-Net · ResNet152 · Pascal VOC
    'icnet_resnet50_citys': get_icnet_resnet50_citys,  # ICNet · ResNet50 · Cityscapes
    'icnet_resnet101_citys': get_icnet_resnet101_citys,  # ICNet · ResNet101 · Cityscapes
    'icnet_resnet152_citys': get_icnet_resnet152_citys,  # ICNet · ResNet152 · Cityscapes
    'enet_citys': get_enet_citys,  # ENet · Cityscapes（轻量级实时）
    'base_ocnet_resnet101_citys': get_base_ocnet_resnet101_citys,  # OCNet(BASE 结构) · ResNet101 · Cityscapes
    'pyramid_ocnet_resnet101_citys': get_pyramid_ocnet_resnet101_citys,  # OCNet(金字塔结构) · ResNet101 · Cityscapes
    'asp_ocnet_resnet101_citys': get_asp_ocnet_resnet101_citys,  # OCNet(ASP 结构) · ResNet101 · Cityscapes
    'psanet_resnet50_voc': get_psanet_resnet50_voc,  # PSANet · ResNet50 · Pascal VOC
    'psanet_resnet101_voc': get_psanet_resnet101_voc,  # PSANet · ResNet101 · Pascal VOC
    'psanet_resnet152_voc': get_psanet_resnet152_voc,  # PSANet · ResNet152 · Pascal VOC
    'psanet_resnet50_citys': get_psanet_resnet50_citys,  # PSANet · ResNet50 · Cityscapes
    'psanet_resnet101_citys': get_psanet_resnet101_citys,  # PSANet · ResNet101 · Cityscapes
    'psanet_resnet152_citys': get_psanet_resnet152_citys,  # PSANet · ResNet152 · Cityscapes
    'cgnet_citys': get_cgnet_citys,  # CGNet · Cityscapes（轻量级）
    'espnet_citys': get_espnet_citys,  # ESPNet · Cityscapes（轻量级实时）
    'lednet_citys': get_lednet_citys,  # LEDNet · Cityscapes（轻量级实时）
    'dfanet_citys': get_dfanet_citys,  # DFANet · Cityscapes（轻量级实时）
}  # 模型注册表 _models 字典结束（预训练权重级名称 → 预设工厂函数）


# ----------------------- get_model 参数说明 -----------------------
# name：模型注册名（大小写不敏感），须为 _models 的键
# **kwargs：透传给对应工厂函数（pretrained、local_rank 等）
# 返回：构建好的网络实例
def get_model(name, **kwargs):  # 按预训练权重级注册名创建模型实例
    name = name.lower()  # 名称统一转小写，便于匹配
    if name not in _models:  # 未注册的名称
        err_str = '"%s" is not among the following model list:\n\t' % (name)  # 错误信息头部（\n\t 为换行加制表缩进）
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))  # 拼接排序后的全部可用模型名列表
        raise ValueError(err_str)  # 抛 ValueError 并列出可用模型供参考
    net = _models[name](**kwargs)  # 调用对应工厂函数构建网络
    return net  # 返回模型实例


# ----------------------- get_model_list 参数说明 -----------------------
# 返回：全部已注册模型名的集合（字典键视图）
def get_model_list():  # 列出所有可用的模型注册名
    return _models.keys()  # 返回 dict_keys 视图（【注意】非副本，遍历期间修改 _models 会影响该视图）


# ----------------------- get_segmentation_model 参数说明 -----------------------
# model：算法短名（如 'fcn'、'deeplabv3_plus'），为下方字典的键
# **kwargs：透传给通用工厂（dataset/backbone/pretrained 等）
# 返回：构建好的分割模型实例
def get_segmentation_model(model, **kwargs):  # 主入口：按算法短名创建分割模型（数据集/骨干由 kwargs 指定）
    models = {  # 算法短名 → 通用工厂函数映射（不含骨干/数据集信息，与 _models 互补）
        'fcn32s': get_fcn32s,  # FCN32s 通用工厂（VGG 骨干）
        'fcn16s': get_fcn16s,  # FCN16s 通用工厂
        'fcn8s': get_fcn8s,  # FCN8s 通用工厂
        'fcn': get_fcn,  # FCN(v2) 通用工厂（ResNet 骨干）
        'psp': get_psp,  # PSPNet 通用工厂
        'deeplabv3': get_deeplabv3,  # DeepLabV3 通用工厂
        'deeplabv3_plus': get_deeplabv3_plus,  # DeepLabV3+ 通用工厂
        'danet': get_danet,  # DANet 通用工厂
        'denseaspp': get_denseaspp,  # DenseASPP 通用工厂
        'bisenet': get_bisenet,  # BiSeNet 通用工厂
        'encnet': get_encnet,  # EncNet 通用工厂
        'dunet': get_dunet,  # DU-Net 通用工厂
        'icnet': get_icnet,  # ICNet 通用工厂
        'enet': get_enet,  # ENet 通用工厂
        'ocnet': get_ocnet,  # OCNet 通用工厂
        'psanet': get_psanet,  # PSANet 通用工厂
        'cgnet': get_cgnet,  # CGNet 通用工厂
        'espnet': get_espnet,  # ESPNet 通用工厂
        'lednet': get_lednet,  # LEDNet 通用工厂
        'dfanet': get_dfanet,  # DFANet 通用工厂
    }  # 算法短名 → 通用工厂映射字典结束
    return models[model](**kwargs)  # 调用对应通用工厂构建并返回模型；【注意】未做键存在性检查，未知算法名会直接 KeyError
