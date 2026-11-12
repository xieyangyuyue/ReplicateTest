# ============================================================================
# 【文件说明】pascal_aug.py —— PASCAL VOC 增强版（SBD 边界增强）分割数据集
#             加载器：与 pascal_voc.py 的区别在于使用 Semantic Boundaries
#             Dataset（SBD）的 .mat（MATLAB）格式标注、训练样本量更大；
#             掩膜经 scipy.io.loadmat 读取 GTcls.Segmentation 字段转为 PIL。
# 【数据集】VOCaug（root/VOCaug/dataset/）：21 类（与 VOC 一致，
#           NUM_CLASS=21）；SBD trainval 共 10582 张训练图；
#           目录结构（root 默认 ../datasets/voc）：
#             VOCaug/dataset/img/*.jpg        原图
#             VOCaug/dataset/cls/*.mat        MATLAB 类别掩膜（GTcls.Segmentation 字段）
#             VOCaug/dataset/trainval.txt     trainval 划分列表（split='train' 时读取）
#             VOCaug/dataset/val.txt          val 划分列表
#           标签格式：0 = 背景，1~20 = 物体类别（SBD 标注无 255 忽略值）。
#           【注意】train 读取的是 trainval.txt（SBD 全量 10582 张），若后续
#           在 VOC val 上评估，可能存在训练/验证图像重叠。
# ============================================================================
"""Pascal Augmented VOC Semantic Segmentation Dataset."""
import os  # 标准库：路径拼接/存在性检查
import torch  # 深度学习框架：把掩膜转成 LongTensor
import scipy.io as sio  # SciPy 的 MATLAB .mat 文件读写模块（用于读取 SBD 标注）
import numpy as np  # 数值库：掩膜转 int32 数组

from PIL import Image  # PIL 图像读取、由 ndarray 构建图像
from .segbase import SegmentationDataset  # 相对导入分割数据集基类（同步增广逻辑所在）


# VOC 增强版（SBD）分割数据集类：继承基类增广，自实现 .mat 标注读取
class VOCAugSegmentation(SegmentationDataset):
    # 类 docstring：VOC+SBD 增强分割数据集说明（数据路径/划分/变换与用法示例）
    """Pascal VOC Augmented Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/voc'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = VOCAugSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'VOCaug/dataset/'  # 数据集子目录（【注意】末尾带斜杠，与 os.path.join 拼接无害但与其他文件风格不一致）
    NUM_CLASS = 21  # 类别数：背景 + 20 个物体类（与 VOC 相同）

    # root: 数据集根目录，默认 '../datasets/voc'
    # split: 'train'（读 trainval.txt）或 'val'（读 val.txt），其余直接报错
    # mode: 'train'/'val'，None 时等于 split；transform: 图像变换；**kwargs: 透传基类参数
    def __init__(self, root='../datasets/voc', split='train', mode=None, transform=None, **kwargs):
        super(VOCAugSegmentation, self).__init__(root, split, mode, transform, **kwargs)  # 初始化基类（保存 root/split/mode/transform/增广尺寸）
        # train/val/test splits are pre-cut
        _voc_root = os.path.join(root, self.BASE_DIR)  # VOCaug 数据根目录
        _mask_dir = os.path.join(_voc_root, 'cls')  # .mat 标注目录
        _image_dir = os.path.join(_voc_root, 'img')  # 原图目录
        if split == 'train':  # 训练划分
            _split_f = os.path.join(_voc_root, 'trainval.txt')  # 读 SBD trainval 全量列表（10582 张）
        elif split == 'val':  # 验证划分
            _split_f = os.path.join(_voc_root, 'val.txt')  # 读验证列表
        else:  # 未知划分名
            raise RuntimeError('Unknown dataset split: {}'.format(split))  # 直接报错

        self.images = []  # 图像路径累积列表
        self.masks = []  # 掩膜路径累积列表
        with open(os.path.join(_split_f), "r") as lines:  # 打开划分文件逐行读取（【注意】os.path.join 单参数冗余；CRLF 行尾同样有残留 \r 隐患）
            for line in lines:  # 每行一个样本 id
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")  # 拼出 jpg 图像路径
                assert os.path.isfile(_image)  # 断言图像存在
                self.images.append(_image)  # 收录图像路径
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".mat")  # 拼出同名 .mat 标注路径
                assert os.path.isfile(_mask)  # 断言标注存在
                self.masks.append(_mask)  # 收录标注路径

        assert (len(self.images) == len(self.masks))  # 断言图像与掩膜一一配对
        print('Found {} images in the folder {}'.format(len(self.images), _voc_root))  # 打印找到的样本数

    # 按索引取一条样本：读图 → 读 .mat 标注 → 按 mode 增广 → 转 Tensor
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')  # 打开图像并统一转 RGB 三通道
        target = self._load_mat(self.masks[index])  # 从 .mat 读取类别掩膜并转成 PIL 图像
        # synchrosized transform
        if self.mode == 'train':  # 训练模式
            img, target = self._sync_transform(img, target)  # 随机增广（翻转/缩放/裁剪/模糊；内部末尾会调用 _mask_transform 转 Tensor）
        elif self.mode == 'val':  # 验证模式
            img, target = self._val_sync_transform(img, target)  # 确定性变换（等比缩放 + 中心裁剪）
        else:  # 其余模式
            raise RuntimeError('unknown mode for dataloader: {}'.format(self.mode))  # 【注意】本数据集不支持 test/testval 模式（其他数据集支持 testval），直接报错
        # general resize, normalize and toTensor
        if self.transform is not None:  # 若提供了外部变换
            img = self.transform(img)  # 仅对图像做 ToTensor + ImageNet Normalize（掩膜不归一化）
        return img, target, os.path.basename(self.images[index])  # 返回 (图像张量, 标签, 文件名)

    # 掩膜后处理：数组 → LongTensor（SBD 标签已是 0~20，无需重映射）
    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))  # 转 int32 后再转 LongTensor

    # 读取 SBD 的 MATLAB .mat 标注文件，返回 PIL 灰度掩膜
    def _load_mat(self, filename):
        mat = sio.loadmat(filename, mat_dtype=True, squeeze_me=True, struct_as_record=False)  # 读 .mat：保留原始数据类型、压缩单维度、结构体转为可属性访问的对象
        mask = mat['GTcls'].Segmentation  # 取 GTcls 结构体的 Segmentation 字段（H×W 类别矩阵）
        return Image.fromarray(mask)  # 由 ndarray 构建 PIL 图像（供基类增广使用）

    def __len__(self):  # 数据集样本总数
        return len(self.images)  # 以图像列表长度为准

    @property  # property 装饰器
    def classes(self):  # 21 个类别名称元组（与 VOC 相同：0 背景 + 20 物体）
        """Category names."""  # 原英文 docstring：类别名称列表
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',  # 0 背景 / 1 飞机 / 2 自行车 / 3 鸟 / 4 船 / 5 瓶子
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',  # 6 公交 / 7 汽车 / 8 猫 / 9 椅子 / 10 牛 / 11 餐桌 / 12 狗 / 13 马
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',  # 14 摩托 / 15 人 / 16 盆栽 / 17 羊 / 18 沙发 / 19 火车
                'tv')  # 20 电视


if __name__ == '__main__':  # 直接运行本文件时的自测入口
    dataset = VOCAugSegmentation()  # 用默认参数实例化（root='../datasets/voc', split='train'）
