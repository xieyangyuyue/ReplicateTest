# ============================================================================
# 【文件说明】pascal_voc.py —— PASCAL VOC 2012 语义分割数据集加载器：
#             逐行解析 ImageSets/Segmentation 下的 train.txt/val.txt/test.txt
#             划分文件，配对 JPEGImages/*.jpg 与 SegmentationClass/*.png，
#             并在 _mask_transform 中把忽略值 255 映射为 -1。
# 【数据集】PASCAL VOC 2012（VOCdevkit/VOC2012）：21 类 = 背景 + 20 物体
#           （NUM_CLASS=21）；分割标注 1464 train / 1449 val；
#           目录结构（root 默认 ../datasets/voc）：
#             VOC2012/JPEGImages/*.jpg                             原图
#             VOC2012/SegmentationClass/*.png                      标注掩膜（调色板 PNG）
#             VOC2012/ImageSets/Segmentation/{train,val,test}.txt  划分列表（每行一个样本 id）
#           标签格式：0 = 背景，1~20 = 物体类别，255 = 忽略区域（转 -1）；
#           test 划分只有图像、没有掩膜。
# ============================================================================
"""Pascal VOC Semantic Segmentation Dataset."""
import os  # 标准库：路径拼接/存在性检查/取文件名
import torch  # 深度学习框架：把掩膜转成 LongTensor
import numpy as np  # 数值库：掩膜转 int32 数组

from PIL import Image  # PIL 图像读取与格式转换
from .segbase import SegmentationDataset  # 相对导入分割数据集基类（同步增广逻辑所在）


# PASCAL VOC 2012 分割数据集类：继承基类增广，自实现 txt 划分解析与 255→-1 标签清理
class VOCSegmentation(SegmentationDataset):
    # 类 docstring：Pascal VOC 语义分割数据集说明（数据路径/划分/变换与用法示例）
    """Pascal VOC Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/VOCdevkit'
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
    >>> trainset = VOCSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'VOC2012'  # 数据集子目录名（root 之下）
    NUM_CLASS = 21  # 类别数：背景 + 20 个物体类

    # root: 数据集根目录，默认 '../datasets/voc'
    # split: 'train'/'val'/'test'，决定读取哪个 txt 划分文件
    # mode: 'train'/'val'/'test'/'testval'，None 时等于 split；transform: 图像变换；**kwargs: 透传基类参数
    def __init__(self, root='../datasets/voc', split='train', mode=None, transform=None, **kwargs):
        super(VOCSegmentation, self).__init__(root, split, mode, transform, **kwargs)  # 初始化基类（保存 root/split/mode/transform/增广尺寸）
        _voc_root = os.path.join(root, self.BASE_DIR)  # VOC2012 数据根目录
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')  # 掩膜目录
        _image_dir = os.path.join(_voc_root, 'JPEGImages')  # 原图目录
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')  # 划分文件目录（train/val/test 已预先切分好）
        if split == 'train':  # 训练划分
            _split_f = os.path.join(_splits_dir, 'train.txt')  # 训练样本 id 列表文件
        elif split == 'val':  # 验证划分
            _split_f = os.path.join(_splits_dir, 'val.txt')  # 验证样本 id 列表文件
        elif split == 'test':  # 测试划分
            _split_f = os.path.join(_splits_dir, 'test.txt')  # 测试样本 id 列表文件
        else:  # 未知划分名
            raise RuntimeError('Unknown dataset split.')  # 直接报错

        self.images = []  # 图像路径累积列表
        self.masks = []  # 掩膜路径累积列表（test 划分时保持为空）
        with open(os.path.join(_split_f), "r") as lines:  # 打开划分文件准备逐行读取（【注意】os.path.join 只传一个参数属于冗余写法）
            for line in lines:  # 每行一个样本 id
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")  # 拼出 jpg 图像路径（【注意】只去除 \n，若文件为 CRLF 行尾会残留 \r 导致路径错误）
                assert os.path.isfile(_image)  # 断言图像存在
                self.images.append(_image)  # 收录图像路径
                if split != 'test':  # 非 test 划分才需要掩膜
                    _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")  # 拼出同名 png 掩膜路径
                    assert os.path.isfile(_mask)  # 断言掩膜存在
                    self.masks.append(_mask)  # 收录掩膜路径

        if split != 'test':  # 非 test 划分
            assert (len(self.images) == len(self.masks))  # 断言图像与掩膜一一配对
        print('Found {} images in the folder {}'.format(len(self.images), _voc_root))  # 打印找到的样本数

    # 按索引取一条样本：读图 → 按 mode 增广 → 转 Tensor
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')  # 打开图像并统一转 RGB 三通道
        if self.mode == 'test':  # 纯测试模式：无标注，只返回变换后的图像与文件名
            img = self._img_transform(img)  # 先转 ndarray 再送 transform（【注意】与 cityscapes/sbu 直接传 PIL 的写法不一致）
            if self.transform is not None:  # 若提供了外部变换
                img = self.transform(img)  # 应用变换（ToTensor/Normalize）
            return img, os.path.basename(self.images[index])  # 返回 (图像张量, 文件名)
        mask = Image.open(self.masks[index])  # 打开对应标注掩膜（调色板 PNG，像素值即类别）
        # synchronized transform
        if self.mode == 'train':  # 训练模式
            img, mask = self._sync_transform(img, mask)  # 随机增广（翻转/缩放/裁剪/模糊），img 与 mask 同步
        elif self.mode == 'val':  # 验证模式
            img, mask = self._val_sync_transform(img, mask)  # 确定性变换（等比缩放 + 中心裁剪）
        else:  # 其余模式只允许 testval
            assert self.mode == 'testval'  # testval：不做几何增广，原尺寸直接转数组（用于全图评估）
            img, mask = self._img_transform(img), self._mask_transform(mask)  # 图像转 ndarray，掩膜转 Tensor 并把 255 映射为 -1
        # general resize, normalize and totensor
        if self.transform is not None:  # 若提供了外部变换
            img = self.transform(img)  # 仅对图像做 ToTensor + ImageNet Normalize（掩膜不归一化）

        return img, mask, os.path.basename(self.images[index])  # 返回 (图像张量, 标签, 文件名)

    def __len__(self):  # 数据集样本总数
        return len(self.images)  # 以图像列表长度为准

    # 掩膜后处理：PIL → int32 数组，忽略值 255 → -1，再转 LongTensor
    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')  # 掩膜转 int32 数组
        target[target == 255] = -1  # 官方忽略标签 255 统一改为 -1（忽略类，不参与损失计算）
        return torch.from_numpy(target).long()  # 转为 torch 长整型张量（交叉熵要求整型标签）

    @property  # property 装饰器
    def classes(self):  # 21 个类别名称元组（0 背景 + 20 物体，与标签值一一对应）
        """Category names."""  # 原英文 docstring：类别名称列表
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',  # 0 背景 / 1 飞机 / 2 自行车 / 3 鸟 / 4 船 / 5 瓶子
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',  # 6 公交 / 7 汽车 / 8 猫 / 9 椅子 / 10 牛 / 11 餐桌 / 12 狗 / 13 马
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',  # 14 摩托 / 15 人 / 16 盆栽 / 17 羊 / 18 沙发 / 19 火车
                'tv')  # 20 电视


if __name__ == '__main__':  # 直接运行本文件时的自测入口
    dataset = VOCSegmentation()  # 用默认参数实例化（root='../datasets/voc', split='train'）
