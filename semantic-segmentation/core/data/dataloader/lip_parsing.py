# ============================================================================
# 【文件说明】lip_parsing.py —— LIP（Look Into Person）人体解析数据集
#             加载器：逐行解析 train_id.txt / val_id.txt / test_id.txt
#             划分列表，配对 TrainVal_images 原图与
#             TrainVal_parsing_annotations 解析标注。
# 【数据集】LIP（单人人体解析）：20 类 = 背景 + 19 个身体部位
#           （NUM_CLASS=20）；30462 train / 10000 val / 13673 test；
#           目录结构（root 默认 ../datasets/LIP）：
#             TrainVal_images/train_images、val_images               原图（jpg）
#             TrainVal_parsing_annotations/train_segmentations、
#               val_segmentations                                     标注掩膜（png）
#             Testing_images/testing_images                          测试原图
#             TrainVal_images/{train,val}_id.txt、
#               Testing_images/test_id.txt                            划分列表（每行一个样本 id）
#           标签格式：掩膜像素值 0~19 直接为类别索引（0=背景），无需重映射。
#           【注意】本数据集未注册进 __init__.py 的 datasets 字典，
#           无法通过 get_segmentation_dataset('lip') 使用，需直接 import。
# ============================================================================
"""Look into Person Dataset"""
import os  # 标准库：路径拼接/存在性检查/取文件名
import torch  # 深度学习框架：把掩膜转成 LongTensor
import numpy as np  # 数值库：掩膜转 int32 数组

from PIL import Image  # PIL 图像读取与格式转换
from core.data.dataloader.segbase import SegmentationDataset  # 【注意】绝对导入（其他数据集文件均用相对导入 .segbase），要求以项目根目录为工作路径运行


# LIP 人体解析数据集类：继承基类增广，自实现 train/val/test 三个划分的 id 列表解析
class LIPSegmentation(SegmentationDataset):
    """Look into person parsing dataset """  # 原英文 docstring：LIP 人体解析数据集

    BASE_DIR = 'LIP'  # 数据集子目录名（【注意】定义后未使用，各目录在 __init__ 中按分支自行拼接）
    NUM_CLASS = 20  # 类别数：背景 + 19 个身体部位

    # root: 数据集根目录，默认 '../datasets/LIP'
    # split: 'train'/'val'/'test'，决定读取哪个 id 列表与图像目录
    # mode: 'train'/'val'/'test'/'testval'，None 时等于 split；transform: 图像变换；**kwargs: 透传基类的 base_size/crop_size 等参数
    def __init__(self, root='../datasets/LIP', split='train', mode=None, transform=None, **kwargs):
        super(LIPSegmentation, self).__init__(root, split, mode, transform, **kwargs)  # 初始化基类（保存 root/split/mode/transform/增广尺寸）
        _trainval_image_dir = os.path.join(root, 'TrainVal_images')  # 训练+验证原图总目录
        _testing_image_dir = os.path.join(root, 'Testing_images')  # 测试原图总目录
        _trainval_mask_dir = os.path.join(root, 'TrainVal_parsing_annotations')  # 训练+验证标注总目录
        if split == 'train':  # 训练划分
            _image_dir = os.path.join(_trainval_image_dir, 'train_images')  # 训练图像目录
            _mask_dir = os.path.join(_trainval_mask_dir, 'train_segmentations')  # 训练掩膜目录
            _split_f = os.path.join(_trainval_image_dir, 'train_id.txt')  # 训练 id 列表文件
        elif split == 'val':  # 验证划分
            _image_dir = os.path.join(_trainval_image_dir, 'val_images')  # 验证图像目录
            _mask_dir = os.path.join(_trainval_mask_dir, 'val_segmentations')  # 验证掩膜目录
            _split_f = os.path.join(_trainval_image_dir, 'val_id.txt')  # 验证 id 列表文件
        elif split == 'test':  # 测试划分（只有图像、没有标注）
            _image_dir = os.path.join(_testing_image_dir, 'testing_images')  # 测试图像目录
            _split_f = os.path.join(_testing_image_dir, 'test_id.txt')  # 测试 id 列表文件（【注意】与 train/val 不同，放在 Testing_images 目录下）
        else:  # 未知划分名
            raise RuntimeError('Unknown dataset split.')  # 直接报错

        self.images = []  # 图像路径累积列表
        self.masks = []  # 掩膜路径累积列表（test 划分时保持为空）
        with open(os.path.join(_split_f), 'r') as lines:  # 打开划分文件逐行读取（【注意】os.path.join 单参数冗余）
            for line in lines:  # 每行一个样本 id
                _image = os.path.join(_image_dir, line.rstrip('\n') + '.jpg')  # 拼出 jpg 图像路径（【注意】CRLF 行尾文件同样有残留 \r 的隐患）
                assert os.path.isfile(_image)  # 断言图像存在
                self.images.append(_image)  # 收录图像路径
                if split != 'test':  # 非 test 划分才需要掩膜
                    _mask = os.path.join(_mask_dir, line.rstrip('\n') + '.png')  # 拼出同名 png 掩膜路径
                    assert os.path.isfile(_mask)  # 断言掩膜存在
                    self.masks.append(_mask)  # 收录掩膜路径

        if split != 'test':  # 非 test 划分
            assert (len(self.images) == len(self.masks))  # 断言图像与掩膜一一配对
        print('Found {} {} images in the folder {}'.format(len(self.images), split, root))  # 打印找到的样本数（带划分名）

    # 按索引取一条样本：读图 → 按 mode 增广 → 转 Tensor
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')  # 打开图像并统一转 RGB 三通道
        if self.mode == 'test':  # 纯测试模式：无标注，只返回变换后的图像与文件名
            img = self._img_transform(img)  # 先转 ndarray 再送 transform（【注意】与 cityscapes/sbu 直接传 PIL 的写法不一致）
            if self.transform is not None:  # 若提供了外部变换
                img = self.transform(img)  # 应用变换（ToTensor/Normalize）
            return img, os.path.basename(self.images[index])  # 返回 (图像张量, 文件名)
        mask = Image.open(self.masks[index])  # 打开对应解析标注掩膜（要求为单通道/调色板 png，像素值即类别索引）
        # synchronized transform
        if self.mode == 'train':  # 训练模式
            img, mask = self._sync_transform(img, mask)  # 随机增广（翻转/缩放/裁剪/模糊），img 与 mask 同步
        elif self.mode == 'val':  # 验证模式
            img, mask = self._val_sync_transform(img, mask)  # 确定性变换（等比缩放 + 中心裁剪）
        else:  # 其余模式只允许 testval
            assert self.mode == 'testval'  # testval：不做几何增广，原尺寸直接转数组
            img, mask = self._img_transform(img), self._mask_transform(mask)  # 图像转 ndarray，掩膜转 Tensor
        # general resize, normalize and toTensor
        if self.transform is not None:  # 若提供了外部变换
            img = self.transform(img)  # 仅对图像做 ToTensor + Normalize（掩膜不归一化）

        return img, mask, os.path.basename(self.images[index])  # 返回 (图像张量, 标签, 文件名)

    def __len__(self):  # 数据集样本总数
        return len(self.images)  # 以图像列表长度为准

    # 掩膜后处理：int32 数组 → LongTensor（LIP 标签已是 0~19，无需重映射）
    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')  # 掩膜转 int32 数组
        return torch.from_numpy(target).long()  # 转为 torch 长整型张量（交叉熵要求整型标签）

    @property  # property 装饰器
    def classes(self):  # 20 个类别名称元组（0 背景 + 19 个身体部位，与标签值一一对应）
        """Category name."""  # 原英文 docstring：类别名称列表
        return ('background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',  # 0 背景 / 1 帽子 / 2 头发 / 3 手套 / 4 太阳镜 / 5 上衣
                'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',  # 6 连衣裙 / 7 外套 / 8 袜子 / 9 裤子 / 10 连体服 / 11 围巾 / 12 裙子
                'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',  # 13 脸 / 14 左臂 / 15 右臂 / 16 左腿 / 17 右腿 / 18 左鞋
                'rightShoe')  # 19 右鞋


if __name__ == '__main__':  # 直接运行本文件时的自测入口
    dataset = LIPSegmentation(base_size=280, crop_size=256)  # 实例化（LIP 图像较小，基准/裁剪尺寸用 280/256 而非默认 520/480）
