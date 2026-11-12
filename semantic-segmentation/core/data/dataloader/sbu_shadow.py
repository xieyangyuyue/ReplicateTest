# ============================================================================
# 【文件说明】sbu_shadow.py —— SBU Shadow 阴影分割数据集加载器：遍历
#             ShadowImages/ShadowMasks 目录配对“图像-掩膜”，并在
#             _mask_transform 中把掩膜二值化（非零 → 1 = 阴影）。
# 【数据集】SBU Shadow（SBU-ShadowDataset）：2 类（0 背景 / 1 阴影，
#           NUM_CLASS=2）；训练集 SBUTrain4KRecoveredSmall 约 4085 张，
#           测试集 SBU-Test（约 638 张）；
#           目录结构（root 默认 ../datasets/sbu）：
#             SBUTrain4KRecoveredSmall/ShadowImages/*.jpg   训练原图
#             SBUTrain4KRecoveredSmall/ShadowMasks/*.png    训练掩膜（0/255 二值）
#             SBU-Test/ShadowImages、SBU-Test/ShadowMasks   测试图与掩膜
#           标签格式：掩膜 0 = 非阴影、255 = 阴影，加载时统一转为 0/1。
#           【注意】'val' 与 'test' 两个 split 共用 SBU-Test 目录（同一批数据）。
# ============================================================================
"""SBU Shadow  Segmentation Dataset."""
import os  # 标准库：路径拼接/存在性检查/递归遍历/取文件名
import torch  # 深度学习框架：把掩膜转成 LongTensor
import numpy as np  # 数值库：掩膜转 int32 数组

from PIL import Image  # PIL 图像读取与格式转换
from .segbase import SegmentationDataset  # 相对导入分割数据集基类（同步增广逻辑所在）


# SBU 阴影分割数据集类：继承基类增广，自实现目录扫描配对与掩膜二值化
class SBUSegmentation(SegmentationDataset):
    # 类 docstring：SBU 阴影分割数据集简要说明（次行闭合三引号）
    """SBU Shadow Segmentation Dataset
    """
    NUM_CLASS = 2  # 类别数：背景 / 阴影（二分类）

    # root: 数据集根目录，默认 '../datasets/sbu'
    # split: 'train' 走训练目录；'val'/'test' 共用测试目录
    # mode: 'train'/'val'/'test'/'testval'，None 时等于 split；transform: 图像变换；**kwargs: 透传基类参数
    def __init__(self, root='../datasets/sbu', split='train', mode=None, transform=None, **kwargs):
        super(SBUSegmentation, self).__init__(root, split, mode, transform, **kwargs)  # 初始化基类（保存 root/split/mode/transform/增广尺寸）
        assert os.path.exists(self.root)  # 断言根目录存在（无自定义错误提示信息）
        self.images, self.masks = _get_sbu_pairs(self.root, self.split)  # 扫描目录得到配对的图像/掩膜路径列表
        assert (len(self.images) == len(self.masks))  # 断言图像与掩膜数量一致（配对成功）
        if len(self.images) == 0:  # 未找到任何图像
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")  # 报错提示数据路径有误

    # 按索引取一条样本：读图 → 按 mode 增广 → 转 Tensor
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')  # 打开图像并统一转 RGB 三通道
        if self.mode == 'test':  # 纯测试模式：无标注，只返回变换后的图像与文件名
            if self.transform is not None:  # 若提供了外部变换
                img = self.transform(img)  # 应用变换（ToTensor/Normalize；【注意】此处直接传 PIL 图像，与 ade/voc/lip 先转 ndarray 的写法不一致）
            return img, os.path.basename(self.images[index])  # 返回 (图像张量, 文件名)
        mask = Image.open(self.masks[index])  # 打开对应二值掩膜
        # synchrosized transform
        if self.mode == 'train':  # 训练模式
            img, mask = self._sync_transform(img, mask)  # 随机增广（翻转/缩放/裁剪/模糊），img 与 mask 同步
        elif self.mode == 'val':  # 验证模式
            img, mask = self._val_sync_transform(img, mask)  # 确定性变换（等比缩放 + 中心裁剪）
        else:  # 其余模式只允许 testval
            assert self.mode == 'testval'  # testval：不做几何增广，原尺寸直接转数组
            img, mask = self._img_transform(img), self._mask_transform(mask)  # 图像转 ndarray，掩膜转 Tensor 并二值化
        # general resize, normalize and toTensor
        if self.transform is not None:  # 若提供了外部变换
            img = self.transform(img)  # 仅对图像做 ToTensor + Normalize（掩膜不归一化）
        return img, mask, os.path.basename(self.images[index])  # 返回 (图像张量, 标签, 文件名)

    # 掩膜后处理：任意非零值（如 255）统一压为 1，再转 LongTensor
    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')  # 掩膜转 int32 数组
        target[target > 0] = 1  # 非零像素（阴影，原始多为 255）统一置 1，背景保持 0
        return torch.from_numpy(target).long()  # 转为 torch 长整型张量（交叉熵要求整型标签）

    def __len__(self):  # 数据集样本总数
        return len(self.images)  # 以图像列表长度为准

    @property  # property 装饰器
    def pred_offset(self):  # 预测偏移：模型输出 0/1 与二值化后的存储标签一致
        return 0  # 无偏移


# 扫描目录构造 SBU 的 (图像路径, 掩膜路径) 配对列表（模块级辅助函数）
# folder: 数据集根目录；split: 'train' 或 'val'/'test'
def _get_sbu_pairs(folder, split='train'):
    # 内部函数：递归遍历 img_folder 下所有 jpg，在 mask_folder 下找同名 png 掩膜
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []  # 图像路径累积列表
        mask_paths = []  # 掩膜路径累积列表（与 img_paths 按下标一一对应）
        for root, _, files in os.walk(img_folder):  # 递归遍历图像目录
            print(root)  # 【注意】打印每个遍历到的目录路径，疑似调试残留代码，数据量大时输出冗余
            for filename in files:  # 遍历当前目录下的文件
                if filename.endswith('.jpg'):  # 只处理 jpg 图像
                    imgpath = os.path.join(root, filename)  # 拼出图像完整路径
                    maskname = filename.replace('.jpg', '.png')  # 掩膜文件名 = 同名 .png
                    maskpath = os.path.join(mask_folder, maskname)  # 拼出掩膜完整路径
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):  # 图像与掩膜都存在才收录
                        img_paths.append(imgpath)  # 收录图像路径
                        mask_paths.append(maskpath)  # 收录掩膜路径
                    else:  # 缺图或缺掩膜
                        print('cannot find the mask or image:', imgpath, maskpath)  # 打印缺失信息（仅警告，不中断）
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))  # 打印本目录配对成功的图像数
        return img_paths, mask_paths  # 返回配对结果

    if split == 'train':  # 训练划分
        img_folder = os.path.join(folder, 'SBUTrain4KRecoveredSmall/ShadowImages')  # 训练图像目录
        mask_folder = os.path.join(folder, 'SBUTrain4KRecoveredSmall/ShadowMasks')  # 训练掩膜目录
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)  # 配对训练集
    else:  # 其余划分
        assert split in ('val', 'test')  # 只允许 val/test（【注意】两者共用同一测试目录）
        img_folder = os.path.join(folder, 'SBU-Test/ShadowImages')  # 测试图像目录
        mask_folder = os.path.join(folder, 'SBU-Test/ShadowMasks')  # 测试掩膜目录
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)  # 配对测试集
    return img_paths, mask_paths  # 返回配对结果


if __name__ == '__main__':  # 直接运行本文件时的自测入口
    dataset = SBUSegmentation(base_size=280, crop_size=256)  # 实例化（SBU 图像较小，基准/裁剪尺寸用 280/256 而非默认 520/480）
