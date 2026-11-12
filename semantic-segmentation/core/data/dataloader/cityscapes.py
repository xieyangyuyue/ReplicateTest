# ============================================================================
# 【文件说明】cityscapes.py —— Cityscapes 街景语义分割数据集加载器：
#             扫描 leftImg8bit/gtFine 目录构造“图像-掩膜”路径对，
#             把官方 labelId(0~33) 重映射为 19 个训练类别索引(0~18)，
#             并按 train/val 模式执行基类增广流程。
# 【数据集】Cityscapes：欧洲城市车载街景，图像 2048×1024；
#           官方 34 个类别中仅 19 个“有效类”参与训练（NUM_CLASS=19）；
#           精细标注 2975 train + 500 val（trainval 合计 3475）；
#           目录结构（root 默认 ../datasets/citys）：
#             leftImg8bit/{train,val}/<城市>/xxx_leftImg8bit.png   原图
#             gtFine/{train,val}/<城市>/xxx_gtFine_labelIds.png    标签掩膜
#           标签格式：掩膜像素值为官方 labelId，_class_to_index 映射为
#           0~18 连续训练类别，无效类映射为 -1（忽略，不参与损失）。
#           【注意】docstring 声称支持 split='test'，但 _get_city_pairs
#           实际只支持 'train'/'val'/'trainval'，传 'test' 会触发断言失败。
# ============================================================================
"""Prepare Cityscapes dataset"""
import os  # 标准库：路径拼接/存在性检查/递归遍历/取文件名
import torch  # 深度学习框架：把掩膜转成 LongTensor
import numpy as np  # 数值库：类别映射用 np.array/np.digitize/np.unique

from PIL import Image  # PIL 图像读取与格式转换
from .segbase import SegmentationDataset  # 相对导入分割数据集基类（同步增广逻辑所在）


# Cityscapes 街景分割数据集类：继承基类增广，自实现目录配对与 labelId → 训练 id 的重映射
class CitySegmentation(SegmentationDataset):
    # 类 docstring：Cityscapes 语义分割数据集说明（数据路径/划分/变换参数与用法示例）
    """Cityscapes Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/citys'
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
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'cityscapes'  # 数据集子目录名（【注意】下面第 42 行的拼接语句被注释掉，该常量实际未被使用）
    NUM_CLASS = 19  # 类别数：34 个官方类中的 19 个有效类

    # root: 数据集根目录，默认 '../datasets/citys'
    # split: 'train'/'val'/'trainval'（【注意】docstring 写了 'test'，代码并不支持，见 _get_city_pairs）
    # mode: 'train'/'val'/'test'/'testval'，None 时等于 split
    # transform: 图像变换（ToTensor + Normalize）；**kwargs: 透传基类的 base_size/crop_size 等参数
    def __init__(self, root='../datasets/citys', split='train', mode=None, transform=None, **kwargs):
        super(CitySegmentation, self).__init__(root, split, mode, transform, **kwargs)  # 初始化基类：保存 root/split/mode/transform 及增广尺寸
        # self.root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(self.root), "Please setup the dataset using ../datasets/cityscapes.py"  # 断言根目录存在，否则提示先用下载脚本准备数据集
        self.images, self.mask_paths = _get_city_pairs(self.root, self.split)  # 扫描目录得到配对的图像/掩膜路径列表
        assert (len(self.images) == len(self.mask_paths))  # 断言图像与掩膜数量一致（配对成功）
        if len(self.images) == 0:  # 一张图都没找到
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")  # 报错提示数据路径配置有误
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,  # 有效类的官方 labelId 前半段（道路/人行道/建筑/墙/栅栏/杆/交通灯/交通牌/植被/地形）
                              23, 24, 25, 26, 27, 28, 31, 32, 33]  # 有效类的官方 labelId 后半段（天空/人/骑手/汽车/卡车/公交/火车/摩托/自行车）
        self._key = np.array([-1, -1, -1, -1, -1, -1,  # 查找表：下标 i 对应 labelId=i-1；labelId -1~4 → -1（忽略）
                              -1, -1, 0, 1, -1, -1,  # labelId 5~10：仅 7→0(道路)、8→1(人行道)
                              2, 3, 4, -1, -1, -1,  # labelId 11~16：11→2(建筑)、12→3(墙)、13→4(栅栏)
                              5, -1, 6, 7, 8, 9,  # labelId 17~22：17→5(杆)、19→6(交通灯)、20→7(交通牌)、21→8(植被)、22→9(地形)
                              10, 11, 12, 13, 14, 15,  # labelId 23~28：23→10(天空)、24→11(人)、25→12(骑手)、26→13(汽车)、27→14(卡车)、28→15(公交)
                              -1, -1, 16, 17, 18])  # labelId 29~34：31→16(火车)、32→17(摩托)、33→18(自行车)
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')  # 辅助数组 [-1,0,1,…,34]，配合 np.digitize 把 labelId 转成 _key 的下标（labelId+1）

    # 把官方 labelId 掩膜重映射为 0~18 连续训练类别索引
    # mask: numpy 数组形式的 labelId 掩膜；返回同形状的类别索引掩膜（无效类为 -1）
    def _class_to_index(self, mask):
        # assert the value
        values = np.unique(mask)  # 取掩膜中出现过的所有标签值
        for value in values:  # 逐一检查
            assert (value in self._mapping)  # 断言值 ∈ [-1,34]，超出范围说明数据异常
        index = np.digitize(mask.ravel(), self._mapping, right=True)  # digitize 把每个 labelId 映射为 _key 下标（labelId+1；-1→0）
        return self._key[index].reshape(mask.shape)  # 查表取训练 id 并恢复成原掩膜形状（无效类变 -1）

    # 按索引取一条样本：读图 → 按 mode 增广 → 转 Tensor
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')  # 打开图像并统一转 RGB 三通道
        if self.mode == 'test':  # 纯测试模式：无标注，只返回变换后的图像与文件名
            if self.transform is not None:  # 若提供了外部变换
                img = self.transform(img)  # 应用变换（ToTensor/Normalize；【注意】此处直接传 PIL 图像，与 ade/voc/lip 先转 ndarray 的写法不一致）
            return img, os.path.basename(self.images[index])  # 返回 (图像张量, 文件名)
        mask = Image.open(self.mask_paths[index])  # 打开对应标签掩膜（gtFine labelIds PNG）
        # synchrosized transform
        if self.mode == 'train':  # 训练模式
            img, mask = self._sync_transform(img, mask)  # 随机增广（翻转/缩放/裁剪/模糊），img 与 mask 同步
        elif self.mode == 'val':  # 验证模式
            img, mask = self._val_sync_transform(img, mask)  # 确定性变换（等比缩放 + 中心裁剪）
        else:  # 其余模式只允许 testval
            assert self.mode == 'testval'  # testval：不做任何几何增广，原尺寸直接转数组（用于全图评估）
            img, mask = self._img_transform(img), self._mask_transform(mask)  # 图像转 ndarray，掩膜走 _mask_transform 完成类别重映射
        # general resize, normalize and toTensor
        if self.transform is not None:  # 若提供了外部变换
            img = self.transform(img)  # 仅对图像做 ToTensor + ImageNet Normalize（掩膜不归一化）
        return img, mask, os.path.basename(self.images[index])  # 返回 (图像张量, 标签, 文件名)

    # 掩膜后处理：重映射 labelId → 训练类别索引，并转 LongTensor
    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))  # PIL 掩膜转 int32 数组后做类别重映射（19 类 + 忽略 -1）
        return torch.LongTensor(np.array(target).astype('int32'))  # 转为 torch.LongTensor（交叉熵损失要求整型标签）

    def __len__(self):  # 数据集样本总数
        return len(self.images)  # 以图像列表长度为准

    @property  # property 装饰器：以 dataset.pred_offset 形式访问
    def pred_offset(self):  # 预测偏移：模型输出 id + offset = 原数据集标签 id
        return 0  # Cityscapes 训练 id 与重映射规则一一对应，无需偏移


# 扫描目录构造 Cityscapes 的 (图像路径, 掩膜路径) 配对列表（模块级辅助函数）
# folder: 数据集根目录；split: 'train'/'val'/'trainval'
def _get_city_pairs(folder, split='train'):
    # 内部函数：递归遍历 img_folder 下所有 png，在 mask_folder/<同名城市子目录>/ 下找对应掩膜
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []  # 图像路径累积列表
        mask_paths = []  # 掩膜路径累积列表（与 img_paths 按下标一一对应）
        for root, _, files in os.walk(img_folder):  # 递归遍历（leftImg8bit 按城市分子目录存放）
            for filename in files:  # 遍历当前目录下的文件
                if filename.endswith('.png'):  # 只处理 png 图像
                    imgpath = os.path.join(root, filename)  # 拼出图像完整路径
                    foldername = os.path.basename(os.path.dirname(imgpath))  # 取图像所在的城市名（如 frankfurt）
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')  # 按命名规则换算掩膜文件名
                    maskpath = os.path.join(mask_folder, foldername, maskname)  # 拼出掩膜完整路径
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):  # 图像与掩膜都存在才收录
                        img_paths.append(imgpath)  # 收录图像路径
                        mask_paths.append(maskpath)  # 收录掩膜路径
                    else:  # 缺图或缺标注
                        print('cannot find the mask or image:', imgpath, maskpath)  # 打印缺失信息（仅警告，不中断）
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))  # 打印本目录配对成功的图像数
        return img_paths, mask_paths  # 返回配对结果

    if split in ('train', 'val'):  # train 或 val：只扫对应子目录
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)  # 图像目录 leftImg8bit/train 或 leftImg8bit/val
        mask_folder = os.path.join(folder, 'gtFine/' + split)  # 掩膜目录 gtFine/train 或 gtFine/val
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)  # 配对
        return img_paths, mask_paths  # 直接返回
    else:  # 其余情况只允许 trainval（【注意】docstring 里的 'test' 会走到这里并触发断言失败）
        assert split == 'trainval'  # 断言为 trainval
        print('trainval set')  # 提示正在合并训练集 + 验证集
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')  # 训练图像目录
        train_mask_folder = os.path.join(folder, 'gtFine/train')  # 训练掩膜目录
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')  # 验证图像目录
        val_mask_folder = os.path.join(folder, 'gtFine/val')  # 验证掩膜目录
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)  # 配对训练集
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)  # 配对验证集
        img_paths = train_img_paths + val_img_paths  # 拼接图像路径
        mask_paths = train_mask_paths + val_mask_paths  # 拼接掩膜路径
    return img_paths, mask_paths  # 返回合并后的配对结果


if __name__ == '__main__':  # 直接运行本文件时的自测入口
    dataset = CitySegmentation()  # 用默认参数实例化（root='../datasets/citys', split='train'）
