# ============================================================================
# 【文件说明】visualize.py —— 语义分割预测结果的可视化与指标打印工具。
# 核心逻辑：
#   1) print_iou：打印 mIoU / 去背景 mIoU / 平均像素精度（逐类 IoU 的
#      打印行已被原作者注释掉，故只输出均值行）；
#   2) set_img_color / show_prediction：按调色板把类别标签叠加着色到
#      原图（背景类不上色、255 忽略区可涂白）；
#   3) show_colorful_images / save_colorful_images：以标签值直接索引
#      预生成调色板数组得到 RGB 彩图，用于弹窗显示/落盘保存；
#   4) get_color_pallete：把单通道预测标签转成带调色板的 PIL P 模式
#      伪彩图——pascal_voc/pascal_aug 先把 -1（边界类）恢复成 255（忽略
#      标签），用 VOC 位运算生成的 256 色调色板；ade20k 标签整体 +1
#      （模型输出 0 起，而数据集标签 0 为背景，+1 对齐后调色板 0 号色
#      即黑色背景）；citys 直接套 19 类 Cityscapes 调色板。
# 数据集-颜色映射的建立方式：
#   VOC——_getvocpallete 用位运算从类别 id 生成颜色（见函数内注释），
#   共 256 条保证任意 0~255 标签（含 255 忽略色）都可着色；
#   ADE20K——硬编码官方 151 色（背景 + 150 类）；
#   Cityscapes——硬编码官方 19 类 trainId 颜色。
# ============================================================================
import os  # 拼接保存路径、创建输出目录
import numpy as np  # 数组运算：标签布尔筛选、nanmean 忽略 NaN 求均值
from PIL import Image  # 图像读写，以及 P 模式（调色板）伪彩着色

__all__ = ['get_color_pallete', 'print_iou', 'set_img_color',  # 模块对外导出的接口清单（首行）
           'show_prediction', 'show_colorful_images', 'save_colorful_images']  # （续）预测显示与彩图保存接口


# 参数：iu——各类 IoU 的一维数组；mean_pixel_acc——已算好的平均像素精度(0~1)；
#      class_names——类别名列表（None 则用序号占位）；show_no_back——是否额外显示去背景 mIoU
def print_iou(iu, mean_pixel_acc, class_names=None, show_no_back=False):
    n = iu.size  # 类别总数（numpy 数组元素个数）
    lines = []  # 待打印的文本行缓冲
    for i in range(n):  # 遍历每个类别
        if class_names is None:  # 未提供类别名
            cls = 'Class %d:' % (i + 1)  # 用序号占位（类别号从 1 开始显示）
        else:  # 提供了类别名
            cls = '%d %s' % (i + 1, class_names[i])  # 序号 + 类别名
        # lines.append('%-8s: %.3f%%' % (cls, iu[i] * 100))
        # 【注意】上一行逐类 IoU 打印被注释掉：导致 cls 变量计算后未被使用（死存储），
        #        且 lines 里没有任何逐类信息，本函数实际只输出均值行
    mean_IU = np.nanmean(iu)  # 全类 mIoU：nanmean 自动忽略 NaN（该类在验证集未出现时为 NaN）
    mean_IU_no_back = np.nanmean(iu[1:])  # 去背景 mIoU：跳过下标 0 的背景类再求均值
    if show_no_back:  # 需要同时显示去背景指标
        lines.append('mean_IU: %.3f%% || mean_IU_no_back: %.3f%% || mean_pixel_acc: %.3f%%' % (  # 拼接三指标输出行（*100 转百分比）
            mean_IU * 100, mean_IU_no_back * 100, mean_pixel_acc * 100))  # 三指标拼一行（*100 转百分比）
    else:  # 只显示常规指标
        lines.append('mean_IU: %.3f%% || mean_pixel_acc: %.3f%%' % (mean_IU * 100, mean_pixel_acc * 100))  # mIoU 与像素精度
    lines.append('=================================================')  # 结尾分隔线，美化输出
    line = "\n".join(lines)  # 多行文本拼接为整体

    print(line)  # 打印到控制台


# 参数：img——待上色图（H,W,3 数组，会被就地修改）；label——同类形状的标签图；
#      colors——调色板（第 i 项为类别 i 的 RGB）；background——背景类下标（跳过不上色）；
#      show255——是否把标签 255（忽略/边界区）涂成白色
def set_img_color(img, label, colors, background=0, show255=False):
    for i in range(len(colors)):  # 遍历每个类别
        if i != background:  # 跳过背景类：背景保持原图像素，形成“类别区覆盖、背景透出”的效果
            img[np.where(label == i)] = colors[i]  # 标签等于 i 的全部像素一次性替换为类别 i 的颜色
    if show255:  # 需要显示 255 忽略区
        img[np.where(label == 255)] = 255  # 标签 255 的像素置为纯白（R=G=B=255）

    return img  # 返回上色结果（注意 img 本身也已被就地修改）


# 参数：img——原图（PIL Image 或 ndarray）；pred——预测标签图；colors——调色板；background——背景类下标
def show_prediction(img, pred, colors, background=0):
    im = np.array(img, np.uint8)  # 原图转为 uint8 ndarray（供后续按标签覆写颜色）
    set_img_color(im, pred, colors, background)  # 就地按调色板给各类区域着色
    out = np.array(im)  # 再拷贝一份作为返回值，避免与输入共享内存

    return out  # 返回着色后的 ndarray


# 参数：prediction——单张标签图（含多余维度，内部 squeeze）；palettes——预生成的调色板数组（标签值作下标取 RGB 行）
def show_colorful_images(prediction, palettes):
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])  # 标签转 uint8、squeeze 掉多余维度后索引调色板得到 H,W,3 彩图，再转 PIL
    im.show()  # 调用系统看图工具弹出显示（依赖桌面环境，服务器上不可用）


# 参数：prediction——预测标签（形状 [B,H,W,C]，内部 squeeze）；filename——保存文件名；
#      output_dir——输出目录；palettes——预生成的调色板数组
def save_colorful_images(prediction, filename, output_dir, palettes):
    # docstring：说明 prediction 参数的形状为 [B, H, W, C]
    '''
    :param prediction: [B, H, W, C]
    '''
    im = Image.fromarray(palettes[prediction.astype('uint8').squeeze()])  # 标签转 uint8、squeeze 掉 batch/通道维，索引调色板映射成 RGB 彩图
    fn = os.path.join(output_dir, filename)  # 拼出完整保存路径
    out_dir = os.path.split(fn)[0]  # 取路径中的目录部分
    if not os.path.exists(out_dir):  # 目录不存在
        os.mkdir(out_dir)  # 【注意】用 mkdir 而非 makedirs：若中间目录也缺失会抛 FileNotFoundError
    im.save(fn)  # 按扩展名保存为图像文件


# 参数：npimg——单通道预测标签数组（H,W 或 H,W,1，会被就地修改）；dataset——数据集名，决定边界恢复与调色板选择
def get_color_pallete(npimg, dataset='pascal_voc'):
    # docstring：可视化说明（npimg 形状、dataset 取值与返回带调色板的 PIL 图）
    """Visualize image.

    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W, 1`.
    dataset : str, default: 'pascal_voc'
        The dataset that model pretrained on. ('pascal_voc', 'ade20k')
    Returns
    -------
    out_img : PIL.Image
        Image with color pallete
    """
    # recovery boundary
    # 边界恢复：pascal_aug 数据集把物体边界标为 -1 类，模型可能预测出 -1，
    # 需映射回 255（VOC 约定的忽略标签，调色板中为白灰色）
    if dataset in ('pascal_voc', 'pascal_aug'):  # VOC 及其增广版数据集
        npimg[npimg == -1] = 255  # 就地把所有 -1 像素改写为 255
    # put colormap
    if dataset == 'ade20k':  # ADE20K 分支
        npimg = npimg + 1  # 标签整体 +1：模型输出 0~149（0 号起是真实类别），而数据集标签 0 为背景，+1 对齐后 0 留给背景（黑色）
        out_img = Image.fromarray(npimg.astype('uint8'))  # 标签转 uint8 灰度 PIL 图（L 模式）
        out_img.putpalette(adepallete)  # 挂载 ADE20K 调色板（151 色）→ 变为 P 模式伪彩图
        return out_img  # 返回着色后的 PIL 图
    elif dataset == 'citys':  # Cityscapes 分支
        out_img = Image.fromarray(npimg.astype('uint8'))  # 标签转 uint8 灰度 PIL 图
        out_img.putpalette(cityspallete)  # 挂载 Cityscapes 19 类调色板
        return out_img  # 返回着色结果
    out_img = Image.fromarray(npimg.astype('uint8'))  # 默认分支（pascal_voc / pascal_aug 及其他）
    out_img.putpalette(vocpallete)  # 挂载 VOC 256 色调色板（255 号为忽略色）
    return out_img  # 返回着色结果


# 参数：num_cls——调色板条目数（类别数，VOC 传 256）
# VOC 官方调色板生成算法：把类别 id 视为二进制位串，从高位起每 3 位一组，
# 依次分给 R/G/B 三通道的对应二进制位——保证相邻类别颜色差异明显且可复现
def _getvocpallete(num_cls):
    n = num_cls  # 类别数
    pallete = [0] * (n * 3)  # 展平的 RGB 调色板：n 类 × 每类 3 字节，初始全黑
    for j in range(0, n):  # 为每个类别 j 生成一种颜色
        lab = j  # 把类别 id 当作待拆位的位串
        pallete[j * 3 + 0] = 0  # R 通道清零（随后按位或上色）
        pallete[j * 3 + 1] = 0  # G 通道清零
        pallete[j * 3 + 2] = 0  # B 通道清零
        i = 0  # 位组轮次：第 i 轮取 lab 的低 3 位，写入各通道的第 (7-i) 二进制位
        while (lab > 0):  # 位串尚未取尽
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))  # lab 第 3i 位 → R 的第 7-i 位
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))  # lab 第 3i+1 位 → G 的第 7-i 位
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))  # lab 第 3i+2 位 → B 的第 7-i 位
            i = i + 1  # 进入下一轮位组
            lab >>= 3  # 丢弃已消费的低 3 位
    return pallete  # 返回长度 3n 的扁平调色板（可直接传给 PIL 的 putpalette）


vocpallete = _getvocpallete(256)  # VOC 调色板：256 条保证任意 0~255 标签（含 255 忽略类）都有颜色

# ADE20K 官方调色板（硬编码）：每 3 个数为一个类别的 R,G,B，共 151 色——
# 0 号色 (0,0,0) 黑色为背景/未标注，1~150 号对应 ADE20K 的 150 个类别
adepallete = [
    0, 0, 0, 120, 120, 120, 180, 120, 120, 6, 230, 230, 80, 50, 50, 4, 200, 3, 120, 120, 80, 140, 140, 140, 204,  # 类别 0~8（0 号为背景黑色）
    5, 255, 230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7, 150, 5, 61, 120, 120, 70, 8, 255, 51, 255, 6, 82,  # 类别 9~16
    143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3, 0, 102, 200, 61, 230, 250, 255, 6, 51, 11, 102, 255, 255,  # 类别 17~25
    7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220, 255, 9, 92, 112, 9, 255, 8, 255, 214, 7, 255, 224, 255, 184, 6,  # 类别 26~33
    10, 255, 71, 255, 41, 10, 7, 255, 255, 224, 255, 8, 102, 8, 255, 255, 61, 6, 255, 194, 7, 255, 122, 8, 0, 255,  # 类别 34~42
    20, 255, 8, 41, 255, 5, 153, 6, 51, 255, 235, 12, 255, 160, 150, 20, 0, 163, 255, 140, 140, 140, 250, 10, 15,  # 类别 43~50
    20, 255, 0, 31, 255, 0, 255, 31, 0, 255, 224, 0, 153, 255, 0, 0, 0, 255, 255, 71, 0, 0, 235, 255, 0, 173, 255,  # 类别 51~59
    31, 0, 255, 11, 200, 200, 255, 82, 0, 0, 255, 245, 0, 61, 255, 0, 255, 112, 0, 255, 133, 255, 0, 0, 255, 163,  # 类别 60~68
    0, 255, 102, 0, 194, 255, 0, 0, 143, 255, 51, 255, 0, 0, 82, 255, 0, 255, 41, 0, 255, 173, 10, 0, 255, 173, 255,  # 类别 69~77
    0, 0, 255, 153, 255, 92, 0, 255, 0, 255, 255, 0, 245, 255, 0, 102, 255, 173, 0, 255, 0, 20, 255, 184, 184, 0,  # 类别 78~86
    31, 255, 0, 255, 61, 0, 71, 255, 255, 0, 204, 0, 255, 194, 0, 255, 82, 0, 10, 255, 0, 112, 255, 51, 0, 255, 0,  # 类别 87~95
    194, 255, 0, 122, 255, 0, 255, 163, 255, 153, 0, 0, 255, 10, 255, 112, 0, 143, 255, 0, 82, 0, 255, 163, 255,  # 类别 96~103
    0, 255, 235, 0, 8, 184, 170, 133, 0, 255, 0, 255, 92, 184, 0, 255, 255, 0, 31, 0, 184, 255, 0, 214, 255, 255,  # 类别 104~112
    0, 112, 92, 255, 0, 0, 224, 255, 112, 224, 255, 70, 184, 160, 163, 0, 255, 153, 0, 255, 71, 255, 0, 255, 0,  # 类别 113~120
    163, 255, 204, 0, 255, 0, 143, 0, 255, 235, 133, 255, 0, 255, 0, 235, 245, 0, 255, 255, 0, 122, 255, 245, 0,  # 类别 121~128
    10, 190, 212, 214, 255, 0, 0, 204, 255, 20, 0, 255, 255, 255, 0, 0, 153, 255, 0, 41, 255, 0, 255, 204, 41, 0,  # 类别 129~137
    255, 41, 255, 0, 173, 0, 255, 0, 245, 255, 71, 0, 255, 122, 0, 255, 0, 255, 184, 0, 92, 255, 184, 255, 0, 0,  # 类别 138~146
    133, 255, 255, 214, 0, 25, 194, 194, 102, 255, 0, 92, 0, 255]  # 类别 147~150（末尾 5 类）

# Cityscapes 官方调色板（硬编码）：19 个 trainId 类别，每行一类的 R,G,B
cityspallete = [
    128, 64, 128,  # 0: road（道路）
    244, 35, 232,  # 1: sidewalk（人行道）
    70, 70, 70,  # 2: building（建筑）
    102, 102, 156,  # 3: wall（墙体）
    190, 153, 153,  # 4: fence（栅栏）
    153, 153, 153,  # 5: pole（电线杆）
    250, 170, 30,  # 6: traffic light（交通灯）
    220, 220, 0,  # 7: traffic sign（交通标志）
    107, 142, 35,  # 8: vegetation（植被）
    152, 251, 152,  # 9: terrain（地形/裸地）
    0, 130, 180,  # 10: sky（天空）
    220, 20, 60,  # 11: person（行人）
    255, 0, 0,  # 12: rider（骑手）
    0, 0, 142,  # 13: car（汽车）
    0, 0, 70,  # 14: truck（卡车）
    0, 60, 100,  # 15: bus（公交车）
    0, 80, 100,  # 16: train（火车）
    0, 0, 230,  # 17: motorcycle（摩托车）
    119, 11, 32,  # 18: bicycle（自行车）
]  # Cityscapes 19 类调色板结束
