# ============================================================================
# 【文件说明】demo.py —— 单张图片推理演示脚本：加载已训练的分割模型，对指定
#               的输入图片做语义分割预测，将预测的类别索引图按数据集调色板
#               着色后保存为 PNG 结果图。
# 【运行方式】python demo.py --model fcn32s_vgg16_voc --dataset pascal_voc
#               --input-pic <图片路径> --outdir ./eval [--save-folder ~/.torch/models]
#               （模型名格式为 模型_骨干_数据集，权重从 save-folder 目录加载）
# 【整体流程】解析命令行参数 → demo() 中创建输出目录并预处理输入图片
#               （ToTensor + ImageNet Normalize，加 batch 维）→ get_model()
#               加载预训练权重构建模型 → eval 模式下 no_grad 前向推理 →
#               对输出 logits 做 argmax 得每像素类别 → get_color_pallete 着色 →
#               以原文件名保存为 <outdir>/<原文件名>.png。
# ============================================================================
import os  # 操作系统接口：输出目录创建、路径拆分与拼接
import sys  # 系统功能：把项目根目录加入模块搜索路径
import argparse  # 命令行参数解析库
import torch  # PyTorch 主库：张量计算与模型推理

cur_path = os.path.abspath(os.path.dirname(__file__))  # 获取当前脚本所在目录（scripts/）的绝对路径
root_path = os.path.split(cur_path)[0]  # 取其上一级目录，即项目根目录
sys.path.append(root_path)  # 将项目根目录加入 sys.path，保证 core 包可被导入

from torchvision import transforms  # 图像变换库：构建 ToTensor + Normalize 预处理流水线
from PIL import Image  # PIL 图像库：读取输入图片、保存着色后的预测结果
from core.utils.visualize import get_color_pallete  # 可视化工具：按数据集调色板给类别索引图着色
from core.models import get_model  # 模型工厂函数：按"模型_骨干_数据集"组合名加载预训练分割模型

parser = argparse.ArgumentParser(  # 创建命令行参数解析器
    description='Predict segmentation result from a given image')  # 脚本描述：对给定图片预测分割结果
parser.add_argument('--model', type=str, default='fcn32s_vgg16_voc',  # 模型组合名，格式为 模型_骨干_数据集，默认 fcn32s_vgg16_voc
                    help='model name (default: fcn32_vgg16)')  # 【注意】help 写的默认值 fcn32_vgg16 与实际 default='fcn32s_vgg16_voc' 不一致
parser.add_argument('--dataset', type=str, default='pascal_aug', choices=['pascal_voc, pascal_aug, ade20k, citys'],  # 数据集名称（决定调色板），默认 pascal_aug
                    help='dataset name (default: pascal_voc)')  # 【注意】choices 误写成一个整体字符串而非列表，用户传入任何合法数据集名都会被 argparse 判为非法选择（仅默认值可用）；help 默认值也与实际不符
parser.add_argument('--save-folder', default='~/.torch/models',  # 预训练权重文件所在目录（支持 ~ 展开），默认 ~/.torch/models
                    help='Directory for saving checkpoint models')  # 帮助文本：权重目录
parser.add_argument('--input-pic', type=str, default='../datasets/voc/VOC2012/JPEGImages/2007_000032.jpg',  # 待推理的输入图片路径，默认为 VOC2012 的一张示例图
                    help='path to the input picture')  # 帮助文本：输入图片路径
parser.add_argument('--outdir', default='./eval', type=str,  # 预测结果保存目录，默认 ./eval
                    help='path to save the predict result')  # 帮助文本：预测结果保存路径
parser.add_argument('--local_rank', type=int, default=0)  # GPU 序号参数（单卡演示用，默认 0；仅为兼容 get_model 接口保留）
args = parser.parse_args()  # 解析命令行参数


def demo(config):  # 推理演示函数：config 为命令行参数命名空间
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择计算设备：有 GPU 用 cuda，否则用 cpu
    # output folder
    if not os.path.exists(config.outdir):  # 若输出目录不存在
        os.makedirs(config.outdir)  # 创建输出目录

    # image transform
    transform = transforms.Compose([  # 组合输入图像预处理流水线
        transforms.ToTensor(),  # PIL 图像转张量，像素值由 [0,255] 归一化到 [0,1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 按 ImageNet 训练集均值/标准差做标准化
    ])  # 预处理流水线定义结束
    image = Image.open(config.input_pic).convert('RGB')  # 打开输入图片并统一转为 RGB 三通道（防止灰度/RGBA 图导致通道数不符）
    images = transform(image).unsqueeze(0).to(device)  # 预处理后加 batch 维得 (1,C,H,W)，并搬运到计算设备

    model = get_model(args.model, local_rank=args.local_rank, pretrained=True, root=args.save_folder).to(device)  # 【注意】函数形参为 config，但此处用的是全局变量 args（config 仅用于 outdir/input_pic），风格不统一但可运行；pretrained=True 表示从 root 目录加载训练好的权重
    print('Finished loading model!')  # 打印模型加载完成提示

    model.eval()  # 切换评估模式：BN 使用运行统计量，Dropout 关闭
    with torch.no_grad():  # 关闭梯度追踪，纯推理，省显存提速
        output = model(images)  # 前向推理，得到分割输出列表

    pred = torch.argmax(output[0], 1).squeeze(0).cpu().data.numpy()  # 取主输出（列表第 0 项）沿通道维 argmax 得类别图 → 去 batch 维 → 转 CPU numpy 数组 (H,W)
    mask = get_color_pallete(pred, args.dataset)  # 按数据集调色板将类别索引图着色为 PIL 彩色图
    outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '.png'  # 取输入文件名（去扩展名）拼接 .png 作为输出文件名
    mask.save(os.path.join(args.outdir, outname))  # 将着色后的预测掩码保存到输出目录


if __name__ == '__main__':  # 脚本直接执行时的入口（被 import 时不运行）
    demo(args)  # 以命令行参数运行推理演示
