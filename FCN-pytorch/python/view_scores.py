# -*- coding: utf-8 -*-
# ============================================================================
# 【文件说明】view_scores.py —— 查看 train.py 保存的验证指标文件（.npy"得分"文件）
#             【注释版新增】
# 【解决什么问题】meanIU.npy / meanPixel.npy 是 numpy 的二进制格式：双击打不开、
#             记事本打开是乱码，必须用 numpy 才能读。本脚本把它们变成人能读的
#             终端表格（逐轮指标、最优轮、每类 IoU），并可选画出训练曲线图。
# 【两个文件里存的是什么】
#   meanIU.npy    形状 (总轮数, 类别数)：第 e 行 = 第 e 轮验证时"每个类"的 IoU
#   meanPixel.npy 形状 (总轮数,)：     第 e 个 = 第 e 轮验证的像素精度 pix_acc
#   注意：train.py 开头会先验证一次"未训练模型"（基线），结果写在第 0 行；
#   之后每训练完一轮，验证结果就写入对应行。中断过的训练只有已完成的行有数。
#   另外每次重新运行 train.py 会把这两个文件清零重写（历史会丢失）。
# 【运行方式】在仓库根目录（与 scores/ 同级）执行：
#   python python/view_scores.py                     # 自动选 scores/ 下最新的实验目录
#   python python/view_scores.py scores/FCNs-xxx     # 指定某个实验目录
#   python python/view_scores.py --iu 路径/meanIU.npy --pixel 路径/meanPixel.npy
#   python python/view_scores.py --colors CamVid/label_colors.txt  # 附带类别名
#   python python/view_scores.py --plot curves.png   # 额外保存训练曲线图
#   python python/view_scores.py --tail 0            # 表格显示全部轮次（默认最近 20 轮）
# 【依赖】numpy 必需；--plot 需要 matplotlib；tqdm 不需要
# ============================================================================
from __future__ import print_function  # 兼容 Python2/3 的 print 函数写法

import argparse   # 命令行参数解析
import os         # 路径拼接、目录扫描、文件修改时间
import sys        # 错误退出码
import warnings   # 屏蔽 nanmean 对全 nan 切片的无害警告

import numpy as np  # 读取 .npy 数组与统计计算


def find_score_dir():  # 自动在 scores/ 下找"最近训练过"（meanIU.npy 修改时间最新）的实验目录
    if not os.path.isdir('scores'):  # scores 目录都不存在
        return None  # 返回未找到
    cands = []  # 候选目录列表
    for d in os.listdir('scores'):  # 遍历 scores/ 下每个条目
        p = os.path.join('scores', d)  # 拼出完整路径
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, 'meanIU.npy')):  # 是目录且含指标文件
            cands.append(p)  # 收为候选
    if not cands:  # 一个候选都没有
        return None  # 返回未找到
    return max(cands, key=os.path.getmtime)  # 按文件修改时间取最新（最近训练过的配置）


def load_palette(path):  # 解析 label_colors.txt：行号 = 类别索引（与 CamVid_utils.py 的解析一致）
    names = []  # 类别名列表（下标即类别索引）
    try:  # 文件可能不存在或读取失败
        for line in open(path).read().split('\n'):  # 逐行读取
            if not line.strip():  # 空行跳过（含末尾空行）
                continue  # 继续下一行
            parts = line.split()  # 按空白切分：前三个数字 + 类别名
            names.append(parts[-1])  # 最后一个词是类别名
    except Exception:  # 任何读取问题
        return None  # 返回 None（调用方降级为只显示类别索引）
    return names if names else None  # 返回类别名列表（空文件也返回 None）


def class_name(idx, names):  # 取第 idx 类的显示名：有对照表用类别名，否则用序号
    if names and idx < len(names):  # 有名字且索引没越界
        return names[idx]  # 返回类别名
    return 'class%d' % idx  # 兜底：class+序号


def main():  # 主流程：定位并读取 npy → 打印逐轮表格 → 最优/最新摘要 → 每类 IoU → 可选画图
    parser = argparse.ArgumentParser(description='查看 train.py 保存的 IoU / 像素精度指标（.npy 得分文件）')  # 参数解析器
    parser.add_argument('dir', nargs='?', default=None,  # 可选位置参数：实验目录
                        help='实验目录（如 scores/FCNs-xxx；省略则自动选最新）')  # 说明
    parser.add_argument('--iu', default=None, help='直接指定 meanIU.npy 路径')  # 直接指定 IoU 文件
    parser.add_argument('--pixel', default=None, help='直接指定 meanPixel.npy 路径')  # 直接指定像素精度文件
    parser.add_argument('--colors', default=None, help='label_colors.txt 路径（用于显示类别名）')  # 类别对照表
    parser.add_argument('--plot', default=None, help='训练曲线图保存路径（如 curves.png，需 matplotlib）')  # 画图输出
    parser.add_argument('--tail', type=int, default=20, help='表格只显示最近 N 轮（0=全部，默认 20）')  # 表格截断
    args = parser.parse_args()  # 解析命令行参数

    # ---------- 定位两个 npy 文件 ----------
    if args.iu and args.pixel:  # 用户直接给了两个文件路径
        iu_path, px_path = args.iu, args.pixel  # 直接采用
    else:  # 走目录方式
        d = args.dir or find_score_dir()  # 指定的目录，或自动发现最新的
        if not d:  # 没找到
            sys.exit('[错误] 找不到指标目录/文件。请在仓库根目录运行，或用 --iu/--pixel 指定路径')  # 提示后退出
        iu_path = os.path.join(d, 'meanIU.npy')       # 该目录下的 IoU 表
        px_path = os.path.join(d, 'meanPixel.npy')    # 该目录下的像素精度表
    for p in (iu_path, px_path):  # 逐一检查文件存在性
        if not os.path.isfile(p):  # 缺失
            sys.exit('[错误] 文件不存在: %s' % p)  # 报错退出

    iu = np.load(iu_path)  # 读入 IoU 数组，形状应为 (总轮数, 类别数)
    px = np.load(px_path)  # 读入像素精度数组，形状应为 (总轮数,)
    if iu.ndim != 2 or px.ndim != 1 or iu.shape[0] != px.shape[0]:  # 形状不符合 train.py 的约定
        sys.exit('[错误] npy 形状异常：meanIU %s、meanPixel %s（期望 (轮数, 类别数) 与 (轮数,)）' % (iu.shape, px.shape))  # 报错退出
    n_total, n_class = iu.shape  # 总轮数（数组行数）与类别数（列数）

    # ---------- 类别名（可选）：用户指定 > 常见位置自动查找 ----------
    names = load_palette(args.colors) if args.colors else None  # 用户显式指定时直接解析
    if not names:  # 没有指定或解析失败
        for cand in (os.path.join('CamVid', 'label_colors.txt'),  # 当前工作目录下的 CamVid/
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'CamVid', 'label_colors.txt')):  # 脚本上一级的 CamVid/
            if os.path.isfile(cand):  # 找到第一个存在的
                names = load_palette(cand)  # 解析之
                break  # 不再找下一个

    # ---------- 判断哪些轮已经完成验证（全零且无 nan 的行 = 还没轮到） ----------
    with warnings.catch_warnings():  # 临时屏蔽 nanmean 对全 nan 行的警告
        warnings.simplefilter('ignore')  # 忽略 RuntimeWarning
        done = [e for e in range(n_total)  # 逐行判断
                if px[e] != 0 or np.any(iu[e] != 0) or np.any(np.isnan(iu[e]))]  # 有任何非零值或 nan 即已完成
        meanious = np.array([np.nanmean(iu[e]) for e in done]) if done else np.array([])  # 各已完成轮的 meanIoU

    # ---------- 基本信息 ----------
    print('=' * 78)  # 分隔线
    print('指标文件: %s / %s' % (iu_path, px_path))  # 数据来源
    print('数组形状: meanIU %s（轮 x 类）, meanPixel %s（轮）' % (iu.shape, px.shape))  # 形状
    if names:  # 有类别名可显示
        print('类别对照: 已加载 label_colors.txt（%d 类，索引=文件行号-1）' % len(names))  # 说明索引含义
    if not done:  # 一行有效数据都没有
        print('尚无任何一轮验证记录（全为 0）——训练可能还没跑完第 0 轮。')  # 说明情况
        return  # 正常结束
    print('已完成验证: %d / %d 轮（第 0 行通常是训练前的基线；每轮数值含义见文件头注释）' % (len(done), n_total))  # 进度概况
    print('=' * 78)  # 分隔线

    # ---------- 逐轮汇总表 ----------
    shown = done if args.tail == 0 else done[-args.tail:]  # 按参数截取要显示的轮次
    if args.tail != 0 and len(done) > args.tail:  # 发生了截断
        print('（仅显示最近 %d 轮，全部 %d 轮请加 --tail 0）\n' % (args.tail, len(done)))  # 截断提示
    print('%-8s %-14s %-14s' % ('epoch', 'pix_acc', 'meanIoU'))  # 表头
    print('-' * 38)  # 表头下横线
    pos = {e: i for i, e in enumerate(done)}  # 轮号 → 在 meanious 数组中的下标（避免重复查找）
    for e in shown:  # 逐行输出
        print('%-8d %-14.4f %-14.4f' % (e, px[e], meanious[pos[e]]))  # 一行：轮次 / 像素精度 / 平均 IoU

    # ---------- 最优轮 / 最新轮摘要 ----------
    best_i = int(np.argmax(meanious))  # meanIoU 最大的轮在 done 列表中的下标
    best_e = done[best_i]  # 对应的轮号
    print('\n[最优] epoch %d: pix_acc %.4f, meanIoU %.4f' % (best_e, px[best_e], meanious[best_i]))  # 最优摘要
    last_e = done[-1]  # 最新完成的轮号
    print('[最新] epoch %d: pix_acc %.4f, meanIoU %.4f' % (last_e, px[last_e], meanious[-1]))  # 最新摘要

    # ---------- 每类 IoU（取最优轮） ----------
    row = iu[best_e]  # 最优轮的各类 IoU
    print('\n----- 每类 IoU（最优轮 epoch %d；nan = 该类在验证集与预测中都未出现，不参与平均）-----' % best_e)  # 小节标题
    order = sorted(range(n_class),  # 排序规则：非 nan 类按 IoU 从大到小在前，nan 类沉底
                   key=lambda c: (0, 0.0) if np.isnan(row[c]) else (1, row[c]), reverse=True)  # 用元组避免 nan 参与比较
    print('%-6s %-24s %s' % ('索引', '类别', 'IoU'))  # 表头
    for c in order:  # 逐类输出
        v = row[c]  # 该类 IoU 值
        print('%-6d %-24s %s' % (c, class_name(c, names), 'nan（未出现）' if np.isnan(v) else '%.4f' % v))  # 一行

    # 全程 nan 的类：从头到尾都没在验证集出现过（被 meanIoU 的 nanmean 忽略）
    always_nan = [c for c in range(n_class) if all(np.isnan(iu[e][c]) for e in done)]  # 收集这些类
    if always_nan:  # 存在时补充说明
        print('\n[说明] 以下类别在已完成的 %d 轮里从未在验证集出现（meanIoU 自动忽略它们）:' % len(done))  # 提示
        print('       ' + ', '.join('%d:%s' % (c, class_name(c, names)) for c in always_nan))  # 列出索引与类名

    # ---------- 可选：画训练曲线图 ----------
    if args.plot:  # 用户要求出图
        try:  # matplotlib 可能未安装
            import matplotlib  # 导入 matplotlib 主模块
            matplotlib.use('Agg')  # 使用无界面后端：没有显示器也能保存图片
            from matplotlib import font_manager, pyplot as plt  # 字体管理与绘图接口
        except ImportError:  # 没装 matplotlib
            sys.exit('[错误] 未安装 matplotlib，无法画图：pip install matplotlib')  # 提示后退出
        for fp in ('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf',  # 常见中文字体（Linux，Noto 思源黑体）
                   '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # 另一路径（Noto CJK）
                   'C:/Windows/Fonts/msyh.ttc'):  # Windows 微软雅黑
            if os.path.isfile(fp):  # 找到第一个存在的
                font_manager.fontManager.addfont(fp)  # 注册给 matplotlib
                break  # 不再尝试后续路径
        plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'Noto Sans CJK SC', 'Microsoft YaHei', 'WenQuanYi Zen Hei', 'sans-serif']  # 中文字体优先级（按机器上装了的生效）
        plt.rcParams['axes.unicode_minus'] = False  # 让负号正常显示（避免中文字体缺负号字形）
        ep = np.array(done)  # x 轴：已完成的轮次
        iu_done = iu[done]  # 一次性取出已完成轮的子表（避免循环内反复花式索引）
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)  # 上下两个子图
        axes[0].plot(ep, px[done], 'o-', label='像素精度 pix_acc', color='tab:blue')  # 上图：像素精度曲线
        axes[0].plot(ep, meanious, 's-', label='平均IoU meanIoU', color='tab:red')  # 上图：meanIoU 曲线
        axes[0].set_title('验证指标随训练轮数的变化')  # 上图标题
        axes[0].set_xlabel('epoch（轮次）')  # x 轴名
        axes[0].set_ylabel('精度 / IoU')  # y 轴名
        axes[0].set_ylim(0, 1)  # 固定 0~1 范围，方便观察离 1 还有多远
        axes[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1))  # 图例放绘图区外右侧，避免遮住曲线
        axes[0].grid(alpha=0.3)  # 半透明网格
        with warnings.catch_warnings():  # 又要屏蔽全 nan 类的平均警告
            warnings.simplefilter('ignore')  # 忽略
            class_avg = np.nanmean(iu_done, axis=0)  # 每类在已完成轮上的平均 IoU（忽略 nan，nan 类得 nan）
        top = sorted(range(n_class), key=lambda c: -float(np.nan_to_num(class_avg[c])))[:8]  # 平均值最高的前 8 类
        for c in top:  # 下图逐类画曲线
            axes[1].plot(ep, iu_done[:, c], 'o-', ms=3, label='%d:%s' % (c, class_name(c, names)))  # 该类的 IoU 曲线
        axes[1].set_title('各类别 IoU 的变化（平均最高的前 8 类）')  # 下图标题
        axes[1].set_xlabel('epoch（轮次）')  # x 轴名
        axes[1].set_ylabel('IoU')  # y 轴名
        axes[1].set_ylim(0, 1)  # 固定 0~1
        axes[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)  # 图例在绘图区外右侧
        axes[1].grid(alpha=0.3)  # 半透明网格
        fig.savefig(args.plot, dpi=150)  # 保存图片（150 dpi 足够清晰）
        print('\n[画图] 训练曲线已保存: %s' % args.plot)  # 输出提示


if __name__ == '__main__':  # 作为主程序运行
    main()  # 执行主流程
