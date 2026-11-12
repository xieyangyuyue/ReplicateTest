# ============================================================================
# 【文件说明】resnext.py —— ResNeXt 分组卷积骨干网络，提供
#              resnext50_32x4d / resnext101_32x8d 两个工厂函数。实现风格与
#              resnetv1b.py 同源（v1b 风格，支持 dilated 空洞模式），可作
#              分割模型的编码器（经 base_models/__init__.py 对外导出）。
# 【核心思想】"分组卷积 + 增大基数（cardinality）"：在 Bottleneck 基础上把
#              3×3 卷积拆成 groups 组并行分支，实际分组卷积总通道
#              width = int(planes * (base_width / 64)) * groups；"32x4d" 即
#              32 组、每组基准宽度 4。分组在同等参数量/计算量下换来了更多
#              的变换路径（网络更"宽"），精度优于同规模的普通 ResNet。
# 【注意】     本文件 dilated 默认 False（与 resnetv1b.py 默认 True 不同），
#              若需输出步长 8 的分割骨干，须显式传 dilated=True；此时
#              stage3/4 的 stride 改写为 1、空洞率分别取 2/4。
# ============================================================================
import torch.nn as nn  # PyTorch 神经网络模块：层定义、容器、参数初始化
import torch.utils.model_zoo as model_zoo  # 预训练权重下载工具【注意】load_url 在新版 PyTorch 已废弃，建议改用 torch.hub.load_state_dict_from_url

__all__ = ['ResNext', 'resnext50_32x4d', 'resnext101_32x8d']  # from xxx import * 时对外导出的符号清单：ResNext 类与两个工厂函数

model_urls = {  # ResNeXt 各配置的 ImageNet 预训练权重下载地址表（pytorch.org 官方托管）
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',  # ResNeXt-50（32 组×4 宽）权重地址
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',  # ResNeXt-101（32 组×8 宽）权重地址
}  # 权重地址表结束


class Bottleneck(nn.Module):  # ResNeXt 瓶颈残差块：1×1 → 分组 3×3 → 1×1，支持空洞卷积
    expansion = 4  # 输出通道 = planes × 4（末个 1×1 卷积升维 4 倍）

    # ----------------------- Bottleneck.__init__ 参数说明 -----------------------
    # inplanes：块输入通道数；planes：基准通道数（决定 width 与输出 planes*4）
    # stride：下采样步长（放在 conv2 的 3×3 上）；downsample：捷径投影子模块
    # groups：分组卷积的组数（基数 cardinality，如 32）；base_width：每组基准宽度（如 4/8）
    # dilation：3×3 卷积的空洞率；norm_layer：归一化层构造函数
    # **kwargs：吸收并忽略多余关键字参数【注意】静默丢弃参数，传错参数名不会报错，排错时需留意
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs):  # 构造分组卷积瓶颈块
        super(Bottleneck, self).__init__()  # 调用父类构造函数
        width = int(planes * (base_width / 64.)) * groups  # 计算分组卷积的实际总通道数：如 planes=64、base_width=4、groups=32 → width=128

        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)  # 1×1 卷积①：通道降维 inplanes → width
        self.bn1 = norm_layer(width)  # BN①【注意】norm_layer 默认 None，若调用方不显式传入，此处会抛 NoneType 不可调用异常（本工程 _make_layer 恒有传入，故未触发）
        self.conv2 = nn.Conv2d(width, width, 3, stride, dilation, dilation, groups, bias=False)  # 3×3 分组卷积：位置参数依次 kernel=3、stride、padding=dilation、dilation=dilation、groups；stride>1 时下采样
        self.bn2 = norm_layer(width)  # BN②
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)  # 1×1 卷积③：通道升维 width → planes*4
        self.bn3 = norm_layer(planes * self.expansion)  # BN③：残差分支最后一个 BN（zero_init_residual 的置零对象）
        self.relu = nn.ReLU(True)  # ReLU 激活（位置参数 inplace=True）
        self.downsample = downsample  # 保存捷径投影（None 表示恒等捷径）
        self.stride = stride  # 记录步长（存档用，forward 不使用）

    def forward(self, x):  # 前向：三卷积瓶颈结构 + 捷径相加
        identity = x  # 保存输入作为捷径分支

        out = self.conv1(x)  # 1×1 降维：(N, inplanes, H, W) → (N, width, H, W)
        out = self.bn1(out)  # BN①
        out = self.relu(out)  # ReLU①

        out = self.conv2(out)  # 分组 3×3 卷积（stride>1 时 H、W 减半；dilation>1 时扩大感受野且分辨率不变）
        out = self.bn2(out)  # BN②
        out = self.relu(out)  # ReLU②

        out = self.conv3(out)  # 1×1 升维：(N, width, H, W) → (N, planes*4, H, W)
        out = self.bn3(out)  # BN③

        if self.downsample is not None:  # 尺寸/通道不匹配时启用投影捷径
            identity = self.downsample(x)  # 1×1 卷积 + BN 对齐捷径形状

        out += identity  # 残差相加：F(x) + x
        out = self.relu(out)  # 相加之后再激活

        return out  # 返回 (N, planes*4, H/stride, W/stride)


class ResNext(nn.Module):  # ResNeXt 主体：stem + 4 个残差 stage + 分类头（v1b 风格，支持 dilated）

    # ----------------------- ResNext.__init__ 参数说明 -----------------------
    # block：残差块类型；layers：四 stage 块数列表
    # num_classes：分类头类别数（作分割编码器时不使用）；zero_init_residual：末 BN 零初始化开关
    # groups：分组数（基数）；width_per_group：每组基准宽度
    # dilated：True 时 stage3/4 用 stride=1 + 空洞率 2/4（输出步长 8；本文件工厂函数默认 False）
    # norm_layer：归一化层构造函数；**kwargs：吸收多余参数（兼容透传）
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1,
                 width_per_group=64, dilated=False, norm_layer=nn.BatchNorm2d, **kwargs):  # 按配置构建 ResNeXt
        super(ResNext, self).__init__()  # 调用父类构造函数
        self.inplanes = 64  # 通道数游标初值（stem 输出 64 通道）
        self.groups = groups  # 记录分组数，供 _make_layer 传给各残差块
        self.base_width = width_per_group  # 记录每组基准宽度

        self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)  # 7×7 stem 卷积：kernel=7、stride=2、padding=3，(N,3,H,W) → (N,64,H/2,W/2)
        self.bn1 = norm_layer(self.inplanes)  # stem 的批归一化
        self.relu = nn.ReLU(True)  # ReLU 激活
        self.maxpool = nn.MaxPool2d(3, 2, 1)  # 3×3 最大池化 stride=2：分辨率累计降至 1/4

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)  # stage1：64 通道、stride=1 → 输出 64*E，分辨率 1/4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)  # stage2：128 通道、stride=2 → 分辨率 1/8
        if dilated:  # 空洞模式：stage3/4 不再下采样，改用空洞卷积保持 1/8 分辨率
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)  # stage3：stride 由 2 改写为 1、空洞率 2 → 仍为 1/8
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)  # stage4：stride=1、空洞率 4 → 1/8 且感受野最大，输出 512*E
        else:  # 常规模式：逐级下采样到 1/32
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)  # stage3：stride=2 → 分辨率 1/16
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)  # stage4：stride=2 → 分辨率 1/32

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应全局平均池化（分类头专用）
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 分类全连接层：512*E → num_classes

        for m in self.modules():  # 遍历所有子模块统一初始化
            if isinstance(m, nn.Conv2d):  # 卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming 正态初始化（适配 ReLU）
            elif isinstance(m, nn.BatchNorm2d):  # 批归一化层
                nn.init.constant_(m.weight, 1)  # γ 初始化为 1
                nn.init.constant_(m.bias, 0)  # β 初始化为 0

        if zero_init_residual:  # 可选：残差分支末 BN 零初始化
            for m in self.modules():  # 再次遍历所有模块
                if isinstance(m, Bottleneck):  # 瓶颈块（本文件仅有此块型）
                    nn.init.constant_(m.bn3.weight, 0)  # γ=0 → 初始时残差分支输出 0，块退化为恒等映射

    # ----------------------- _make_layer 参数说明 -----------------------
    # block：残差块类型；planes：本 stage 基准通道数；blocks：块数量
    # stride：首块步长（dilated 模式的 stage3/4 传 1）；dilation：本 stage 空洞率（1/2/4）
    # norm_layer：归一化层构造函数
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):  # 堆叠一个 stage（含空洞率渐进过渡策略）
        downsample = None  # 捷径投影，默认 None（恒等捷径）
        if stride != 1 or self.inplanes != planes * block.expansion:  # 需要下采样或通道数变化时
            downsample = nn.Sequential(  # 捷径投影 = 1×1 卷积 + BN
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),  # 1×1 卷积：对齐通道并按 stride 下采样
                norm_layer(planes * block.expansion)  # 投影后的 BN
            )  # 捷径投影 Sequential 定义结束

        layers = list()  # 本 stage 的残差块列表
        if dilation in (1, 2):  # 空洞率为 1 或 2 的 stage（stage1/2/3）
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,  # 首块：以位置参数依次传 stride、downsample、groups、base_width
                                self.base_width, norm_layer=norm_layer))  # 未传 dilation → 3×3 用默认空洞率 1（即沿用上一 stage 的空洞率，平滑过渡）
        elif dilation == 4:  # 空洞率为 4 的 stage（dilated 模式的 stage4）
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,  # 首块：同样以位置参数传 groups/base_width
                                self.base_width, dilation=2, norm_layer=norm_layer))  # 显式传 dilation=2：首块 3×3 沿用上一 stage（stage3）的空洞率
        else:  # 传入了未知空洞率
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))  # 抛运行时错误：空洞率只允许 1/2/4
        self.inplanes = planes * block.expansion  # 更新通道游标为本 stage 输出通道
        for _ in range(1, blocks):  # 其余 blocks-1 个常规块
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,  # 以关键字传分组配置（此处 kwargs 均为函数内新建字典，无副作用）
                                dilation=dilation, norm_layer=norm_layer))  # 空洞率取本级 dilation

        return nn.Sequential(*layers)  # 将块列表串成顺序容器返回

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像张量 (N, 3, H, W)；返回：分类 logits (N, num_classes)
    # 形状以 Bottleneck（E=expansion=4）为例；dilated=False（本文件默认）时输出步长为 32
    def forward(self, x):  # 前向：stem → 4 stage → 全局池化 → 分类
        x = self.conv1(x)  # 7×7 卷积 stride=2：(N, 3, H, W) → (N, 64, H/2, W/2)
        x = self.bn1(x)  # BN（形状不变）
        x = self.relu(x)  # ReLU（形状不变）
        x = self.maxpool(x)  # 最大池化 stride=2 → (N, 64, H/4, W/4)

        x = self.layer1(x)  # stage1（stride=1）→ (N, 256, H/4, W/4)
        x = self.layer2(x)  # stage2（stride=2）→ (N, 512, H/8, W/8)
        x = self.layer3(x)  # stage3：dilated=False → (N, 1024, H/16, W/16)；dilated=True（空洞率 2、stride=1）→ (N, 1024, H/8, W/8)
        x = self.layer4(x)  # stage4：dilated=False → (N, 2048, H/32, W/32)；dilated=True（空洞率 4、stride=1）→ (N, 2048, H/8, W/8)

        x = self.avgpool(x)  # 全局平均池化 → (N, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # 展平 → (N, 2048)
        x = self.fc(x)  # 全连接 → (N, num_classes)

        return x  # 返回分类 logits


# 工厂函数：构建 ResNeXt-50（32 组、每组宽 4，Bottleneck × [3,4,6,3]）
# 通过向 kwargs 注入 groups/width_per_group 配置分组超参；pretrained=True 时下载 ImageNet 权重并严格加载
def resnext50_32x4d(pretrained=False, **kwargs):  # **kwargs 透传给 ResNext（如 dilated、norm_layer）
    kwargs['groups'] = 32  # 注入基数 32：3×3 卷积分成 32 组
    kwargs['width_per_group'] = 4  # 注入每组基准宽度 4
    model = ResNext(Bottleneck, [3, 4, 6, 3], **kwargs)  # 构建 ResNeXt-50（默认 dilated=False）
    if pretrained:  # 加载预训练权重
        state_dict = model_zoo.load_url(model_urls['resnext50_32x4d'])  # 下载 ResNeXt-50 的 ImageNet 权重【注意】load_url 为废弃 API
        model.load_state_dict(state_dict)  # 加载权重（strict 默认 True，结构与预训练不一致会报错）
    return model  # 返回模型


# 工厂函数：构建 ResNeXt-101（32 组、每组宽 8，Bottleneck × [3,4,23,3]）
# 通过向 kwargs 注入 groups/width_per_group 配置分组超参；pretrained=True 时下载 ImageNet 权重并严格加载
def resnext101_32x8d(pretrained=False, **kwargs):  # **kwargs 透传给 ResNext
    kwargs['groups'] = 32  # 注入基数 32
    kwargs['width_per_group'] = 8  # 注入每组基准宽度 8（比 50 层更宽）
    model = ResNext(Bottleneck, [3, 4, 23, 3], **kwargs)  # 构建 ResNeXt-101（stage3 加深到 23 块）
    if pretrained:  # 加载预训练权重
        state_dict = model_zoo.load_url(model_urls['resnext101_32x8d'])  # 下载 ResNeXt-101 的 ImageNet 权重【注意】load_url 为废弃 API
        model.load_state_dict(state_dict)  # 加载权重（strict 默认 True）
    return model  # 返回模型


if __name__ == '__main__':  # 直接运行本文件时的简易自测入口
    model = resnext101_32x8d()  # 构建 ResNeXt-101（不加载预训练权重，仅验证网络结构可构建）
