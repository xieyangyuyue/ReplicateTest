# ============================================================================
# 【文件说明】resnetv1b.py —— ResNetV1b / ResNetV1s 骨干网络（从 gluon-cv
#              移植），提供 resnet18/34/50/101/152_v1b 与
#              resnet50/101/152_v1s 两组工厂函数。本工程中是分割模型的默认
#              编码器：segbase.py 的 SegBaseModel 默认构建 resnet50_v1s
#              （dilated=True），FCN(v2)/PSPNet/DeepLab 等模型经 base_forward
#              逐层调用其 conv1/bn1/relu/maxpool/layer1~4 取 c1~c4 四阶段特征。
# 【核心思想】v1b 相对 v1 的改动：下采样（stride=2）从 1×1 卷积移到 3×3 卷积
#              上（等价 torchvision 的 ResNet-v1.5 结构，gluon-cv 命名为 v1b，
#              属改进型/预激活式残差单元设计）；并原生支持 dilated 空洞卷积
#              模式（默认开启）：dilated=True 时 stage3/4 的 stride 由 2 改写
#              为 1、空洞率分别取 2/4（stage 内首块沿用上一级空洞率做渐进
#              过渡），使特征图保持 1/8 分辨率的同时扩大感受野，便于分割头
#              恢复细节。v1s 变体另启用 deep_stem：以三个 3×3 卷积替换 7×7
#              stem（输出 128 通道），更适配空洞卷积/分割场景。
# 【预训练】   v1b 系列：经 torch.utils.model_zoo 下载 torchvision 官方
#              ImageNet 权重，按 key 过滤后合并加载（等效 strict=False）；
#              v1s 系列：经 model_store.get_resnet_file 从 encoding S3 仓库
#              下载 gluon 转换权重，以 strict=False 宽松加载。
# ============================================================================
import torch  # PyTorch 主库：张量运算及 torch.load 反序列化权重文件
import torch.nn as nn  # PyTorch 神经网络模块：层定义与容器
import torch.utils.model_zoo as model_zoo  # torchvision 风格权重下载工具【注意】load_url 在新版 PyTorch 已废弃，建议改用 torch.hub.load_state_dict_from_url

__all__ = ['ResNetV1b', 'resnet18_v1b', 'resnet34_v1b', 'resnet50_v1b',  # from xxx import * 时对外导出的符号清单：ResNetV1b 类与 v1b 系工厂函数
           'resnet101_v1b', 'resnet152_v1b', 'resnet152_v1s', 'resnet101_v1s', 'resnet50_v1s']  # 清单续行：v1b 续 + v1s 系工厂函数

model_urls = {  # v1b 系列加载用的 torchvision 官方 ImageNet 权重地址表（v1s 系列改用 model_store，不走此表）
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',  # ResNet-18 权重地址
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',  # ResNet-34 权重地址
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',  # ResNet-50 权重地址
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',  # ResNet-101 权重地址
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',  # ResNet-152 权重地址
}  # 权重地址表结束


class BasicBlockV1b(nn.Module):  # v1b 基础残差块：两个 3×3 卷积 + 捷径相加，支持空洞卷积（resnet18/34_v1b 使用）
    expansion = 1  # 输出通道相对 planes 的放大倍数：BasicBlock 不扩通道，恒为 1

    # ----------------------- BasicBlockV1b.__init__ 参数说明 -----------------------
    # inplanes：块输入通道数；planes：块内卷积通道数（= 输出通道数）
    # stride：下采样步长（放在 conv1 的 3×3 上，仅 stage 首块 >1）
    # dilation：conv1 的空洞率（dilated stage 的首块传"上一级空洞率"，实现渐进过渡）
    # downsample：捷径投影子模块（通道/分辨率不一致时才非 None）
    # previous_dilation：conv2 的空洞率（= 本 stage 的空洞率）
    # norm_layer：归一化层构造函数
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):  # 构造支持空洞卷积的两卷积残差块
        super(BasicBlockV1b, self).__init__()  # 调用父类构造函数
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride,  # 3×3 卷积①：位置参数 kernel=3、stride；承担下采样
                               dilation, dilation, bias=False)  # padding=dilation、dilation=dilation：空洞卷积下仍保持分辨率不变
        self.bn1 = norm_layer(planes)  # BN①
        self.relu = nn.ReLU(True)  # ReLU 激活（位置参数 inplace=True）
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,  # 3×3 卷积②：stride=1，padding 取 previous_dilation
                               dilation=previous_dilation, bias=False)  # 以关键字显式再传 dilation=previous_dilation（与上一位置参数一致），空洞率用本 stage 的新值
        self.bn2 = norm_layer(planes)  # BN②（相加之前不再激活）
        self.downsample = downsample  # 保存捷径投影（None 表示恒等捷径）
        self.stride = stride  # 记录步长（存档用，forward 不使用）

    def forward(self, x):  # 前向：两卷积 + 捷径相加（结构与经典 BasicBlock 相同，仅卷积带空洞率）
        identity = x  # 保存输入作为捷径分支

        out = self.conv1(x)  # 3×3 卷积①（stride>1 时 H、W 减半）
        out = self.bn1(out)  # BN①
        out = self.relu(out)  # ReLU①

        out = self.conv2(out)  # 3×3 卷积②（空洞率取本 stage 值）
        out = self.bn2(out)  # BN②

        if self.downsample is not None:  # 通道/分辨率不匹配时启用投影捷径
            identity = self.downsample(x)  # 1×1 卷积 + BN 对齐捷径形状

        out += identity  # 残差相加：F(x) + x
        out = self.relu(out)  # 相加之后再激活

        return out  # 返回 (N, planes*1, H/stride, W/stride)


class BottleneckV1b(nn.Module):  # v1b 瓶颈残差块：1×1→3×3→1×1，3×3 卷积支持空洞率（resnet50+ 的 v1b/v1s 使用）
    expansion = 4  # 输出通道 = planes × 4（末个 1×1 卷积升维 4 倍）

    # ----------------------- BottleneckV1b.__init__ 参数说明 -----------------------
    # inplanes：块输入通道数；planes：中间瓶颈通道数（输出为 planes*4）
    # stride：下采样步长（放在 conv2 的 3×3 上，v1b 的关键改动）
    # dilation：conv2 的空洞率（dilated stage 首块传上一级空洞率，后续块传本级空洞率）
    # downsample：捷径投影子模块；previous_dilation：上一级空洞率【注意】本类中从未被使用（仅为对齐 gluon-cv 接口而保留）
    # norm_layer：归一化层构造函数
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=nn.BatchNorm2d):  # 构造支持空洞卷积的三卷积瓶颈块
        super(BottleneckV1b, self).__init__()  # 调用父类构造函数
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)  # 1×1 卷积①：通道降维（如 256→64）
        self.bn1 = norm_layer(planes)  # BN①
        self.conv2 = nn.Conv2d(planes, planes, 3, stride,  # 3×3 卷积②：位置参数 kernel=3、stride（v1b 下采样位置）
                               dilation, dilation, bias=False)  # padding=dilation、dilation=dilation：空洞卷积且分辨率不变
        self.bn2 = norm_layer(planes)  # BN②
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)  # 1×1 卷积③：通道升维 4 倍（64→256）
        self.bn3 = norm_layer(planes * self.expansion)  # BN③：残差分支最后一个 BN
        self.relu = nn.ReLU(True)  # ReLU 激活（位置参数 inplace=True）
        self.downsample = downsample  # 保存捷径投影
        self.stride = stride  # 记录步长（存档用）

    def forward(self, x):  # 前向：三卷积瓶颈结构 + 捷径相加
        identity = x  # 保存输入作为捷径分支

        out = self.conv1(x)  # 1×1 降维：(N, inplanes, H, W) → (N, planes, H, W)
        out = self.bn1(out)  # BN①
        out = self.relu(out)  # ReLU①

        out = self.conv2(out)  # 3×3 空间卷积（stride>1 时 H、W 减半；dilated 模式下空洞率扩感受野）
        out = self.bn2(out)  # BN②
        out = self.relu(out)  # ReLU②

        out = self.conv3(out)  # 1×1 升维：(N, planes, H, W) → (N, planes*4, H, W)
        out = self.bn3(out)  # BN③

        if self.downsample is not None:  # 尺寸/通道不匹配时启用投影捷径
            identity = self.downsample(x)  # 1×1 卷积 + BN 对齐捷径形状

        out += identity  # 残差相加：F(x) + x
        out = self.relu(out)  # 相加之后再激活

        return out  # 返回 (N, planes*4, H/stride, W/stride)


class ResNetV1b(nn.Module):  # v1b 主体：stem（可选 deep_stem）+ 4 个残差 stage + 分类头，支持 dilated 输出步长 8

    # ----------------------- ResNetV1b.__init__ 参数说明 -----------------------
    # block：残差块类型（BasicBlockV1b 或 BottleneckV1b）；layers：四 stage 块数列表
    # num_classes：分类头类别数（作分割编码器时不使用该头）
    # dilated：True（默认）时 stage3/4 改 stride=1 并用空洞率 2/4 → 输出步长 8，分割默认配置
    # deep_stem：True 时用三个 3×3 卷积替换 7×7 stem（v1s 变体，输出 128 通道）
    # zero_init_residual：残差分支末 BN 零初始化开关；norm_layer：归一化层构造函数
    def __init__(self, block, layers, num_classes=1000, dilated=True, deep_stem=False,
                 zero_init_residual=False, norm_layer=nn.BatchNorm2d):  # 按配置构建 ResNetV1b/V1s
        self.inplanes = 128 if deep_stem else 64  # 通道游标初值：deep_stem 的 stem 输出 128 通道，普通 stem 输出 64【注意】在 super().__init__() 之前给 self 赋普通属性，依赖 nn.Module.__setattr__ 对非参数/模块值的兜底路径，能正常运行但写法不规范
        super(ResNetV1b, self).__init__()  # 调用父类构造函数（初始化模块树）
        if deep_stem:  # v1s 路径：深层 stem，用小卷积替换大核 7×7
            self.conv1 = nn.Sequential(  # stem 为三个 3×3 卷积的顺序容器
                nn.Conv2d(3, 64, 3, 2, 1, bias=False),  # 3×3 卷积①：stride=2 下采样 (N,3,H,W) → (N,64,H/2,W/2)
                norm_layer(64),  # BN①
                nn.ReLU(True),  # ReLU①
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # 3×3 卷积②：stride=1，通道不变
                norm_layer(64),  # BN②
                nn.ReLU(True),  # ReLU②
                nn.Conv2d(64, 128, 3, 1, 1, bias=False)  # 3×3 卷积③：升维到 128 (N,64,…) → (N,128,H/2,W/2)
            )  # deep_stem 三连 3×3 卷积 Sequential 定义结束
        else:  # 常规 v1b 路径：单层 7×7 stem
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)  # 7×7 卷积：kernel=7、stride=2、padding=3，(N,3,H,W) → (N,64,H/2,W/2)
        self.bn1 = norm_layer(self.inplanes)  # stem 后的 BN（通道数随 deep_stem 取 128/64）
        self.relu = nn.ReLU(True)  # ReLU 激活
        self.maxpool = nn.MaxPool2d(3, 2, 1)  # 3×3 最大池化 stride=2：分辨率累计降至 1/4
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)  # stage1：64 通道、stride=1 → 输出 64*E，分辨率 1/4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)  # stage2：128 通道、stride=2 → 分辨率 1/8
        if dilated:  # 空洞模式（本文件默认，分割标准配置）：stage3/4 不再下采样，改用空洞卷积
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer)  # stage3：stride 由 2 改写为 1、空洞率 2 → 仍保持 1/8 分辨率
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer)  # stage4：stride=1、空洞率 4 → 1/8 分辨率且感受野最大，输出 512*E
        else:  # 常规分类模式：逐级下采样到 1/32
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)  # stage3：stride=2 → 分辨率 1/16
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)  # stage4：stride=2 → 分辨率 1/32
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化（分类头专用；分割作编码器时不走）
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 分类全连接层：512*E → num_classes

        for m in self.modules():  # 遍历所有子模块统一初始化
            if isinstance(m, nn.Conv2d):  # 卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming 正态初始化（适配 ReLU）
            elif isinstance(m, nn.BatchNorm2d):  # 批归一化层
                nn.init.constant_(m.weight, 1)  # γ 初始化为 1
                nn.init.constant_(m.bias, 0)  # β 初始化为 0

        if zero_init_residual:  # 可选：残差分支末 BN 零初始化
            for m in self.modules():  # 再次遍历所有模块
                if isinstance(m, BottleneckV1b):  # 瓶颈块 → 置零 bn3
                    nn.init.constant_(m.bn3.weight, 0)  # γ=0 → 初始时残差分支输出 0，块退化为恒等映射
                elif isinstance(m, BasicBlockV1b):  # 基础块 → 置零 bn2
                    nn.init.constant_(m.bn2.weight, 0)  # 同上

    # ----------------------- _make_layer 参数说明 -----------------------
    # block：残差块类型；planes：本 stage 基准通道数；blocks：块数量
    # stride：首块步长（dilated 模式的 stage3/4 传 1）；dilation：本 stage 空洞率（1/2/4）
    # norm_layer：归一化层构造函数
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):  # 堆叠一个 stage（含空洞率渐进过渡策略）
        downsample = None  # 捷径投影，默认 None（恒等捷径）
        if stride != 1 or self.inplanes != planes * block.expansion:  # 需要下采样或通道数变化时
            downsample = nn.Sequential(  # 捷径投影 = 1×1 卷积 + BN
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),  # 1×1 卷积：对齐通道并按 stride 下采样（1×1 卷积空洞率无意义，捷径不受 dilate 影响）
                norm_layer(planes * block.expansion),  # 投影后的 BN
            )  # 捷径投影 Sequential 定义结束

        layers = []  # 本 stage 的残差块列表
        if dilation in (1, 2):  # 空洞率为 1 或 2 的 stage（stage1/2/3）：首块沿用上一级空洞率
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample,  # 首块：3×3 卷积空洞率传 1（= 上一 stage 的空洞率，从旧率平滑过渡）
                                previous_dilation=dilation, norm_layer=norm_layer))  # previous_dilation=本级空洞率（仅 BasicBlockV1b 的 conv2 使用）
        elif dilation == 4:  # 空洞率为 4 的 stage（dilated 模式的 stage4）
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample,  # 首块：3×3 卷积空洞率传 2（= 上一 stage 即 stage3 的空洞率）
                                previous_dilation=dilation, norm_layer=norm_layer))  # previous_dilation=4（同样仅 BasicBlockV1b 会用到）
        else:  # 传入了未知空洞率
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))  # 抛运行时错误：空洞率只允许 1/2/4
        self.inplanes = planes * block.expansion  # 更新通道游标为本 stage 输出通道
        for _ in range(1, blocks):  # 其余 blocks-1 个常规块
            layers.append(block(self.inplanes, planes, dilation=dilation,  # 常规块：空洞率取本级 dilation（BottleneckV1b 的 3×3 卷积用它）
                                previous_dilation=dilation, norm_layer=norm_layer))  # previous_dilation 同为本级空洞率

        return nn.Sequential(*layers)  # 将块列表串成顺序容器返回

    # ----------------------- forward 参数说明 -----------------------
    # x：输入图像张量 (N, 3, H, W)；返回：分类 logits (N, num_classes)
    # 形状以 dilated=True、BottleneckV1b（E=4）为例；dilated=False 时 stage3/4 分辨率再各减半
    def forward(self, x):  # 前向：stem → 4 stage → 池化 → 分类（分割场景只前向到 layer4 取特征）
        x = self.conv1(x)  # stem（7×7 或 deep_stem 三连 3×3）：(N, 3, H, W) → (N, 64/128, H/2, W/2)
        x = self.bn1(x)  # stem 后 BN（形状不变）
        x = self.relu(x)  # ReLU（形状不变）
        x = self.maxpool(x)  # 最大池化 stride=2 → (N, 64/128, H/4, W/4)

        x = self.layer1(x)  # stage1（stride=1）→ (N, 256, H/4, W/4)（BasicBlock 为 64 通道）
        x = self.layer2(x)  # stage2（stride=2）→ (N, 512, H/8, W/8)
        x = self.layer3(x)  # stage3：dilated=True（stride=1、空洞率 2）→ (N, 1024, H/8, W/8)；dilated=False → (N, 1024, H/16, W/16)
        x = self.layer4(x)  # stage4：dilated=True（stride=1、空洞率 4）→ (N, 2048, H/8, W/8)；dilated=False → (N, 2048, H/32, W/32)

        x = self.avgpool(x)  # 全局平均池化 → (N, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # 展平 → (N, 2048)
        x = self.fc(x)  # 全连接 → (N, num_classes)

        return x  # 返回分类 logits


# 工厂函数：构建 ResNet-18 v1b（BasicBlockV1b × [2,2,2,2]，默认 dilated=True）
# pretrained=True 时下载 torchvision ImageNet 权重，按 key 过滤合并后加载（等效 strict=False）
def resnet18_v1b(pretrained=False, **kwargs):  # **kwargs 透传给 ResNetV1b（如 dilated、norm_layer）
    model = ResNetV1b(BasicBlockV1b, [2, 2, 2, 2], **kwargs)  # 用基础块、每 stage 2 块构建 18 层 v1b
    if pretrained:  # 加载预训练权重
        old_dict = model_zoo.load_url(model_urls['resnet18'])  # 下载 torchvision 的 ResNet-18 权重（OrderedDict）【注意】load_url 为废弃 API
        model_dict = model.state_dict()  # 取当前模型的参数字典
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}  # 只保留 key 与模型匹配的预训练项（num_classes 不同时 fc.* 会被过滤掉）【注意】仅按 key 名过滤，不校验形状
        model_dict.update(old_dict)  # 用过滤后的预训练权重覆盖模型参数字典
        model.load_state_dict(model_dict)  # 加载合并后的字典（key 均来自模型自身，默认 strict=True 亦能通过）
    return model  # 返回模型


# 工厂函数：构建 ResNet-34 v1b（BasicBlockV1b × [3,4,6,3]，默认 dilated=True）
# pretrained=True 时下载 torchvision ImageNet 权重，按 key 过滤合并后加载（等效 strict=False）
def resnet34_v1b(pretrained=False, **kwargs):  # **kwargs 透传给 ResNetV1b
    model = ResNetV1b(BasicBlockV1b, [3, 4, 6, 3], **kwargs)  # 用基础块、块数 [3,4,6,3] 构建 34 层 v1b
    if pretrained:  # 加载预训练权重
        old_dict = model_zoo.load_url(model_urls['resnet34'])  # 下载 torchvision 的 ResNet-34 权重【注意】load_url 为废弃 API
        model_dict = model.state_dict()  # 取当前模型参数字典
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}  # 按 key 过滤预训练项（fc 等不匹配项被剔除）
        model_dict.update(old_dict)  # 合并进模型参数字典
        model.load_state_dict(model_dict)  # 加载合并后的字典
    return model  # 返回模型


# 工厂函数：构建 ResNet-50 v1b（BottleneckV1b × [3,4,6,3]，默认 dilated=True）
# pretrained=True 时下载 torchvision ImageNet 权重，按 key 过滤合并后加载（等效 strict=False）
def resnet50_v1b(pretrained=False, **kwargs):  # **kwargs 透传给 ResNetV1b
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], **kwargs)  # 用瓶颈块、块数 [3,4,6,3] 构建 50 层 v1b
    if pretrained:  # 加载预训练权重
        old_dict = model_zoo.load_url(model_urls['resnet50'])  # 下载 torchvision 的 ResNet-50 权重【注意】load_url 为废弃 API
        model_dict = model.state_dict()  # 取当前模型参数字典
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}  # 按 key 过滤预训练项（num_classes 不同时 fc.* 被剔除）
        model_dict.update(old_dict)  # 合并进模型参数字典
        model.load_state_dict(model_dict)  # 加载合并后的字典
    return model  # 返回模型


# 工厂函数：构建 ResNet-101 v1b（BottleneckV1b × [3,4,23,3]，默认 dilated=True）
# pretrained=True 时下载 torchvision ImageNet 权重，按 key 过滤合并后加载（等效 strict=False）
def resnet101_v1b(pretrained=False, **kwargs):  # **kwargs 透传给 ResNetV1b
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], **kwargs)  # 用瓶颈块、块数 [3,4,23,3] 构建 101 层 v1b
    if pretrained:  # 加载预训练权重
        old_dict = model_zoo.load_url(model_urls['resnet101'])  # 下载 torchvision 的 ResNet-101 权重【注意】load_url 为废弃 API
        model_dict = model.state_dict()  # 取当前模型参数字典
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}  # 按 key 过滤预训练项
        model_dict.update(old_dict)  # 合并进模型参数字典
        model.load_state_dict(model_dict)  # 加载合并后的字典
    return model  # 返回模型


# 工厂函数：构建 ResNet-152 v1b（BottleneckV1b × [3,8,36,3]，默认 dilated=True）
# pretrained=True 时下载 torchvision ImageNet 权重，按 key 过滤合并后加载（等效 strict=False）
def resnet152_v1b(pretrained=False, **kwargs):  # **kwargs 透传给 ResNetV1b
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], **kwargs)  # 用瓶颈块、块数 [3,8,36,3] 构建 152 层 v1b
    if pretrained:  # 加载预训练权重
        old_dict = model_zoo.load_url(model_urls['resnet152'])  # 下载 torchvision 的 ResNet-152 权重【注意】load_url 为废弃 API
        model_dict = model.state_dict()  # 取当前模型参数字典
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}  # 按 key 过滤预训练项
        model_dict.update(old_dict)  # 合并进模型参数字典
        model.load_state_dict(model_dict)  # 加载合并后的字典
    return model  # 返回模型


# 工厂函数：构建 ResNet-50 v1s（deep_stem=True + 默认 dilated=True），segbase.py 的默认分割骨干
# pretrained=True 时经 model_store.get_resnet_file 从 encoding S3 仓库下载 gluon 转换权重，strict=False 宽松加载
def resnet50_v1s(pretrained=False, root='~/.torch/models', **kwargs):  # root：权重缓存目录（get_resnet_file 内部会 expanduser 展开 ~）
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, **kwargs)  # 构建 50 层、深 stem 的 v1s（stem 输出 128 通道）
    if pretrained:  # 加载预训练权重
        from ..model_store import get_resnet_file  # 延迟导入：按名称定位/下载权重文件（含 SHA1 校验，返回本地 .pth 路径）
        model.load_state_dict(torch.load(get_resnet_file('resnet50', root=root)), strict=False)  # torch.load 反序列化后加载；strict=False 容忍 fc/命名差异
    return model  # 返回模型


# 工厂函数：构建 ResNet-101 v1s（deep_stem=True + 默认 dilated=True）
# pretrained=True 时经 model_store.get_resnet_file 从 encoding S3 仓库下载 gluon 转换权重，strict=False 宽松加载
def resnet101_v1s(pretrained=False, root='~/.torch/models', **kwargs):  # root：权重缓存目录
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, **kwargs)  # 构建 101 层、深 stem 的 v1s
    if pretrained:  # 加载预训练权重
        from ..model_store import get_resnet_file  # 延迟导入权重定位/下载工具
        model.load_state_dict(torch.load(get_resnet_file('resnet101', root=root)), strict=False)  # 加载 gluon 风格权重（strict=False）
    return model  # 返回模型


# 工厂函数：构建 ResNet-152 v1s（deep_stem=True + 默认 dilated=True）
# pretrained=True 时经 model_store.get_resnet_file 从 encoding S3 仓库下载 gluon 转换权重，strict=False 宽松加载
def resnet152_v1s(pretrained=False, root='~/.torch/models', **kwargs):  # root：权重缓存目录
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], deep_stem=True, **kwargs)  # 构建 152 层、深 stem 的 v1s
    if pretrained:  # 加载预训练权重
        from ..model_store import get_resnet_file  # 延迟导入权重定位/下载工具
        model.load_state_dict(torch.load(get_resnet_file('resnet152', root=root)), strict=False)  # 加载 gluon 风格权重（strict=False）
    return model  # 返回模型


if __name__ == '__main__':  # 直接运行本文件时的简易自测入口
    import torch  # 导入 torch 构造随机输入张量

    img = torch.randn(4, 3, 224, 224)  # 构造 batch=4 的随机图像 (4, 3, 224, 224)
    model = resnet50_v1b(True)  # 构建 resnet50_v1b 并加载 ImageNet 预训练权重【注意】会联网下载权重，离线环境将失败
    output = model(img)  # 前向推理 → 输出分类 logits (4, 1000)
