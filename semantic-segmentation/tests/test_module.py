# ============================================================================
# 【文件说明】test_module.py —— core.nn.SyncBatchNorm（自实现同步 BN）的等价性单元测试
# 测试内容：验证 SyncBatchNorm 与标准 torch.nn.BatchNorm2d 在数值上等价：
#   1) 训练模式（train）：前向输出、bias/weight 梯度、输入梯度、running_mean 逐一比对；
#   2) 推理模式（eval + torch.no_grad）：前向输出比对（该分支当前被注释未启用）；
#   3) 经 torch.nn.DataParallel 包装（单卡场景）后 SyncBatchNorm 的行为。
# 测试输入：torch.rand(16, 10, 16, 16)（batch=16、通道=10、double 精度），随机重复 10 轮，
#   容差：rtol=1e-3、atol=1e-3。
# 【注意】本测试强制使用 CUDA（.cuda()），无 GPU 环境会直接报错；
#   结尾使用早已停止维护的 nose 框架（Python 3.10+ 无法运行，建议改用 pytest）。
# ============================================================================
import core  # 导入 core 包（被测组件 core.nn.SyncBatchNorm 经由它访问）
import torch  # PyTorch 主库：张量、nn 模块、autograd
import numpy as np  # NumPy：张量转数组后用 np.allclose 做数值比较

from torch.autograd import Variable  # 【注意】Variable 是 PyTorch 0.3 时代的废弃 API（现代写法直接用张量的 requires_grad=True），此处仅作带梯度的输入包装

EPS = 1e-3  # 相对误差容差 rtol（比较时 |a-b| <= atol + rtol*|b|）
ATOL = 1e-3  # 绝对误差容差 atol


# 形参说明: a/b 待比较的两个张量; atol 绝对容差（默认 ATOL）; rtol 相对容差（默认 EPS）
def _assert_tensor_close(a, b, atol=ATOL, rtol=EPS):  # 断言两个张量在容差范围内逐元素接近
    npa, npb = a.cpu().numpy(), b.cpu().numpy()  # 两个张量都移回 CPU 并转为 NumPy 数组
    # 断言 np.allclose 通过，否则抛出带差异统计的 AssertionError
    assert np.allclose(npa, npb, rtol=rtol, atol=atol), \
        'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(  # 失败信息模板：两张量内容及差值统计
            a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max())  # adiff=最大绝对差；rdiff=最大相对差（分母取下限 1e-5 防除零）


def testSyncBN():  # 主测试函数：SyncBatchNorm 与 BatchNorm2d 的等价性验证（函数名以 test 开头，可被 nose/pytest 收集）
    # 形参说明: bn1/bn2 待比较的两个 BN 模块（参照的普通 BN 与被测的同步 BN，可能被 DataParallel 包装）
    # 形参说明: input 共享的测试输入（变量名遮蔽内建 input，仅限局部作用域）; is_train 是否训练模式; cuda 是否把输入搬到 GPU
    def _check_batchnorm_result(bn1, bn2, input, is_train, cuda=False):  # 执行一次完整的前向（+反向）等价性检查
        # 形参说明: module 待搜索的模块（可能是 DataParallel 包装对象）
        def _find_bn(module):  # 在模块树中递归查找第一个 BN 层（用于穿透 DataParallel 拿到真实 BN）
            for m in module.modules():  # 递归遍历所有子模块
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,  # 匹配标准 BN1d/BN2d
                                  core.nn.SyncBatchNorm)):  # 或自实现的 SyncBatchNorm
                    return m  # 返回找到的第一个 BN 层；【注意】找不到时隐式返回 None，后续调用会抛 AttributeError

        # 形参说明: bn1/bn2 需要对齐参数的两个 BN 层
        def _syncParameters(bn1, bn2):  # 同步两个 BN 的仿射参数与滑动统计量，保证比较起点一致
            bn1.reset_parameters()  # 重置 bn1 的 weight/bias（按默认策略重新初始化）
            bn2.reset_parameters()  # 重置 bn2 的参数（两模块各自随机，需下面手动对齐）
            if bn1.affine and bn2.affine:  # 两个模块都启用仿射变换（weight/bias）时
                bn2.weight.data.copy_(bn1.weight.data)  # bn2.weight ← bn1.weight（对齐 gamma）
                bn2.bias.data.copy_(bn1.bias.data)  # bn2.bias ← bn1.bias（对齐 beta）
                bn2.running_mean.copy_(bn1.running_mean)  # 对齐滑动均值
                bn2.running_var.copy_(bn1.running_var)  # 对齐滑动方差

        bn1.train(mode=is_train)  # 按传入标志设置 bn1 的训练/推理模式
        bn2.train(mode=is_train)  # 同样设置 bn2（保持两者模式一致）

        if cuda:  # 需要 GPU 时
            input = input.cuda()  # 把输入移到默认 GPU 设备

        # using the same values for gamma and beta
        _syncParameters(_find_bn(bn1), _find_bn(bn2))  # 取出两模块内部的 BN 层并对齐参数（原注释：使用相同的 gamma/beta 值）

        input1 = Variable(input.clone().detach(), requires_grad=True)  # bn1 的输入副本（开启梯度）；【注意】Variable 已废弃，可直接用张量
        input2 = Variable(input.clone().detach(), requires_grad=True)  # bn2 的输入副本（开启梯度）
        if is_train:  # 训练模式分支：验证前向+反向
            bn1.train()  # 确认 bn1 处于训练模式（BN 用批统计并更新 running 统计）
            bn2.train()  # 确认 bn2 处于训练模式
            output1 = bn1(input1)  # bn1 前向
            output2 = bn2(input2)  # bn2 前向
        else:  # 推理模式分支：只验证前向
            bn1.eval()  # 切换 bn1 到 eval（BN 改用 running 统计，不再更新）
            bn2.eval()  # 切换 bn2 到 eval
            with torch.no_grad():  # 关闭梯度记录：推理无需反向，节省显存并加速
                output1 = bn1(input1)  # bn1 前向（无梯度）
                output2 = bn2(input2)  # bn2 前向（无梯度）
        # assert forwarding
        # _assert_tensor_close(input1.data, input2.data)
        _assert_tensor_close(output1.data, output2.data)  # 断言两个模块的前向输出一致
        if not is_train:  # 推理模式只验证前向
            return  # 直接返回，不做反向检查
        (output1 ** 2).sum().backward()  # 对 bn1 输出的平方和反向传播（人为构造标量损失）
        (output2 ** 2).sum().backward()  # 对 bn2 输出的平方和同样反向
        _assert_tensor_close(_find_bn(bn1).bias.grad.data, _find_bn(bn2).bias.grad.data)  # 断言两者 bias（beta）梯度一致
        _assert_tensor_close(_find_bn(bn1).weight.grad.data, _find_bn(bn2).weight.grad.data)  # 断言两者 weight（gamma）梯度一致
        _assert_tensor_close(input1.grad.data, input2.grad.data)  # 断言两者输入梯度一致
        _assert_tensor_close(_find_bn(bn1).running_mean, _find_bn(bn2).running_mean)  # 断言更新后的滑动均值一致
        # _assert_tensor_close(_find_bn(bn1).running_var, _find_bn(bn2).running_var)

    bn = torch.nn.BatchNorm2d(10).cuda().double()  # 参照模块：标准 BN2d，10 通道，移到 GPU 并转 double 精度；【注意】无 CUDA 的环境会在此直接报错
    sync_bn = core.nn.SyncBatchNorm(10, inplace=True, sync=True).cuda().double()  # 被测模块：自实现同步 BN（10 通道，inplace 激活，开启跨卡同步）
    sync_bn = torch.nn.DataParallel(sync_bn).cuda()  # 用 DataParallel 包装被测模块（单卡部署时 sync 退化为普通 BN 行为）
    # check with unsync version
    # _check_batchnorm_result(bn, sync_bn, torch.rand(2, 1, 2, 2).double(), True, cuda=True)
    for i in range(10):  # 用不同随机输入重复 10 轮等价性检查
        print(i)  # 打印当前轮次编号
        _check_batchnorm_result(bn, sync_bn, torch.rand(16, 10, 16, 16).double(), True, cuda=True)  # 训练模式：16x10x16x16 随机 double 输入，搬运到 GPU
        # _check_batchnorm_result(bn, sync_bn, torch.rand(16, 10, 16, 16).double(), False, cuda=True)


if __name__ == '__main__':  # 直接运行本文件时执行测试
    import nose  # 【注意】nose 测试框架早已停止维护，在 Python 3.10+ 因移除 collections.Callable 而无法导入，建议改用 pytest 运行本文件

    nose.runmodule()  # 收集并运行本模块中的测试函数（等价于 pytest 该文件）
