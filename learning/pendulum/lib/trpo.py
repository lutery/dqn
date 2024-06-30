import numpy as np

import torch


def get_flat_params_from(model):
    '''
    获取网络的参数，将网络参数展平为一维
    返回
    '''

    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    '''
    将更新后的网络参数设置到网路参数中，等于更新网路参数
    todo 这里为什么要使用这种方式更新梯度？
    '''

    # 因为网路的参数已经展平为一维，所以这里需要使用一个索引来进行查询当前更新到了哪个参数
    # prev_ind：flat_params起始索引，flat_size当前网络层的参数数量
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10, device="cpu"):
    '''
    todo 作用 书P318
    这个函数实现了共轭梯度法（Conjugate Gradient Method），这是一种用于解决线性方程组的迭代方法，特别是在形如 Ax = b 的方程中，其中 A 是一个对称正定矩阵。在强化学习中的TRPO（Trust Region Policy Optimization）算法里，共轭梯度法用于近似求解Fisher信息矩阵的逆

    接收参数 Avp（一个函数，代表矩阵向量积 A*v）
    b（方程组右侧的向量）
    nsteps（迭代次数上限）
    residual_tol（残差容忍度，默认值为1e-10）
    device（计算设备，默认为CPU）。
    '''
    # 创建一个和b相同维度的全零张量
    # todo 在书中找到对应的公式
    x = torch.zeros(b.size()).to(device)
    # 初始化残差向量 r 和方向向量 p，它们的初始值都是 b 的副本
    r = b.clone()
    p = b.clone()
    # 计算残差向量的点积，即 r 和自身的点积，用于后续计算
    rdotr = torch.dot(r, r)
    # 开始迭代，最多进行 nsteps 次
    for i in range(nsteps):
        # 计算 A*v，即矩阵 A 和向量 p 的乘积
        _Avp = Avp(p)
        # 计算步长 alpha，它是残差向量的点积与 p 和 _Avp 的点积的比值
        alpha = rdotr / torch.dot(p, _Avp)
        # 更新解向量 x
        x += alpha * p
        # 更新残差向量 r
        r -= alpha * _Avp
        # 计算新的残差向量的点积
        new_rdotr = torch.dot(r, r)
        # 计算比值 betta，用于更新方向向量 p
        betta = new_rdotr / rdotr
        # 更新方向向量 p
        p = r + betta * p
        # 更新残差点积的值，用于下一轮迭代
        rdotr = new_rdotr
        # 如果残差点积小于容忍度，则提前终止迭代
        if rdotr < residual_tol:
            break
    return x


def linesearch(model,
               f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f().data
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        # 这里相当于是对各种可能的更新参数进行尝试，直到找到一个满足条件的参数
        # 有点像自动阈值的二值化方法，通过迭代所有阈值，找到一个合适的阈值
        # 这里也是通过迭代所有的参数，找到一个合适的参数
        # 所以这里需要使用set_flat_params_to更新参数，而不是通常的
        # Adam优化器的方式更新参数
        xnew = x + fullstep * stepfrac
        set_flat_params_to(model, xnew)
        newfval = f().data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            return True, xnew
    return False, x


def trpo_step(model, get_loss, get_kl, max_kl, damping, device="cpu"):
    '''
    todo 找到这边的数学公式原理进行理解

    这段代码实现了TRPO（Trust Region Policy Optimization）算法中的一个关键步骤：TRPO优化步骤。TRPO是一种高级强化学习算法，用于优化策略，同时保持策略更新的稳定性
    '''

    # 获取执行动作的概率与获取优势之间的损失
    loss = get_loss()
    # 计算梯度
    grads = torch.autograd.grad(loss, model.parameters())
    # 将梯度展平为一维
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v):
        '''
        这里的v就是loss_grad和stepdir

        这段代码实现了TRPO（Trust Region Policy Optimization）算法中的一个关键步骤，名为 Fisher-Vector Product (FVP)。FVP 是用来近似计算 Hessian（海森矩阵）的逆，这在优化算法中是非常有用的，特别是在需要高效率地处理大规模问题时。下面是对这个函数中每一行代码的详细解释
        '''
        # 计算策略更新前后策略对于预测动作差异的KL散度
        # 调用 get_kl 函数，计算KL散度。这通常用于衡量两个概率分布（在这里是当前策略和更新前策略）之间的差异
        # todo 这里的kl散度是如何控制网络的更新不离原来的策略太远的？
        kl = get_kl()
        # 计算KL散度的均值。这是因为 get_kl 为每个状态返回一个KL值，而这里我们需要的是整体平均值
        kl = kl.mean()

        # 根据kl散度计算KL散度的梯度
        # 使用PyTorch的自动微分功能计算关于模型参数的KL散度的梯度。create_graph=True 参数使得这些梯度可以在之后再次被微分，这是为了计算二阶导数
        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        # 展平kl散度的梯度
        # 将梯度展平并连接成一个单一的向量。这是为了方便后续的计算
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        v_v = torch.tensor(v).to(device)
        # 计算 flat_grad_kl（KL散度的梯度）和 v_v 的点积。这是Fisher信息矩阵与向量 v 的乘积的一部分
        kl_v = (flat_grad_kl * v_v).sum()
        # 再次计算梯度，这次是关于 kl_v。这实际上是在计算Hessian向量积的一部分
        grads = torch.autograd.grad(kl_v, model.parameters())
        # 将这些梯度展平并连接成一个向量。这个向量是Fisher信息矩阵与 v 的乘积的近似
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        # 返回Fisher向量积的结果，并加上一个阻尼项（v * damping）。这个阻尼项有助于保证数值稳定性
        return flat_grad_grad_kl + v * damping

    # 使用共轭梯度方法计算优化方向。这里使用Fvp作为Hessian向量积的近似，-loss_grad作为共轭梯度方法的输入。
    # todo 共轭梯度是如何控制网络的更新的？
    stepdir = conjugate_gradients(Fvp, -loss_grad, 10, device=device)

    # 计算步长的平方和（step direction squared）。
    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

    # 计算拉格朗日乘子lm，用于控制步长，使KL散度的变化不超过max_kl
    lm = torch.sqrt(shs / max_kl)
    # 计算完整的优化步长
    fullstep = stepdir / lm[0]

    # 计算损失梯度与步长方向的点积的负数
    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)

    # 获取模型当前的参数
    prev_params = get_flat_params_from(model)
    # 使用线搜索方法找到一个满足KL散度约束且减少损失的新参数集
    success, new_params = linesearch(model, get_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])
    # 更新模型参数为新找到的参数集
    set_flat_params_to(model, new_params)

    # 返回计算得到的损失值
    return loss
