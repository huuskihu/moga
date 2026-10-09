# -*- coding: utf-8 -*-
"""
梯度下降求解器（PyTorch Adam）。

关键点：GD 是「单目标」优化器，只能最小化一个标量。而我们的目标是「向量」，
所以 GD 被迫把多目标强行加权成标量。这里 GD 只优化「连续可导的 MSE」，
无法直接处理「发放/不发放」的阶跃目标 f2（其梯度处处为 0 或不存在）。

预期失败模式（与遗传算法对比）：
  1. 从「亚阈值」坏初值出发，MSE 的梯度对「要不要发放」不提供有效信号，
     GD 会收敛到「最优的亚阈值拟合」这个局部极小，永远发不出尖峰。
  2. 刚性(eps 很小)会使梯度幅度剧烈波动，Loss 曲线震荡甚至发散。
  —— 这正是 T-World 这类白盒 ODE 模型弃用 GD、改用 MOGA 的数学原因。
"""
import os
import numpy as np
import torch

import config
from model import fhn_simulate_torch
from objective import objectives_single


def run_gd(t, y_true, n_spikes_true, init="bad", lr=None, epochs=None,
           seed=None, verbose=True):
    """
    运行梯度下降拟合参数，返回字典包含：
      - theta_history   : (epochs, 4) 参数轨迹
      - loss_history    : (epochs,) 标量损失轨迹
      - final_theta     : 最终参数
      - final_objectives: 最终目标的向量 [f_mse, f_fire]（用与 GA 相同的目标函数评估）
      - final_V         : 最终参数对应的电压轨迹
    参数 init: 'bad' 或 'good'，对应 config 中的两种初值。
    """
    lr = lr or config.GD_LR
    epochs = epochs or config.GD_EPOCHS
    torch.manual_seed(seed if seed is not None else config.GD_SEED)

    init_dict = config.GD_INIT_BAD if init == "bad" else config.GD_INIT_GOOD
    theta0 = config.theta_to_vec(init_dict)

    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    y_true_t = torch.as_tensor(y_true, dtype=torch.float32)

    optimizer = torch.optim.Adam([theta], lr=lr)

    theta_history = np.zeros((epochs, 4))
    loss_history = np.zeros(epochs)

    for ep in range(epochs):
        optimizer.zero_grad()
        V = fhn_simulate_torch(theta, t, config)
        # 标量化：仅优化连续可导的 MSE（f2 阶跃项无法求导，被 GD 忽略）
        loss = torch.mean((V - y_true_t) ** 2)
        loss.backward()
        optimizer.step()

        theta_history[ep] = theta.detach().numpy()
        loss_history[ep] = loss.item()

        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  GD epoch {ep:4d}/{epochs}: loss={loss.item():.5f}, "
                  f"theta={theta.detach().numpy().round(4)}")

    final_theta = theta.detach().numpy()
    final_objectives = objectives_single(final_theta, t, y_true, n_spikes_true)

    # 用 numpy 重算最终轨迹（用于可视化和对比）
    from model import fhn_simulate
    final_V = fhn_simulate(final_theta, t, config)

    return {
        "theta_history": theta_history,
        "loss_history": loss_history,
        "final_theta": final_theta,
        "final_objectives": final_objectives,
        "final_V": final_V,
    }