# -*- coding: utf-8 -*-
"""
可微神经网络代理模型（ProteoFlow / UMA 范式）——用于与白盒 ODE 形成对照。

这是本项目的「另一面」。用户的根本问题是：
    同样是 AI4Science 模型，为什么 ProteoFlow / UMA 能用梯度下降(自动微分)训练，
    而 T-World 只能靠多目标遗传算法(MOGA)？

答案就落在「参数化方式」上：
  - ProteoFlow / UMA：把模型参数化成一个光滑可微的神经网络（流匹配速度场 / 等变图网络），
    处处可微、梯度良好 -> 梯度下降直接可用，损失平滑下降。
  - T-World：模型是一套手写的刚性白盒 ODE，参数 eps 是极小时间尺度，
    梯度对其敏感度是 1/eps² 量级，且解在跨越兴奋阈值/分岔时发生突变 -> 梯度下降失效。

本文件用同一个任务（拟合同一条电压轨迹）做最小对照：
  用一个小 MLP 直接学习 (t, 刺激) -> V(t) 的映射，用 Adam 训练。
  结果：损失平滑下降并收敛到接近 0，与 gd.py 中白盒 ODE 的「卡死 / NaN 发散」形成鲜明对比。

（注：ProteoFlow 学的是「速度场 + ODE 积分」这一动力学版本；本文件用其最简单的
  回归版本传达同一个原理——可微参数化让梯度下降可用，且训练快、无需小步长。）
"""
import numpy as np
import torch

import config


class SurrogateNet(torch.nn.Module):
    """2 -> 1 的 MLP：输入 (t, 刺激) 输出 V(t)。光滑可微，梯度良好。"""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, t_norm, stim):
        x = torch.stack([t_norm, stim], dim=-1)
        return self.net(x).squeeze(-1)


def run_neural(t, y_clean, epochs=None, lr=1e-3, seed=None, verbose=True):
    """
    用 MLP 代理模型 + Adam 梯度下降拟合电压轨迹，返回字典：
      - loss_history : 标量损失轨迹（应平滑下降并收敛到接近 0）
      - final_V      : 拟合轨迹（与 t 同网格）
      - final_loss   : 最终 MSE
    """
    epochs = epochs or config.GD_EPOCHS
    torch.manual_seed(seed if seed is not None else config.GD_SEED)

    stim = ((t % config.PACE_CL) < config.PACE_DUR).astype(np.float32)
    t_norm = (t / config.T_END).astype(np.float32)

    y_t = torch.as_tensor(y_clean, dtype=torch.float32)
    t_t = torch.as_tensor(t_norm, dtype=torch.float32)
    stim_t = torch.as_tensor(stim, dtype=torch.float32)

    net = SurrogateNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    loss_history = np.zeros(epochs)
    final_V = None

    for ep in range(epochs):
        pred = net(t_t, stim_t)
        loss = torch.mean((pred - y_t) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history[ep] = loss.item()

        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  神经代理 epoch {ep:4d}/{epochs}: loss={loss.item():.6f}")

    with torch.no_grad():
        final_V = net(t_t, stim_t).numpy()

    return {
        "loss_history": loss_history,
        "final_V": final_V,
        "final_loss": float(loss_history[-1]),
        "n_params": sum(p.numel() for p in net.parameters()),
    }