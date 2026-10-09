# -*- coding: utf-8 -*-
"""
白盒 ODE 模型：FitzHugh-Nagumo(FHN) 型可兴奋系统。

这是 T-World 的「极简等效替身」：
  - 白盒 ODE：由显式微分方程描述（类比 T-World 用常微分方程组描述离子电流/通量/信号）。
  - 刚性(stiffness)：eps << 1，快变量 V 与慢变量 w 时间尺度相差约 1/eps 倍。
  - 阈值/可兴奋性：刺激是否跨越阈值决定了「发放/不发放」这一离散结果，
    类比心肌细胞「发作/不发作 EAD、DAD、alternans」等二值化行为。

提供两套实现：
  - numpy 版：simulate / simulate_batch（供遗传算法批量评估用，向量化到种群维度）。
  - torch 版：simulate_torch（供梯度下降做可微前向，损失经整个轨迹反传）。
"""
import numpy as np


def _pacing(t, pace_cl, pace_dur):
    """脉冲刺激电流：周期 pace_cl，每个周期前 pace_dur 时间内为 1，其余 0。"""
    return (np.mod(t, pace_cl) < pace_dur).astype(np.float64)


def fhn_simulate(theta, t, config):
    """
    用固定步长前向欧拉积分 FHN，返回电压轨迹 V(t)。

    参数 theta：numpy 数组 [eps, a, b, amp]。
    """
    eps, a, b, amp = float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])
    dt = config.DT
    n = len(t)
    V = np.zeros(n)
    w = np.zeros(n)

    # 初始状态取静息状态附近
    V[0] = -1.4
    w[0] = V[0] - V[0] ** 3 / 3.0

    I = amp * _pacing(t, config.PACE_CL, config.PACE_DUR)

    for i in range(n - 1):
        dV = (V[i] - V[i] ** 3 / 3.0 - w[i] + I[i]) / max(eps, 1e-6)
        dw = V[i] + a - b * w[i]
        V[i + 1] = np.clip(V[i] + dt * dV, -10.0, 10.0)
        w[i + 1] = np.clip(w[i] + dt * dw, -10.0, 10.0)

    return V


def fhn_simulate_batch(theta_arr, t, config):
    """
    批量版：theta_arr 形状 (P, 4)。同时积分 P 个候选个体，返回 (P, n) 的轨迹。
    时间维串行、种群维向量化——遗传算法评估全体种群时只需一趟时间循环。
    """
    P = theta_arr.shape[0]
    eps = theta_arr[:, 0]
    a = theta_arr[:, 1]
    b = theta_arr[:, 2]
    amp = theta_arr[:, 3]

    dt = config.DT
    n = len(t)

    V = np.full(P, -1.4)
    w = V - V ** 3 / 3.0

    pacing = _pacing(t, config.PACE_CL, config.PACE_DUR)  # (n,)

    traj = np.zeros((P, n))
    traj[:, 0] = V

    eps_safe = np.maximum(eps, 1e-6)

    for i in range(n - 1):
        I = amp * pacing[i]
        dV = (V - V ** 3 / 3.0 - w + I) / eps_safe
        dw = V + a - b * w
        V = np.clip(V + dt * dV, -10.0, 10.0)
        w = np.clip(w + dt * dw, -10.0, 10.0)
        traj[:, i + 1] = V

    return traj


def count_spikes(V):
    """统计 V 上穿 SPIKE_THRESHOLD 的次数（即动作电位/尖峰发放次数）。"""
    thr = 0.0
    return int(np.sum((V[:-1] < thr) & (V[1:] >= thr)))


def count_spikes_multi(traj):
    """对 (P, n) 的轨迹矩阵逐行统计发放次数，返回 (P,) 整数数组。"""
    thr = 0.0
    up = (traj[:, :-1] < thr) & (traj[:, 1:] >= thr)
    return up.sum(axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# torch 版（供梯度下降）
# ---------------------------------------------------------------------------
def fhn_simulate_torch(theta, t, config):
    """
    可微前向：theta 为 torch 张量 [eps, a, b, amp]，返回 V 轨迹 (n,) 张量。

    注意：这里用 torch.relu 会把 eps 约束为正（防止 eps<=0 导致数值爆炸），
    这本身就是要展示的一点——GD 必须人工加硬约束才能勉强维持数值稳定，
    而 GA 天然只需要把参数限制在搜索边界内即可。
    """
    import torch

    eps = torch.relu(theta[0]) + 1e-4
    a = theta[1]
    b = theta[2]
    amp = theta[3]

    dt = config.DT
    n = len(t)

    t_t = torch.as_tensor(t, dtype=torch.float32)
    pacing = ((t_t % config.PACE_CL) < config.PACE_DUR).float()  # (n,)

    v = torch.tensor(-1.4, dtype=torch.float32)
    w = torch.tensor(-1.4 - (-1.4) ** 3 / 3.0, dtype=torch.float32)

    Vs = []
    for i in range(n):
        I = amp * pacing[i]
        dV = (v - v ** 3 / 3.0 - w + I) / eps
        dw = v + a - b * w
        Vs.append(v)
        v = v + dt * dV
        w = w + dt * dw

    return torch.stack(Vs)