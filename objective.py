# -*- coding: utf-8 -*-
"""
多目标损失（目标向量）。

这是复现「GD vs GA」差异的核心。损失不能是单一标量，而是一个「向量」：
  f1 = 整条电压轨迹的均方误差 MSE（连续、可导）
  f2 = 「发放/不发放」的阶跃式惩罚（不可导、硬边界）

并叠加「参数合法性」硬约束（eps、b 必须为正；越界则两个目标都置为极大值）。

f2 的物理含义：心肌细胞里 EAD/DAD/alternans 是「发生 or 不发生」的二值事件，
只要发放次数不对，就整体判罚——这种目标对梯度下降是无信号的（导数为 0），
但遗传算法通过变异可以「跨越阈值」（从惩罚 1000 直接跳到 0）。
"""
import numpy as np

import config
from model import fhn_simulate_batch, count_spikes_multi

# 阶跃惩罚的「大小」：发放次数不一致时施加的大常数。
FIRE_PENALTY = 1000.0
# 参数越界(非法)时的目标值。
INFEASIBLE = 1e6


def _is_feasible(theta, lo, hi):
    """参数是否在搜索边界内（硬约束）。"""
    return bool(np.all(theta >= lo) and np.all(theta <= hi))


def objectives(theta_arr, t, y_true, n_spikes_true):
    """
    对 (P, 4) 的候选参数批量计算目标向量，返回 (P, 2)：
      结果[:, 0] = f_MSE       （连续，轨迹贴合程度）
      结果[:, 1] = f_fire      （阶跃，发放次数是否一致）
    """
    lo, hi = config.bounds_as_array()
    P = theta_arr.shape[0]

    # 先标记越界个体
    feas = (theta_arr >= lo).all(axis=1) & (theta_arr <= hi).all(axis=1)

    f = np.full((P, 2), INFEASIBLE, dtype=np.float64)

    if feas.any():
        traj = fhn_simulate_batch(theta_arr[feas], t, config)
        y_pred = traj

        # f1: MSE（向量化，对 (P, n)）
        diff = y_pred - y_true[None, :]
        mse = np.mean(diff * diff, axis=1)

        # f2: 发放次数相符则 0，否则阶跃惩罚（非线性、不可导）
        spikes = count_spikes_multi(traj)
        fire = np.where(spikes == n_spikes_true, 0.0, FIRE_PENALTY)

        f[feas, 0] = mse
        f[feas, 1] = fire

    return f


def objectives_single(theta, t, y_true, n_spikes_true):
    """单个个体的目标向量 [f_mse, f_fire]。"""
    return objectives(np.asarray(theta, dtype=np.float64).reshape(1, -1),
                      t, y_true, n_spikes_true)[0]