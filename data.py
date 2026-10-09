# -*- coding: utf-8 -*-
"""
合成训练数据生成：用已知真实参数跑模型，加观测噪声，作为 Ground Truth。

类比论文：真实 T-World 是把候选模型在「一系列协议」下仿真，输出与实验参考值比较。
这里只用一个起搏协议（一串脉冲），但保留了「由已知参数生成、加噪声」的合成数据思想。
"""
import os
import numpy as np

import config
from model import fhn_simulate, count_spikes


def make_time_grid():
    """生成时间网格 t（0 到 T_END，步长 DT）。"""
    n = int(round(config.T_END / config.DT)) + 1
    t = np.linspace(0.0, config.T_END, n)
    return t


def generate_ground_truth(seed=0):
    """
    生成并保存 ground truth：
      - t     : 时间网格
      - y     : 带噪声的电压观测 V_obs(t)
      - y_clean : 无噪声电压轨迹
      - theta_true : 真实参数向量
      - n_spikes   : 真实发放次数
    返回 (t, y, y_clean, theta_true, n_spikes)。
    """
    rng = np.random.default_rng(seed)
    t = make_time_grid()
    theta_true = config.theta_to_vec(config.THETA_TRUE)
    y_clean = fhn_simulate(theta_true, t, config)
    y = y_clean + rng.normal(0.0, config.NOISE_STD, size=y_clean.shape)
    n_spikes = count_spikes(y_clean)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    np.savez(
        os.path.join(config.DATA_DIR, "ground_truth.npz"),
        t=t,
        y=y,
        y_clean=y_clean,
        theta_true=theta_true,
        n_spikes=n_spikes,
        # 记录生成时用的配置，供 load_ground_truth 校验是否过期
        dt=config.DT,
        t_end=config.T_END,
        pace_cl=config.PACE_CL,
        pace_dur=config.PACE_DUR,
        noise_std=config.NOISE_STD,
    )
    return t, y, y_clean, theta_true, n_spikes


def load_ground_truth():
    """读取 ground truth；若不存在或与当前 config 不匹配，则重新生成。"""
    path = os.path.join(config.DATA_DIR, "ground_truth.npz")
    fresh = True
    if os.path.exists(path):
        d = np.load(path)
        fresh = not (
            float(d["dt"]) == config.DT
            and float(d["t_end"]) == config.T_END
            and float(d["pace_cl"]) == config.PACE_CL
            and float(d["pace_dur"]) == config.PACE_DUR
            and float(d["noise_std"]) == config.NOISE_STD
        )
    if fresh:
        return generate_ground_truth()
    d = np.load(path)
    return d["t"], d["y"], d["y_clean"], d["theta_true"], int(d["n_spikes"])


if __name__ == "__main__":
    t, y, yc, th, ns = generate_ground_truth()
    print(f"时间点数={len(t)}, 真实参数={th}, 真实发放次数={ns}")
    print(f"已保存到 {os.path.join(config.DATA_DIR, 'ground_truth.npz')}")