# -*- coding: utf-8 -*-
"""
结果可视化（训练完成后运行，或交给我来解读）。

读取 results/ 目录下的结果文件，绘制三张图：
  1. 拟合对比：真实数据 + 梯度下降拟合 + 遗传算法最优拟合
  2. 梯度下降 Loss 曲线（坏初值常表现为卡死平台；好初值因刚性发散为 NaN）
  3. 遗传算法种群与帕累托前沿（横轴 f_MSE，纵轴 f_fire 阶跃惩罚）

用法：
    python visualize.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（Windows 常用字体，避免中文标签显示为方框）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

import config
from data import load_ground_truth
from model import fhn_simulate


def _load(key):
    path = os.path.join(config.RESULTS_DIR, f"{key}.npz")
    return np.load(path) if os.path.exists(path) else None


def main():
    t, y, y_clean, theta_true, n_spikes_true = load_ground_truth()

    gd_bad = _load("gd_bad")
    gd_good = _load("gd_good")
    ga = _load("ga")
    neural = _load("neural")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ---- 子图1：拟合对比 ----
    ax = axes[0]
    ax.plot(t, y_clean, color="gray", lw=2, label="ground truth (无噪声)")
    ax.plot(t, y, color="lightgray", lw=1, alpha=0.5, label="观测(含噪声)")

    if gd_bad is not None:
        ax.plot(t, gd_bad["final_V"], color="tab:red", lw=1.2, alpha=0.85,
                label="GD(白盒ODE,坏初值) 拟合")
    if gd_good is not None:
        ax.plot(t, gd_good["final_V"], color="tab:orange", lw=1.2, alpha=0.85,
                label="GD(白盒ODE,好初值) 拟合")
    if ga is not None:
        Vga = fhn_simulate(ga["best_theta"], t, config)
        ax.plot(t, Vga, color="tab:green", ls="--", lw=1.6,
                label="GA 最优拟合")
    if neural is not None:
        ax.plot(t, neural["final_V"], color="tab:purple", ls="-.", lw=1.4,
                label="GD(神经代理) 拟合")

    ax.set_title(f"拟合对比 (真实参数={np.round(theta_true, 3)}, 发放={n_spikes_true})")
    ax.set_xlabel("t")
    ax.set_ylabel("V(t)")
    ax.legend(fontsize=8)

    # ---- 子图2：Loss 曲线 ----
    ax = axes[1]
    for key, color, label in [("gd_bad", "tab:red", "GD(白盒ODE,坏初值)"),
                               ("gd_good", "tab:orange", "GD(白盒ODE,好初值)")]:
        d = _load(key)
        if d is not None:
            hist = d["loss_history"]
            finite = np.isfinite(hist)
            ax.plot(np.arange(len(hist))[finite], hist[finite],
                    color=color, lw=1.2, label=label)
    if neural is not None:
        hist = neural["loss_history"]
        ax.plot(np.arange(len(hist)), hist, color="tab:purple", lw=1.4,
                label="GD(神经代理)")
    ax.set_title("损失曲线（GD 对白盒 ODE 卡死/发散，对神经代理平滑收敛）")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (MSE)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)

    # ---- 子图3：GA 种群与帕累托前沿 ----
    ax = axes[2]
    if ga is not None:
        obj = ga["final_objectives"]      # (P, 2): [f_mse, f_fire]
        pareto = ga["pareto_objectives"]  # (k, 2)
        ax.scatter(obj[:, 0], obj[:, 1], s=14, color="tab:blue", alpha=0.4,
                   label="末代种群")
        ax.scatter(pareto[:, 0], pareto[:, 1], s=90, facecolors="none",
                   edgecolors="tab:red", lw=1.6, label="帕累托前沿")
        ax.set_xscale("log")
        ax.set_xlabel("f_MSE (对数)")
        ax.set_ylabel("f_fire (0=发放正确, 1000=发放错误)")
        ax.legend(fontsize=8)
        ax.set_title("遗传算法：种群与帕累托前沿")
    else:
        ax.set_title("未找到 ga.npz，请先运行 train.py")

    fig.tight_layout()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, "visualization.png")
    fig.savefig(out, dpi=120)
    print(f"已保存可视化图到 {out}")


if __name__ == "__main__":
    main()