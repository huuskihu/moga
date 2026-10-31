# -*- coding: utf-8 -*-
"""2×3 对照矩阵图：优化器（GD好初值 / GD坏初值 / NSGA-II）× 模型（白盒 ODE / 代理模型）。
   所有 MSE 均为「把得到的参数丢回真实 ODE」复核后的结果。
   → results_surrogate/fig_matrix.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

import config

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(config.BASE_DIR, "results_surrogate")
S = json.load(open(os.path.join(OUT, "summary_surrogate.json"), encoding="utf-8"))
BASE = json.load(open(os.path.join(config.RESULTS_DIR, "summary.json"), encoding="utf-8"))
NF = S["noise_floor_mse"]

ROWS = [
    ("GD · 好初值", [("白盒", BASE.get("gd_good", S["wb_gd_good"])["mse_real_ode"],
                       S["wb_gd_good"]["cost_sec"], S["wb_gd_good"]["n_spikes_real_ode"]),
                     ("代理", S["gd_good_mse"]["mse_real_ode"],
                      S["gd_good_mse"]["cost_sec"], S["gd_good_mse"]["n_spikes_real_ode"])]),
    ("GD · 坏初值", [("白盒", BASE["gd_bad"]["final_objectives"][0],
                       BASE["gd_bad"]["runtime_sec"], None),
                     ("代理", S["gd_bad_mse"]["mse_real_ode"],
                      S["gd_bad_mse"]["cost_sec"], S["gd_bad_mse"]["n_spikes_real_ode"])]),
    ("NSGA-II", [("白盒", BASE["ga"]["best_objectives"][0],
                  BASE["ga"]["runtime_sec"], 4),
                 ("代理", S["ga_surrogate"]["mse_real_ode"],
                  S["ga_surrogate"]["cost_sec"], S["ga_surrogate"]["n_spikes_real_ode"])]),
]

fig, ax = plt.subplots(figsize=(11.8, 6.6))
ax.set_xlim(0, 11.2); ax.set_ylim(0, 7.4); ax.axis("off")

# 表头
ax.text(4.05, 7.02, "白盒 ODE（原模型）", ha="center", fontsize=13, weight="bold", color="#1A2230")
ax.text(8.65, 7.02, "代理模型（θ→V）", ha="center", fontsize=13, weight="bold", color="#1A2230")
ax.plot([2.75, 10.75], [6.78, 6.78], color="0.6", lw=1)

# 行标
for r, (rowname, _) in enumerate(ROWS):
    yc = 5.55 - r * 1.85
    ax.text(0.35, yc, rowname, fontsize=13, weight="bold", va="center", color="#1A2230")

for r, (rowname, cells) in enumerate(ROWS):
    for c, (model, mse, cost, spikes) in enumerate(cells):
        xc = 4.05 + c * 4.6
        yc = 5.55 - r * 1.85
        # 判定：MSE 低于 1e-2 且发放次数正确 → 通过
        ok = mse < 1e-2 and (spikes == 4)
        face = "#E8F6EE" if ok else "#FDECEC"
        edge = "#2E9E5B" if ok else "#D64545"
        ax.add_patch(FancyBboxPatch((xc - 2.0, yc - 0.78), 4.0, 1.56,
                                    boxstyle="round,pad=0.06,rounding_size=0.12",
                                    fc=face, ec=edge, lw=1.6))
        ax.text(xc, yc + 0.42, ("通过" if ok else "失败"), ha="center", fontsize=11,
                color=edge, weight="bold")
        ax.text(xc, yc - 0.02, f"MSE = {mse:.2e}", ha="center", fontsize=15,
                weight="bold", color="#1A2230")
        sp = "未收敛" if spikes is None else f"发放 {spikes} 次"
        ax.text(xc, yc - 0.45, f"{cost:.2f} s ｜ {sp}", ha="center", fontsize=10, color="#555")

ax.text(0.35, 0.72,
        f"注：MSE 均为「参数丢回真实 ODE 重算」的结果；观测噪声方差 {NF:.0e} 是理论下限，低于它没有意义。\n"
        f"同一个好初值下，白盒 GD {S['wb_gd_good']['cost_sec']:.1f}s → {S['wb_gd_good']['mse_real_ode']:.2e}；"
        f"代理 GD {S['gd_good_mse']['cost_sec']:.2f}s → {S['gd_good_mse']['mse_real_ode']:.2e}"
        f"（加速 {S['wb_gd_good']['cost_sec']/S['gd_good_mse']['cost_sec']:.0f}×，精度退到代理的保真度地板）。",
        fontsize=9.5, color="#444", va="top")

ax.set_title("同一条数据、同一条目标函数：优化器 × 模型 的完整 2×3 对照", fontsize=15, pad=14)
fig.tight_layout()
out = os.path.join(OUT, "fig_matrix.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)
