# -*- coding: utf-8 -*-
"""代理模型实验结果可视化：4 联图 → results_surrogate/fig_surrogate.png"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import config
from data import make_time_grid

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(config.BASE_DIR, "results_surrogate")
D = np.load(os.path.join(OUT, "artifacts.npz"))
with open(os.path.join(OUT, "summary_surrogate.json"), encoding="utf-8") as f:
    S = json.load(f)

t = make_time_grid()
sp_val = D["spikes_sample"][D["val_idx"]]
mse_v = D["val_mse"]
rel_v = D["val_rel"]

fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.8))

# ---------------- ① 代理拟合：目标区制 vs 爆发区制 ----------------
ax = axes[0, 0]
four = np.where(sp_val == 4)[0]
i_good = four[np.argsort(mse_v[four])[len(four) // 2]]
i_bad = int(np.argmax(mse_v))
OFF = 8.0
for i, off, lab in [(i_good, 0.0, f"4 次发放（目标区制）"),
                    (i_bad, -OFF, f"{int(sp_val[i_bad])} 次发放（爆发区制）")]:
    ax.plot(t, D["true_traj_val"][i] + off, color="black", lw=1.0)
    ax.plot(t, D["surrogate_traj_val"][i] + off, color="tab:red", lw=1.0, ls="--")
    ax.text(0.6, off + 3.4, f"{lab}   MSE={mse_v[i]:.1e}",
            fontsize=9, color="tab:blue", weight="bold")
ax.axhline(-OFF + 4.6, color="gray", ls=":", lw=1)
ax.set_ylim(-12.8, 7.0)
ax.set_title("① 代理拟合：目标区制很准，爆发区制崩掉", fontsize=12)
ax.set_xlabel("时间"); ax.set_ylabel("V（下段整体下移，仅为同图对比）")
ax.legend(handles=[Line2D([], [], color="black", label="真实 ODE"),
                   Line2D([], [], color="tab:red", ls="--", label="代理模型")],
          loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

# ---------------- ② 误差由什么决定 ----------------
ax = axes[0, 1]
ax.scatter(sp_val, rel_v * 100, s=30, alpha=0.75, color="tab:blue")
ax.set_yscale("log")
ax.set_xlabel("该参数下的真实发放次数（动力学区制）")
ax.set_ylabel("轨迹相对误差 (%)")
ea = S["error_attribution"]
ax.set_title("② 误差几乎只由「区制」决定，与「离训练集多远」无关", fontsize=12)
ax.text(0.03, 0.95,
        f"vs 发放次数  Spearman ρ={ea['rho_vs_spike_count']:+.2f} (p<0.001)\n"
        f"vs 最近样本距离 ρ={ea['rho_vs_nn_distance']:+.2f} (p={ea['p_vs_nn_distance']:.2f}，不显著)",
        transform=ax.transAxes, fontsize=9.5, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
ax.axhline(ea["by_regime"]["4"]["rel_median"] * 100, color="tab:green", ls="--", lw=1)
ax.text(1, ea["by_regime"]["4"]["rel_median"] * 100 * 1.18,
        f"目标区制中位数 {ea['by_regime']['4']['rel_median']*100:.1f}%",
        color="tab:green", fontsize=9)
ax.grid(alpha=0.3, which="both")

# ---------------- ③ GD 在代理模型上的损失曲线 ----------------
ax = axes[1, 0]
ax.plot(D["gd_hist_bad"], label="坏初值 · 仅 MSE → 停在 24 次发放", color="tab:red", lw=1.4)
ax.plot(D["gd_hist_bad_spike"], label="坏初值 · +可微软发放 → 仍未跳出", color="tab:orange", lw=1.4)
ax.plot(D["gd_hist_good"], label="好初值 · 仅 MSE → 收敛到 1.5e-3", color="tab:green", lw=1.6)
ax.set_yscale("log")
ax.set_xlabel("Adam 迭代步")
ax.set_ylabel("代理模型上的损失")
ax.set_title("③ 都平滑可导，但初值仍决定成败（代理不消除局部极小）", fontsize=12)
ax.legend(fontsize=9, loc="center right", framealpha=0.95)
ax.grid(alpha=0.3, which="both")

# ---------------- ④ 外推测试 ----------------
ax = axes[1, 1]
cases = S["extrapolation"]["cases"]
names = [c["case"].split("（")[0] for c in cases]
vals = [c["rel_err"] * 100 for c in cases]
colors = ["tab:red" if not c["reference_valid"] else "tab:orange" for c in cases]
ax.bar(range(len(cases)), vals, color=colors, alpha=0.85)
for i, c in enumerate(cases):
    ax.text(i, vals[i] + 2, f"{c['spike_pred']}/{c['spike_true']}", ha="center", fontsize=9.5)
ax.set_xticks(range(len(cases)))
ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0, max(vals) * 1.18)
ax.set_ylabel("相对误差 (%)")
ax.set_title("④ 越界查询：柱顶为「代理预测发放 / 真实发放」", fontsize=12)
ax.text(0.02, 0.96, "橙 = 参考模型仍有效    红 = 参考模型已数值失效",
        transform=ax.transAxes, fontsize=9, va="top")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("代理模型（参数→轨迹）实测：能替代 ODE 求解器到什么程度", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(OUT, "fig_surrogate.png")
fig.savefig(out, dpi=150)
print("saved:", out)
