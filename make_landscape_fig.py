# -*- coding: utf-8 -*-
"""沿「GD 收敛到的坏解 → 真值」切片扫描地形。

上：MSE —— 盆地里有缓坡，但靠近分岔处先抬起一个势垒，然后是一道真正的断崖
下：发放次数 —— 那个"断崖"的物理来源

注：这是 1D 切片，只用来看地形；GD 实际走的是 4D 里的另一条路。
→ results_surrogate/fig_landscape.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import data as dm
from model import fhn_simulate, count_spikes

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

t, y, yc, th_true, ns = dm.load_ground_truth()
D = np.load(os.path.join(config.BASE_DIR, "results_surrogate", "artifacts.npz"))
th_bad = D["theta_gd_bad"]

# 在断崖附近加密采样
S = np.concatenate([np.linspace(0.0, 0.46, 40),
                    np.linspace(0.46, 0.52, 140),
                    np.linspace(0.52, 1.0, 30)])
mse, spk = [], []
for s in S:
    p = th_bad + s * (th_true - th_bad)
    V = fhn_simulate(p, t, config)
    mse.append(float(np.mean((V - y) ** 2)))
    spk.append(count_spikes(V))
mse = np.array(mse)
spk = np.array(spk)

# 找关键点
i_sw = int(np.where(spk == 4)[0][0])          # 切换成 4 次发放
seg = np.where((S > 0.40) & (S < S[i_sw]))[0]
i_peak = seg[int(np.argmax(mse[seg]))]            # 崖口前的势垒顶
pre = np.where((S > 0.30) & (S < S[i_peak]))[0]
i_min = pre[int(np.argmin(mse[pre]))]             # 势垒前的小盆地
cliff = mse[i_peak] / mse[i_sw]                   # 势垒顶 → 崖底 的倍数
width = S[i_sw] - S[i_peak]
print(f"[debug] i_min s={S[i_min]:.4f} mse={mse[i_min]:.4f} | "
      f"i_peak s={S[i_peak]:.4f} mse={mse[i_peak]:.4f} | switch s={S[i_sw]:.4f} | "
      f"崖宽 {width:.4f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.6, 6.4), sharex=True,
                               gridspec_kw={"height_ratios": [2.3, 1]})

ax1.semilogy(S, mse, "-", color="#2E6FD9", lw=1.8)
ax1.axhline(config.NOISE_STD ** 2, color="gray", ls="--", lw=1)
ax1.text(0.015, config.NOISE_STD ** 2 * 1.35, "噪声下限 4.0e-4（理论最优）",
         color="gray", fontsize=9)

ax1.set_ylim(2.5e-4, 25)
ax1.plot([S[0]], [mse[0]], "o", ms=8, color="#D64545")
ax1.annotate(f"GD 收敛到的解\nMSE={mse[0]:.2f}，{spk[0]} 次发放",
             xy=(S[0], mse[0]), xytext=(0.05, 8.0),
             arrowprops=dict(arrowstyle="->", color="#D64545", lw=1.4),
             fontsize=11, color="#D64545", weight="bold")

ax1.plot([S[i_min]], [mse[i_min]], "o", ms=6, color="#8A5CD6")
ax1.annotate(f"小盆地底 {mse[i_min]:.2f}", xy=(S[i_min], mse[i_min]),
             xytext=(0.20, 0.42), fontsize=11,
             color="#8A5CD6", weight="bold",
             arrowprops=dict(arrowstyle="->", color="#8A5CD6", lw=1.2))

ax1.plot([S[i_peak]], [mse[i_peak]], "o", ms=6, color="#E08A1E")
ax1.annotate(f"势垒顶 {mse[i_peak]:.2f}\n比盆地底高 {(mse[i_peak]/mse[i_min]-1)*100:.0f}%",
             xy=(S[i_peak], mse[i_peak]), xytext=(0.52, 6.5),
             arrowprops=dict(arrowstyle="->", color="#E08A1E", lw=1.4),
             fontsize=11, color="#E08A1E", weight="bold")

ax1.annotate(f"断崖：{width:.4f} 的参数区间内\n"
             f"MSE 从 {mse[i_peak]:.2f} 掉到 {mse[i_sw]:.2f}（{cliff:.0f} 倍）",
             xy=(S[i_sw], mse[i_sw]), xytext=(0.545, 4.5e-3),
             arrowprops=dict(arrowstyle="->", color="#D64545", lw=1.6),
             fontsize=11, color="#D64545", weight="bold")

ax1.set_ylabel("MSE（对数轴）", fontsize=11.5)
ax1.set_title("从 GD 收敛到的解走向真值：中间隔着「势垒 + 断崖」", fontsize=14)
ax1.grid(alpha=0.3, which="both")

ax2.step(S, spk, where="mid", color="#D64545", lw=2)
ax2.fill_between(S, 0, spk, step="mid", color="#D64545", alpha=0.08)
ax2.axhline(ns, color="gray", ls=":", lw=1)
ax2.text(0.015, ns + 0.9, f"正确值 {ns} 次", color="gray", fontsize=9)
ax2.set_xlabel("s（0 = GD 收敛到的解，1 = 真实参数）", fontsize=11.5)
ax2.set_ylabel("发放次数", fontsize=11.5)
ax2.set_title("发放次数：断崖的物理来源（20 次 → 4 次，中间没有过渡态）", fontsize=12.5)
ax2.grid(alpha=0.3)

fig.tight_layout()
out = os.path.join(config.BASE_DIR, "results_surrogate", "fig_landscape.png")
fig.savefig(out, dpi=150)
print("saved:", out)
print(f"起点 MSE={mse[0]:.4f} | 盆地底 {mse[i_min]:.4f}(s={S[i_min]:.3f}) | "
      f"势垒顶 {mse[i_peak]:.4f}(s={S[i_peak]:.3f}) 高 {(mse[i_peak]/mse[i_min]-1)*100:.1f}% | "
      f"断崖 {mse[i_sw-1]:.4f}->{mse[i_sw]:.4f} = {cliff:.1f}×")
