# -*- coding: utf-8 -*-
"""支撑实验 A：给「170× 加速」和「势垒 + 断崖」补重复与多起点证据。

1) 计时：白盒 ODE 目标函数 vs 代理目标函数，各重复 5 轮，报中位数与离散度。
   代理模型直接从 results_surrogate/surrogate.pt 加载，不重训。
2) 地形：随机取 3 个「爆发型（>15 次发放）」起点，沿直线走向真值，
   各自测势垒高度与断崖倍数，检验 +27% / 21× 是否只是某一条切片的巧合。

输出：results_surrogate/exp_support.json
"""
import json
import os
import time

import numpy as np
import torch

import config
import data as dm
from model import fhn_simulate, count_spikes
from objective import objectives
from surrogate import TrajSurrogate
from run_surrogate import make_surrogate_objectives


def load_surrogate():
    """从 state_dict 还原代理模型（含 PCA 缓冲），避免重新训练。"""
    p = os.path.join(config.BASE_DIR, "results_surrogate", "surrogate.pt")
    sd = torch.load(p, map_location="cpu", weights_only=True)
    k = sd["head_coef.weight"].shape[0]
    hidden = sd["trunk.0.weight"].shape[0]
    m = TrajSurrogate(sd["lo"].numpy(), sd["hi"].numpy(),
                      sd["pca_mean"].numpy(), sd["pca_comps"].numpy(),
                      sd["coef_std"].numpy(), k=k, hidden=hidden)
    m.load_state_dict(sd)
    m.eval()
    return m


def main():
    t, y, y_clean, th_true, ns = dm.load_ground_truth()
    lo, hi = config.bounds_as_array()
    out = {}

    # ---------- 1. 计时：各 5 轮 ----------
    model = load_surrogate()
    rng = np.random.default_rng(0)
    pop = rng.uniform(lo, hi, size=(100, 4))
    f_sur = make_surrogate_objectives(model)

    N_CALLS = 81                      # 与一次 GA（1 次初始化 + 80 代）相同
    n_rep = 5
    wb, sp = [], []
    for _ in range(n_rep):
        t0 = time.time()
        for _ in range(N_CALLS):
            objectives(pop, t, y, ns)
        wb.append(time.time() - t0)

        t0 = time.time()
        for _ in range(N_CALLS):
            f_sur(pop, t, y, ns)
        sp.append(time.time() - t0)

    wb, sp = np.array(wb), np.array(sp)
    speed = wb / sp
    print(f"[计时] 白盒 {np.median(wb):.2f}s（{wb.min():.2f}–{wb.max():.2f}），"
          f"代理 {np.median(sp):.3f}s（{sp.min():.3f}–{sp.max():.3f}）")
    print(f"       加速 {np.median(speed):.0f}×（{speed.min():.0f}–{speed.max():.0f}×），{n_rep} 轮")
    out["timing"] = {
        "calls": N_CALLS, "repeats": n_rep,
        "whitebox_sec": wb.tolist(), "surrogate_sec": sp.tolist(),
        "speedup_median": float(np.median(speed)),
        "speedup_min": float(speed.min()), "speedup_max": float(speed.max()),
    }

    # ---------- 2. 地形：3 个不同起点 ----------
    # 在参数箱内随机撒点，挑出发放次数 > 15 的（爆发区制）作为起点
    cand = rng.uniform(lo, hi, size=(400, 4))
    keep = []
    for p in cand:
        if count_spikes(fhn_simulate(p, t, config)) > 15:
            keep.append(p)
        if len(keep) >= 3:
            break
    print(f"\n[地形] 取到 {len(keep)} 个爆发型起点")

    rows = []
    for idx, start in enumerate(keep):
        S = np.concatenate([np.linspace(0, 0.45, 60),
                            np.linspace(0.45, 0.56, 160),
                            np.linspace(0.56, 1.0, 40)])
        mse, spk = [], []
        for s in S:
            p = start + s * (th_true - start)
            V = fhn_simulate(p, t, config)
            mse.append(float(np.mean((V - y) ** 2)))
            spk.append(count_spikes(V))
        mse, spk = np.array(mse), np.array(spk)

        hit = np.where(spk == ns)[0]
        if len(hit) == 0:                     # 这条切片切不到目标区制
            print(f"  起点{idx+1}: 未切到 4 次发放区制，跳过")
            continue
        i_sw = int(hit[0])
        seg = np.where((S > 0.2) & (S < S[i_sw]))[0]
        if len(seg) < 10:
            continue
        i_peak = seg[int(np.argmax(mse[seg]))]
        pre = np.where((S > 0.05) & (S < S[i_peak]))[0]
        i_min = pre[int(np.argmin(mse[pre]))] if len(pre) else 0
        barrier = mse[i_peak] / mse[i_min]
        cliff = mse[i_peak] / mse[i_sw]

        print(f"  起点{idx+1}: 起点 {spk[0]} 次 → 切换于 s={S[i_sw]:.3f} | "
              f"盆地 {mse[i_min]:.3f} → 势垒 {mse[i_peak]:.3f}（+{(barrier-1)*100:.0f}%）"
              f" → 崖底 {mse[i_sw]:.3f}（{cliff:.0f}×）")
        rows.append({"start_spikes": int(spk[0]), "s_switch": float(S[i_sw]),
                     "basin_mse": float(mse[i_min]), "peak_mse": float(mse[i_peak]),
                     "cliff_mse": float(mse[i_sw]),
                     "barrier_pct": float((barrier - 1) * 100), "cliff_ratio": float(cliff)})

    if rows:
        bp = [r["barrier_pct"] for r in rows]
        cr = [r["cliff_ratio"] for r in rows]
        print(f"  → 势垒 {min(bp):.0f}%–{max(bp):.0f}%，断崖 {min(cr):.0f}–{max(cr):.0f}×（{len(rows)} 条切片）")
        out["landscape"] = {"slices": rows,
                            "barrier_min": min(bp), "barrier_max": max(bp),
                            "cliff_min": min(cr), "cliff_max": max(cr)}

    p = os.path.join(config.BASE_DIR, "results_surrogate", "exp_support.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已保存:", p)


if __name__ == "__main__":
    main()
