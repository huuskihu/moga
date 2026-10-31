# -*- coding: utf-8 -*-
"""支撑实验 B：多随机种子重复 —— 检验「好初值下 GD 比 GA 更准」是否站得住。

每次换一个观测噪声实现（等价于重采一次实验数据），分别跑：
  · GD + 白盒 ODE（好初值，100 轮）
  · NSGA-II + 白盒 ODE（pop=100, gen=80）
记录真实 ODE 复核后的 MSE 与发放次数、耗时。

注意：不覆盖 data/ground_truth.npz（那里是项目原始数据），噪声在内存里重采。

输出：results_surrogate/seeds.json（每跑完一组就落盘，中途中断也有部分结果）
用法：python exp_seeds.py --n 5
"""
import argparse
import json
import os
import time

import numpy as np

import config
import data as dm
from model import fhn_simulate, count_spikes
from gd import run_gd
from ga import run_ga

OUT = os.path.join(config.BASE_DIR, "results_surrogate", "seeds.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="种子个数")
    ap.add_argument("--epochs", type=int, default=config.GD_EPOCHS, help="GD 轮数")
    ap.add_argument("--gen", type=int, default=config.GA_GENERATIONS)
    ap.add_argument("--pop", type=int, default=config.GA_POP)
    args = ap.parse_args()

    t, _, y_clean, th_true, ns = dm.load_ground_truth()
    rows = []

    print(f"多随种子：{args.n} 组；GD {args.epochs} 轮 / GA pop={args.pop} gen={args.gen}")
    print(f"{'seed':>4} {'方法':<8}{'MSE':>12}{'发放':>5}{'耗时(s)':>9}")
    print("-" * 44)

    for seed in range(args.n):
        rng = np.random.default_rng(1000 + seed)          # 与 ground_truth 的 seed 错开
        y = y_clean + rng.normal(0.0, config.NOISE_STD, size=y_clean.shape)

        # --- GD 好初值 ---
        t0 = time.time()
        r_gd = run_gd(t, y, ns, init="good", epochs=args.epochs, verbose=False)
        dt_gd = time.time() - t0
        th = np.asarray(r_gd["final_theta"], dtype=np.float64)
        V = fhn_simulate(th, t, config)
        mse_gd = float(np.mean((V - y) ** 2))
        sp_gd = count_spikes(V)

        # --- NSGA-II ---
        t0 = time.time()
        r_ga = run_ga(t, y, ns, pop_size=args.pop, generations=args.gen,
                      seed=seed, verbose=False)
        dt_ga = time.time() - t0
        thg = np.asarray(r_ga["best_theta"], dtype=np.float64)
        Vg = fhn_simulate(thg, t, config)
        mse_ga = float(np.mean((Vg - y) ** 2))
        sp_ga = count_spikes(Vg)

        print(f"{seed:>4} {'GD':<8}{mse_gd:>12.3e}{sp_gd:>5}{dt_gd:>9.1f}")
        print(f"{'':>4} {'GA':<8}{mse_ga:>12.3e}{sp_ga:>5}{dt_ga:>9.1f}")

        rows.append({
            "seed": seed,
            "gd": {"mse": mse_gd, "spikes": int(sp_gd), "sec": round(dt_gd, 2),
                   "theta": th.tolist()},
            "ga": {"mse": mse_ga, "spikes": int(sp_ga), "sec": round(dt_ga, 2),
                   "theta": thg.tolist()},
        })

        # 每跑完一组就落盘
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({"n": args.n, "noise_base_seed": 1000, "rows": rows},
                      f, ensure_ascii=False, indent=2)

    # ---------- 统计 ----------
    g = np.array([r["gd"]["mse"] for r in rows])
    a = np.array([r["ga"]["mse"] for r in rows])
    diff = g - a
    print("\n" + "=" * 52)
    print(f"GD 好初值  MSE 均值 {g.mean():.3e} ± {g.std(ddof=1):.3e}  (n={len(g)})")
    print(f"NSGA-II    MSE 均值 {a.mean():.3e} ± {a.std(ddof=1):.3e}  (n={len(a)})")
    print(f"逐对差值(GD-GA)：均值 {diff.mean():+.3e}，{np.sum(diff < 0)}/{len(diff)} 次 GD 更小")
    print(f"GD 全部发放正确：{all(r['gd']['spikes'] == ns for r in rows)}；"
          f"GA 全部发放正确：{all(r['ga']['spikes'] == ns for r in rows)}")
    print(f"噪声方差下限 {config.NOISE_STD**2:.1e}")
    print("=" * 52)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "n": args.n, "noise_base_seed": 1000, "rows": rows,
            "summary": {
                "gd_mean": float(g.mean()), "gd_std": float(g.std(ddof=1)),
                "ga_mean": float(a.mean()), "ga_std": float(a.std(ddof=1)),
                "diff_mean": float(diff.mean()),
                "gd_better_count": int(np.sum(diff < 0)),
                "noise_floor": config.NOISE_STD ** 2,
            },
        }, f, ensure_ascii=False, indent=2)
    print("已保存:", OUT)


if __name__ == "__main__":
    main()
