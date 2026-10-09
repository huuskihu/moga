# -*- coding: utf-8 -*-
"""
训练入口 / 启动器（代码生成与训练分离）。

一次拟合同一份合成数据，跑三/四种「模型」，对比两种训练范式：
  1. 梯度下降 + 白盒 ODE（gd.py）      -> 卡死或发散     （T-World 弃用 GD 的原因）
  2. 多目标遗传算法 + 白盒 ODE（ga.py）-> 稳健恢复真值   （T-World 实际用的方法）
  3. 梯度下降 + 可微神经网络代理（neural.py）-> 平滑收敛  （ProteoFlow/UMA 范式）

用法（在个人电脑上、有空闲时间时自行运行）：
    python train.py               # 完整训练，结果存到 results/
    python train.py --quick       # 快速冒烟验证（代/轮数极少，几秒~十几秒跑完）
    python train.py --gd-init both   # 同时跑坏初值 + 好初值两种 GD，便于对比
    python train.py --gd-init both --skip-neural   # 只看 GD vs GA（不含神经代理）
    python train.py --skip-gd     # 只跑遗传算法
    python train.py --skip-ga     # 只跑梯度下降
    python train.py --epochs 3000 --generations 150   # 自定义训练量

训练完成后，把 results/ 目录下的文件（或整个 results 文件夹）发回来，
由我来做结果解读与可视化（visualize.py 也会读取同样的 results/）。
"""
import argparse
import json
import os
import time

import numpy as np

import config
import data as data_mod


def save_npz(path, **arrays):
    np.savez(path, **arrays)


def _json_safe(obj):
    """把非有限浮点(NaN/Inf)转成 None，避免 json.dump 写出非法 JSON。"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def run_and_save():
    args = _parse_args()

    # ---------- 1. 准备 ground truth ----------
    t, y, y_clean, theta_true, n_spikes_true = data_mod.load_ground_truth()
    n = len(t)
    print(f"[1/4] Ground truth 就绪：时间点数={n}, "
          f"真实参数={np.round(theta_true, 4)}, 真实发放次数={n_spikes_true}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary = {
        "theta_true": theta_true.tolist(),
        "n_spikes_true": int(n_spikes_true),
        "config": {
            "DT": config.DT, "T_END": config.T_END,
            "PACE_CL": config.PACE_CL, "PACE_DUR": config.PACE_DUR,
            "NOISE_STD": config.NOISE_STD,
        },
    }

    # ---------- 2. 梯度下降 ----------
    if not args.skip_gd:
        from gd import run_gd
        init_modes = ["bad", "good"] if args.gd_init == "both" else [args.gd_init]
        for mode in init_modes:
            print(f"\n[2/4] 运行梯度下降（初值={mode}）...")
            t0 = time.time()
            res = run_gd(t, y, n_spikes_true, init=mode,
                         epochs=args.epochs, verbose=True)
            dt = time.time() - t0
            key = f"gd_{mode}"
            save_npz(
                os.path.join(config.RESULTS_DIR, f"{key}.npz"),
                theta_history=res["theta_history"],
                loss_history=res["loss_history"],
                final_theta=res["final_theta"],
                final_objectives=res["final_objectives"],
                final_V=res["final_V"],
            )
            summary[key] = {
                "init": mode,
                "final_theta": res["final_theta"].tolist(),
                "final_objectives": res["final_objectives"].tolist(),
                "runtime_sec": round(dt, 2),
            }
            print(f"  GD({mode}) 完成，耗时 {dt:.1f}s，"
                  f"最终目标 [f_mse, f_fire] = {np.round(res['final_objectives'], 4)}")

    # ---------- 3. 遗传算法 ----------
    if not args.skip_ga:
        from ga import run_ga
        print(f"\n[3/4] 运行遗传算法 NSGA-II（种群={args.pop}, 代数={args.generations}）...")
        t0 = time.time()
        res = run_ga(t, y, n_spikes_true,
                     pop_size=args.pop, generations=args.generations, verbose=True)
        dt = time.time() - t0
        save_npz(
            os.path.join(config.RESULTS_DIR, "ga.npz"),
            best_theta=res["best_theta"],
            best_objectives=res["best_objectives"],
            pareto_theta=res["pareto_theta"],
            pareto_objectives=res["pareto_objectives"],
            final_population=res["final_population"],
            final_objectives=res["final_objectives"],
            history_f1=res["history_f1"],
            history_feasible=res["history_feasible"],
        )
        summary["ga"] = {
            "best_theta": res["best_theta"].tolist(),
            "best_objectives": res["best_objectives"].tolist(),
            "pareto_size": int(len(res["pareto_theta"])),
            "final_feasible_ratio": float((res["final_objectives"][:, 1] == 0).mean()),
            "runtime_sec": round(dt, 2),
        }
        print(f"  GA 完成，耗时 {dt:.1f}s，最优参数 "
              f"{np.round(res['best_theta'], 4)}，"
              f"目标 [f_mse, f_fire] = {np.round(res['best_objectives'], 4)}，"
              f"帕累托前沿大小={len(res['pareto_theta'])}")

    # ---------- 4. 可微神经网络代理（ProteoFlow/UMA 范式：GD 可用） ----------
    if not args.skip_neural:
        from neural import run_neural
        print(f"\n[4/4] 运行可微神经网络代理 + 梯度下降（轮数={args.neural_epochs}）...")
        t0 = time.time()
        res = run_neural(t, y_clean, epochs=args.neural_epochs, verbose=True)
        dt = time.time() - t0
        save_npz(
            os.path.join(config.RESULTS_DIR, "neural.npz"),
            loss_history=res["loss_history"],
            final_V=res["final_V"],
            final_loss=res["final_loss"],
        )
        summary["neural"] = {
            "final_loss": res["final_loss"],
            "n_params": res["n_params"],
            "runtime_sec": round(dt, 2),
        }
        print(f"  神经代理完成，耗时 {dt:.1f}s，最终 MSE={res['final_loss']:.6f}")

    # ---------- 5. 写出总结 ----------
    with open(os.path.join(config.RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)

    print(f"\n完成！结果已保存到 {config.RESULTS_DIR}")
    print("把 results/ 里的文件发回来，我来做结果解读与可视化（或用 visualize.py 自行出图）。")


def _parse_args():
    p = argparse.ArgumentParser(description="T-World(极简替身) GD vs GA 参数标定对比")
    p.add_argument("--quick", action="store_true", help="快速冒烟验证（极少的轮数/代数）")
    p.add_argument("--skip-gd", action="store_true", help="跳过梯度下降")
    p.add_argument("--skip-ga", action="store_true", help="跳过遗传算法")
    p.add_argument("--skip-neural", action="store_true", help="跳过可微神经网络代理")
    p.add_argument("--gd-init", choices=["bad", "good", "both"], default="bad",
                   help="GD 初值选择")
    p.add_argument("--epochs", type=int, default=None, help="GD 训练轮数（默认用 config）")
    p.add_argument("--neural-epochs", type=int, default=None, help="神经代理训练轮数（默认用 config）")
    p.add_argument("--generations", type=int, default=None, help="GA 代数（默认用 config）")
    p.add_argument("--pop", type=int, default=None, help="GA 种群大小（默认用 config）")
    args = p.parse_args()

    if args.quick:
        if args.epochs is None:
            args.epochs = 20
        if args.neural_epochs is None:
            args.neural_epochs = 50
        if args.generations is None:
            args.generations = 5
        if args.pop is None:
            args.pop = 40
    if args.epochs is None:
        args.epochs = config.GD_EPOCHS
    if args.neural_epochs is None:
        args.neural_epochs = config.GD_EPOCHS
    if args.generations is None:
        args.generations = config.GA_GENERATIONS
    if args.pop is None:
        args.pop = config.GA_POP
    return args


if __name__ == "__main__":
    run_and_save()