# -*- coding: utf-8 -*-
"""
代理模型（Surrogate）实测：把已有 GD / NSGA-II 接到代理模型上，看效果到底如何。

实验设计（对照组是 results/summary.json 里的已有结果）：
  A. 白盒 ODE + GD (Adam)       —— 已有：146s → 越界解，f=1e6（失败）
  B. 白盒 ODE + NSGA-II         —— 已有：24s  → f_mse=4.3e-4，发放 4 次（成功）
  C. 代理模型 + GD (Adam)       —— 本次新增
  D. 代理模型 + NSGA-II         —— 本次新增
  E. 外推测试                    —— 代理模型的阿喀琉斯之踵

所有 C/D 得到的参数，最后都用**真实 ODE** 复核，确保不是代理模型自说自话。

用法：
    python run_surrogate.py                # 完整实验
    python run_surrogate.py --n 400 --epochs 2000
    python run_surrogate.py --quick        # 冒烟
"""
import argparse
import json
import os
import time

import numpy as np

import config
import data as data_mod
import objective as obj_mod
import ga as ga_mod
from model import fhn_simulate, count_spikes

from surrogate import build_dataset, fit_surrogate, predict, N_TRAIN

OUT_DIR = os.path.join(config.BASE_DIR, "results_surrogate")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# 代理模型版目标函数（供 NSGA-II 直接调用）
# ---------------------------------------------------------------------------
def make_surrogate_objectives(model):
    import torch
    lo, hi = config.bounds_as_array()
    y_t = None

    def _objectives(theta_arr, t, y_true, n_spikes_true):
        nonlocal y_t
        if y_t is None:
            y_t = torch.tensor(y_true, dtype=torch.float32)
        theta_arr = np.asarray(theta_arr, dtype=np.float64)
        P = len(theta_arr)
        feas = (theta_arr >= lo).all(axis=1) & (theta_arr <= hi).all(axis=1)
        out = np.full((P, 2), obj_mod.INFEASIBLE, dtype=np.float64)
        if feas.any():
            with torch.no_grad():
                th = torch.tensor(theta_arr[feas], dtype=torch.float32)
                mse, sp = model.mse_against(th, y_t)
            sp_int = torch.round(sp).numpy().astype(np.int64)
            fire = np.where(sp_int == n_spikes_true, 0.0, obj_mod.FIRE_PENALTY)
            out[feas, 0] = mse.numpy()
            out[feas, 1] = fire
        return out

    return _objectives


# ---------------------------------------------------------------------------
# 代理模型上跑梯度下降
# ---------------------------------------------------------------------------
def run_gd_on_surrogate(model, y_obs, init_vec, steps=400, lr=0.05,
                        n_spikes_true=None, lam_spike=0.0, verbose=True, tag="代理GD"):
    """
    在代理模型上做参数反演。theta 用 sigmoid 重参数化，天然满足盒子约束，
    不会像白盒版那样跑到 eps<=0 的非法区（这正是平滑带来的好处）。

    lam_spike > 0 时，把「发放次数」也写进损失。白盒 ODE 里发放次数是阶跃量、
    梯度为 0，根本没法优化；但代理模型输出的是连续值，可以直接求导——
    这是代理模型相对白盒 ODE 的一个额外好处。
    """
    import torch

    lo, hi = config.bounds_as_array()
    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)

    u0 = np.clip((np.asarray(init_vec) - lo) / (hi - lo), 1e-4, 1 - 1e-4)
    z = torch.logit(torch.tensor(u0, dtype=torch.float32)).clone().requires_grad_(True)
    y_t = torch.tensor(y_obs, dtype=torch.float32)
    n_sp = float(n_spikes_true) if n_spikes_true is not None else None

    opt = torch.optim.Adam([z], lr=lr)
    hist = np.zeros(steps)
    t0 = time.time()
    theta = None
    for i in range(steps):
        opt.zero_grad()
        theta = lo_t + (hi_t - lo_t) * torch.sigmoid(z)
        mse, sp = model.mse_against(theta.unsqueeze(0), y_t)
        loss = mse[0]
        if lam_spike > 0 and n_sp is not None:
            loss = loss + lam_spike * (sp[0] - n_sp) ** 2
        loss.backward()
        opt.step()
        hist[i] = loss.item()
        if verbose and (i % max(1, steps // 6) == 0 or i == steps - 1):
            print(f"  {tag} {i:4d}/{steps}: loss={hist[i]:.3e}  "
                  f"theta={theta.detach().numpy().round(4)}  "
                  f"发放预测={float(sp[0]):.2f}")
    cost = time.time() - t0
    theta_final = (lo_t + (hi_t - lo_t) * torch.sigmoid(z)).detach().numpy()
    return theta_final, hist, cost, 0


# ---------------------------------------------------------------------------
# 真实 ODE 复核
# ---------------------------------------------------------------------------
def verify_on_real_ode(theta, t, y_obs, n_spikes_true, label=""):
    V = fhn_simulate(theta, t, config)
    mse = float(np.mean((V - y_obs) ** 2))
    fired = count_spikes(V)
    f = obj_mod.objectives_single(theta, t, y_obs, n_spikes_true)
    if label:
        print(f"  [真实ODE复核] {label}: theta={np.round(theta, 4)} | "
              f"MSE={mse:.3e} | 发放={fired}(应={n_spikes_true}) | f=({f[0]:.3e}, {f[1]:.0f})")
    return {"theta": theta.tolist(), "mse_real_ode": mse,
            "n_spikes_real_ode": int(fired),
            "objectives_real_ode": f.tolist(), "V": V}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_TRAIN, help="代理模型训练样本数")
    ap.add_argument("--epochs", type=int, default=2000, help="代理模型训练轮数")
    ap.add_argument("--gd-steps", type=int, default=400, help="代理模型上 GD 的步数")
    ap.add_argument("--gen", type=int, default=config.GA_GENERATIONS, help="代理模型上 GA 的代数")
    ap.add_argument("--pop", type=int, default=config.GA_POP, help="代理模型上 GA 的种群")
    ap.add_argument("--whitebox-gd", action="store_true",
                    help="额外跑「白盒 ODE + GD 好初值」对照（较慢，约 150s）")
    ap.add_argument("--wb-epochs", type=int, default=config.GD_EPOCHS,
                    help="白盒 GD 对照的轮数")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n = 120
        args.epochs = 300
        args.gd_steps = 100
        args.gen = 10
        args.pop = 40

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {}

    # ---------- 0. Ground truth ----------
    t, y_obs, y_clean, theta_true, n_spikes_true = data_mod.load_ground_truth()
    print(f"[0] Ground truth：真实参数={np.round(theta_true, 4)}，真实发放次数={n_spikes_true}，"
          f"观测噪声 sigma={config.NOISE_STD}")
    noise_floor = config.NOISE_STD ** 2
    print(f"    噪声方差={noise_floor:.2e}（MSE 的理论下限，低于它的差异没有意义）")
    summary["theta_true"] = theta_true.tolist()
    summary["n_spikes_true"] = int(n_spikes_true)
    summary["noise_floor_mse"] = noise_floor

    # ---------- 1. 造数据 ----------
    print(f"\n[1] 造数据：LHS 采样 {args.n} 组参数，跑真实 ODE")
    theta_s, traj_s, spikes_s, data_cost = build_dataset(args.n, seed=0, t=t)
    summary["data"] = {"n_samples": args.n, "cost_sec": round(data_cost, 2),
                       "ms_per_sample": round(data_cost / args.n * 1000, 2)}

    # ---------- 2. 训练代理模型 ----------
    print(f"\n[2] 训练代理模型（PCA + MLP，{args.epochs} 轮）")
    model, info = fit_surrogate(theta_s, traj_s, spikes_s, epochs=args.epochs, seed=0)
    summary["surrogate"] = {
        "pca_k": int(model.head_coef.out_features),
        "pca_explained": info["pca_explained"],
        "train_cost_sec": round(info["train_cost_sec"], 2),
        "val_mse": info["val_mse"],
        "val_spike_mae": info["val_spike_mae"],
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }

    # ---------- 3. 保真度评估 ----------
    print("\n[3] 保真度评估（留出验证集）")
    val_idx = info["val_idx"]
    traj_v, sp_v = predict(model, theta_s[val_idx])
    mse_v = np.mean((traj_v - traj_s[val_idx]) ** 2, axis=1)
    rel_v = np.linalg.norm(traj_v - traj_s[val_idx], axis=1) / (
        np.linalg.norm(traj_s[val_idx], axis=1) + 1e-12)
    sp_ok = np.rint(sp_v).astype(int) == spikes_s[val_idx]
    sp_pm1 = np.abs(np.rint(sp_v).astype(int) - spikes_s[val_idx]) <= 1
    print(f"  验证集 {len(val_idx)} 条：MSE 中位数={np.median(mse_v):.3e}，"
          f"最大={mse_v.max():.3e}，相对误差中位数={np.median(rel_v)*100:.2f}%")
    print(f"  发放次数：完全正确 {sp_ok.mean()*100:.1f}%，±1 以内 {sp_pm1.mean()*100:.1f}%")
    summary["fidelity"] = {
        "n_val": int(len(val_idx)),
        "mse_median": float(np.median(mse_v)), "mse_max": float(mse_v.max()),
        "rel_err_median": float(np.median(rel_v)),
        "spike_exact_rate": float(sp_ok.mean()),
        "spike_pm1_rate": float(sp_pm1.mean()),
    }

    # 误差 vs 到训练集的最近距离（外推风险的量化）
    lo, hi = config.bounds_as_array()
    un = (theta_s - lo) / (hi - lo)
    dist = np.linalg.norm(un[val_idx][:, None, :] - un[info["train_idx"]][None, :, :], axis=2).min(axis=1)
    summary["fidelity"]["nn_dist_median"] = float(np.median(dist))

    # 误差到底由什么决定：离训练样本的远近，还是动力学区制（发放次数）？
    from scipy.stats import spearmanr
    sp_val = spikes_s[val_idx]
    rho_d, p_d = spearmanr(dist, rel_v)
    rho_s, p_s = spearmanr(sp_val, rel_v)
    by_regime = {}
    for sval in sorted(set(sp_val.tolist())):
        m = sp_val == sval
        by_regime[int(sval)] = {"n": int(m.sum()),
                                "rel_median": float(np.median(rel_v[m])),
                                "rel_max": float(rel_v[m].max())}
    print(f"  误差 vs 离训练集距离：Spearman rho={rho_d:+.3f} (p={p_d:.3f})")
    print(f"  误差 vs 发放次数    ：Spearman rho={rho_s:+.3f} (p={p_s:.3f})")
    print("  分区制误差中位数：", {k: f"{v['rel_median']*100:.1f}%" for k, v in by_regime.items()})
    summary["error_attribution"] = {
        "rho_vs_nn_distance": float(rho_d), "p_vs_nn_distance": float(p_d),
        "rho_vs_spike_count": float(rho_s), "p_vs_spike_count": float(p_s),
        "by_regime": by_regime,
    }

    # ---------- 4. 代理模型 + 梯度下降（坏初值 / 好初值 × 两种损失） ----------
    print("\n[4] 代理模型 + 梯度下降")
    gd_cases = [
        ("代理GD·坏初值·仅MSE",   config.GD_INIT_BAD,  0.0,  "bad_mse"),
        ("代理GD·坏初值·+软发放", config.GD_INIT_BAD,  0.05, "bad_spike"),
        ("代理GD·好初值·仅MSE",   config.GD_INIT_GOOD, 0.0,  "good_mse"),
    ]
    gd_runs = {}
    for tag, init_dict, lam, key in gd_cases:
        print(f"\n  — {tag}（初值 {init_dict}）—")
        init_vec = config.theta_to_vec(init_dict)
        th_gd, hist_gd, cost_gd, _ = run_gd_on_surrogate(
            model, y_obs, init_vec, steps=args.gd_steps,
            n_spikes_true=n_spikes_true, lam_spike=lam, tag=tag)
        print(f"  耗时 {cost_gd * 1000:.0f} ms（优化过程真实 ODE 调用 0 次）")
        v = verify_on_real_ode(th_gd, t, y_obs, n_spikes_true, tag)
        gd_runs[tag] = {"theta": th_gd, "hist": hist_gd, "cost": cost_gd, "v": v, "lam": lam}
        summary["gd_" + key] = {
            "init": init_dict, "lam_spike": lam, "steps": args.gd_steps,
            "cost_sec": round(cost_gd, 4), "ode_calls_during_opt": 0,
            "loss_final": float(hist_gd[-1]),
            **{k: vv for k, vv in v.items() if k != "V"},
        }

    # ---------- 4b. 对照组：白盒 ODE + GD（同一个好初值） ----------
    wb_gd = None
    if args.whitebox_gd:
        from gd import run_gd as run_gd_whitebox
        print(f"\n[4b] 对照组：白盒 ODE + GD（好初值，与代理版同一初值，{args.wb_epochs} 轮）")
        t0 = time.time()
        try:
            res_wb = run_gd_whitebox(t, y_obs, n_spikes_true, init="good",
                                     epochs=args.wb_epochs, verbose=True)
            cost_wb = time.time() - t0
            th_wb = np.asarray(res_wb["final_theta"], dtype=np.float64)
            diverged = not np.all(np.isfinite(th_wb))
            v = verify_on_real_ode(th_wb, t, y_obs, n_spikes_true, "白盒GD·好初值")
            n_wb = v["n_spikes_real_ode"]
            wb_gd = {"theta": th_wb, "cost": cost_wb, "v": v, "diverged": diverged,
                     "loss_hist": res_wb["loss_history"]}
            summary["wb_gd_good"] = {
                "init": config.GD_INIT_GOOD, "epochs": args.wb_epochs,
                "cost_sec": round(cost_wb, 2), "ode_calls_during_opt": args.wb_epochs,
                "diverged": bool(diverged),
                "loss_first": float(res_wb["loss_history"][0]),
                "loss_last": float(res_wb["loss_history"][-1]),
                **{k: vv for k, vv in v.items() if k != "V"},
            }
        except Exception as e:                     # 数值爆炸时可能直接抛异常
            cost_wb = time.time() - t0
            print(f"  白盒 GD 好初值运行中断：{type(e).__name__}: {e}")
            summary["wb_gd_good"] = {"init": config.GD_INIT_GOOD, "epochs": args.wb_epochs,
                                     "cost_sec": round(cost_wb, 2), "diverged": True,
                                     "error": f"{type(e).__name__}: {e}"}

    # ---------- 5. 代理模型 + NSGA-II ----------
    print(f"\n[5] 代理模型 + NSGA-II（种群={args.pop}, 代数={args.gen}）")
    orig_objectives = ga_mod.objectives           # 备份原目标函数
    ga_mod.objectives = make_surrogate_objectives(model)   # 运行时替换，不改原文件
    try:
        t0 = time.time()
        res_ga = ga_mod.run_ga(t, y_obs, n_spikes_true,
                               pop_size=args.pop, generations=args.gen, verbose=True)
        cost_ga = time.time() - t0
    finally:
        ga_mod.objectives = orig_objectives       # 恢复
    print(f"  耗时 {cost_ga:.2f}s（优化过程真实 ODE 调用 0 次）")
    ga_s = verify_on_real_ode(res_ga["best_theta"], t, y_obs, n_spikes_true, "代理GA")
    summary["ga_surrogate"] = {
        "pop": args.pop, "generations": args.gen, "cost_sec": round(cost_ga, 2),
        "ode_calls_during_opt": 0,
        "pareto_size": int(len(res_ga["pareto_theta"])),
        "surrogate_best_objectives": res_ga["best_objectives"].tolist(),
        **{k: v for k, v in ga_s.items() if k != "V"},
    }
    # 帕累托前沿用真实 ODE 复核（抽前 5 个）
    pareto_verified = []
    order = np.argsort(res_ga["pareto_objectives"][:, 0])[:5]
    for i in order:
        th = res_ga["pareto_theta"][i]
        V = fhn_simulate(th, t, config)
        pareto_verified.append({
            "theta": th.tolist(),
            "mse_real_ode": float(np.mean((V - y_obs) ** 2)),
            "n_spikes": int(count_spikes(V)),
        })
    summary["ga_surrogate"]["pareto_verified_top5"] = pareto_verified

    # ---------- 6. 外推测试 ----------
    print("\n[6] 外推测试：故意问代理模型「它没见过的参数」")
    lo, hi = config.bounds_as_array()
    name_map = {n: i for i, n in enumerate(config.PARAM_NAMES)}

    # 分两类：A) 越界但参考模型仍有意义；B) 越界到参考模型自己数值失效
    cases = [
        ("amp 推到 1.8（上界 1.5）", {"amp": 1.8}),
        ("a  推到 1.9（上界 1.5）", {"a": 1.9}),
        ("b  压到 0.05（下界 0.1）", {"b": 0.05}),
        ("eps 压到 0.006（下界 0.01）", {"eps": 0.006}),
    ]
    out_pts, out_names = [], []
    for label, mod in cases:
        p = theta_true.copy()
        for k, v in mod.items():
            p[name_map[k]] = v
        out_pts.append(p)
        out_names.append(label)
    out_pts = np.array(out_pts)

    in_pts = theta_s[val_idx][:len(out_pts)]
    tr_out, sp_out = predict(model, out_pts)
    tr_in, _ = predict(model, in_pts)
    real_out = np.vstack([fhn_simulate(p, t, config) for p in out_pts])
    real_in = np.vstack([fhn_simulate(p, t, config) for p in in_pts])

    rel_out = np.linalg.norm(tr_out - real_out, axis=1) / (np.linalg.norm(real_out, axis=1) + 1e-12)
    rel_in = np.linalg.norm(tr_in - real_in, axis=1) / (np.linalg.norm(real_in, axis=1) + 1e-12)

    spikes_real = [int(count_spikes(v)) for v in real_out]
    spikes_pred = np.rint(sp_out).astype(int).tolist()

    print(f"  箱内（正常区间）相对误差：{np.round(rel_in * 100, 2)} %")
    print("  箱外：")
    extrap_rows = []
    for i, nm in enumerate(out_names):
        sane = 0 <= spikes_real[i] <= 12          # 参考模型是否还讲道理
        tag = "参考模型有效" if sane else "参考模型已数值失效"
        print(f"    · {nm:<26} 相对误差 {rel_out[i]*100:7.2f}% | "
              f"代理预测发放 {spikes_pred[i]} vs 真实 {spikes_real[i]}  [{tag}]")
        extrap_rows.append({
            "case": nm, "theta": out_pts[i].tolist(),
            "rel_err": float(rel_out[i]),
            "spike_pred": int(spikes_pred[i]), "spike_true": int(spikes_real[i]),
            "reference_valid": bool(sane),
        })

    halluc = [r for r in extrap_rows if not r["reference_valid"]]
    if halluc:
        print(f"  → {len(halluc)} 个点在参考模型已经崩溃处，代理模型仍给出了「看起来正常」的输出，"
              f"这就是外推幻觉。")

    summary["extrapolation"] = {
        "in_box_rel_err": rel_in.tolist(),
        "cases": extrap_rows,
        "out_box_spike_pred": spikes_pred,
        "out_box_spike_true": spikes_real,
    }

    # ---------- 7. 与已有基线合并成对照表 ----------
    try:
        with open(os.path.join(config.RESULTS_DIR, "summary.json"), encoding="utf-8") as f:
            base = json.load(f)
    except Exception:
        base = {}
    table = []
    if "gd_bad" in base:
        table.append({"method": "白盒ODE + GD(坏初值)", "ode_calls": config.GD_EPOCHS,
                      "total_sec": base["gd_bad"]["runtime_sec"],
                      "opt_sec": base["gd_bad"]["runtime_sec"],
                      "f_mse": base["gd_bad"]["final_objectives"][0],
                      "n_spikes": None, "note": "越界/失败"})
    if wb_gd is not None:
        v = wb_gd["v"]
        ok = v["n_spikes_real_ode"] == n_spikes_true
        if wb_gd["diverged"] or v["mse_real_ode"] > 1e5:
            note = "发散/越界"
        elif ok:
            note = "成功"
        else:
            note = "发放次数错"
        table.append({"method": "白盒ODE + GD(好初值)", "ode_calls": args.wb_epochs,
                      "total_sec": round(wb_gd["cost"], 2), "opt_sec": round(wb_gd["cost"], 2),
                      "f_mse": v["mse_real_ode"], "n_spikes": v["n_spikes_real_ode"],
                      "note": note})
    if "ga" in base:
        table.append({"method": "白盒ODE + NSGA-II", "ode_calls": 8100,
                      "total_sec": base["ga"]["runtime_sec"],
                      "opt_sec": base["ga"]["runtime_sec"],
                      "f_mse": base["ga"]["best_objectives"][0],
                      "n_spikes": int(n_spikes_true), "note": "成功"})
    for tag, r in gd_runs.items():
        v = r["v"]
        ok = v["n_spikes_real_ode"] == n_spikes_true
        table.append({"method": tag, "ode_calls": args.n,
                      "total_sec": round(data_cost + info["train_cost_sec"] + r["cost"], 2),
                      "opt_sec": round(r["cost"], 4),
                      "f_mse": v["mse_real_ode"], "n_spikes": v["n_spikes_real_ode"],
                      "note": "成功" if ok else "发放次数错"})
    ok_ga = ga_s["n_spikes_real_ode"] == n_spikes_true
    table.append({"method": "代理 + NSGA-II", "ode_calls": args.n,
                  "total_sec": round(data_cost + info["train_cost_sec"] + cost_ga, 2),
                  "opt_sec": round(cost_ga, 2),
                  "f_mse": ga_s["mse_real_ode"], "n_spikes": ga_s["n_spikes_real_ode"],
                  "note": "成功" if ok_ga else "发放次数错"})

    print("\n" + "=" * 96)
    print(f"{'方法':<26}{'ODE次数':>9}{'总墙钟(s)':>11}{'优化段(s)':>11}"
          f"{'真实MSE':>12}{'发放':>6}  结论")
    print("-" * 96)
    for r in table:
        m = r["method"] + " " * max(0, 26 - sum(2 if ord(c) > 127 else 1 for c in r["method"]))
        print(f"{m}{r['ode_calls']:>9}{r['total_sec']:>11.2f}{r['opt_sec']:>11.2f}"
              f"{r['f_mse']:>12.3e}{str(r['n_spikes']):>6}  {r['note']}")
    print("=" * 96)

    # ---------- 8. 存盘 ----------
    with open(os.path.join(OUT_DIR, "summary_surrogate.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)

    np.savez(os.path.join(OUT_DIR, "artifacts.npz"),
             theta_sample=theta_s, spikes_sample=spikes_s,
             val_idx=val_idx, train_idx=info["train_idx"],
             val_mse=mse_v, val_rel=rel_v, val_nn_dist=dist,
             gd_hist_bad=gd_runs["代理GD·坏初值·仅MSE"]["hist"],
             theta_gd_bad=gd_runs["代理GD·坏初值·仅MSE"]["theta"],
             V_gd_bad=gd_runs["代理GD·坏初值·仅MSE"]["v"]["V"],
             gd_hist_bad_spike=gd_runs["代理GD·坏初值·+软发放"]["hist"],
             theta_gd_bad_spike=gd_runs["代理GD·坏初值·+软发放"]["theta"],
             V_gd_bad_spike=gd_runs["代理GD·坏初值·+软发放"]["v"]["V"],
             gd_hist_good=gd_runs["代理GD·好初值·仅MSE"]["hist"],
             theta_gd_good=gd_runs["代理GD·好初值·仅MSE"]["theta"],
             V_gd_good=gd_runs["代理GD·好初值·仅MSE"]["v"]["V"],
             wb_gd_loss=(wb_gd["loss_hist"] if wb_gd is not None else np.zeros(1)),
             wb_gd_V=(wb_gd["v"]["V"] if wb_gd is not None else np.zeros(1)),
             theta_ga=res_ga["best_theta"], V_ga=ga_s["V"],
             pareto_theta=res_ga["pareto_theta"],
             pareto_objectives=res_ga["pareto_objectives"],
             surrogate_traj_val=traj_v, true_traj_val=traj_s[val_idx],
             out_pts=out_pts, out_pred=tr_out, out_true=real_out,
             in_pred=tr_in, in_true=real_in,
             loss_history=info["loss_history"])

    # 存一份代理模型权重，便于复用
    import torch
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "surrogate.pt"))

    print(f"\n完成。结果在 {OUT_DIR}")
    return summary


if __name__ == "__main__":
    main()
