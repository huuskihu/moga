# -*- coding: utf-8 -*-
"""
手写多目标遗传算法 NSGA-II（实数编码，不依赖 deap/scipy）。

之所以用遗传算法而不是梯度下降：T-World 的开发用「多目标遗传算法」去标定单个
组件的参数、以及整合组件（见原论文 Part I 的方法部分）。遗传算法的关键优势：
  1. 不需要导数 —— 阶跃式/不可导目标(f2)也能直接优化。
  2. 天然处理硬约束(参数边界) —— 越界个体直接判为不可行即可。
  3. 通过变异「跨越阈值」—— 发放/不发放的二值惩罚能被一步突变跨越。
  4. 多目标 —— 用非支配排序 + 拥挤度直接给出一组帕累托解，而非强行加权成标量。

流程：初始化种群 -> 评估目标 -> 非支配排序+拥挤度 -> 锦标赛选择 -> SBX 交叉
     -> 高斯变异 -> 精英合并 -> 截断保留，迭代 GA_GENERATIONS 代。
"""
import numpy as np

import config
from objective import objectives


def fast_non_dominated_sort(f):
    """
    快速非支配排序。f：形状 (N, M) 的目标矩阵（最小化）。
    返回 front_indices 列表，每个元素是该层前沿的个体下标集合（list）。
    """
    N = f.shape[0]
    domination_count = np.zeros(N, dtype=np.int64)
    dominated_sets = [[] for _ in range(N)]
    fronts = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            # p 是否支配 q（两个目标都 <= 且至少一个严格 <）
            if np.all(f[p] <= f[q]) and np.any(f[p] < f[q]):
                dominated_sets[p].append(q)
            elif np.all(f[q] <= f[p]) and np.any(f[q] < f[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_sets[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    # 去掉最后一个空前沿
    if len(fronts[-1]) == 0:
        fronts.pop()
    return fronts


def crowding_distance(front_indices, f):
    """计算某个前沿内个体的拥挤度距离（用于保持解的多样性）。"""
    n = len(front_indices)
    M = f.shape[1]
    if n == 0:
        return np.array([])
    dist = np.zeros(n)

    for m in range(M):
        order = np.argsort(f[front_indices, m])
        f_sorted = f[np.array(front_indices)[order], m]
        dist_sorted = np.zeros(n)
        if f_sorted[-1] - f_sorted[0] > 1e-12:
            dist_sorted[0] = np.inf
            dist_sorted[-1] = np.inf
            for j in range(1, n - 1):
                dist_sorted[j] += (f_sorted[j + 1] - f_sorted[j - 1]) / (f_sorted[-1] - f_sorted[0])
        for j in range(n):
            dist[order[j]] += dist_sorted[j]
    return dist


def tournament_selection(pop, f, k=2):
    """二元锦标赛选择：先比支配等级，再比拥挤度。返回选中的下标。"""
    N = pop.shape[0]
    fronts = fast_non_dominated_sort(f)
    rank = np.zeros(N, dtype=np.int64)
    crowd = np.zeros(N)
    for i, fr in enumerate(fronts):
        rank[list(fr)] = i
        cd = crowding_distance(list(fr), f)
        for j, idx in enumerate(fr):
            crowd[idx] = cd[j]

    sel = []
    for _ in range(N):
        cands = np.random.choice(N, k, replace=False)
        best = cands[0]
        for c in cands[1:]:
            if rank[c] < rank[best] or (rank[c] == rank[best] and crowd[c] > crowd[best]):
                best = c
        sel.append(best)
    return np.array(sel)


def sbx_crossover(parent1, parent2, lo, hi, eta=15.0, prob=0.9):
    """模拟二进制交叉(SBX)：对每个基因以 prob 概率交叉。"""
    c1, c2 = parent1.copy(), parent2.copy()
    for i in range(len(parent1)):
        if np.random.rand() > prob:
            continue
        if abs(parent1[i] - parent2[i]) < 1e-12:
            continue
        u = np.random.rand()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        c1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
        c2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        c1[i] = np.clip(c1[i], lo[i], hi[i])
        c2[i] = np.clip(c2[i], lo[i], hi[i])
    return c1, c2


def gaussian_mutation(ind, lo, hi, prob=0.3, sigma_frac=0.1):
    """高斯变异：每个基因以 prob 概率加一个按参数范围缩放的扰动。"""
    rng = np.arange(len(ind))
    scale = (hi - lo) * sigma_frac
    mask = np.random.rand(len(ind)) < prob
    noise = np.random.normal(0.0, 1.0, size=len(ind)) * scale * mask
    ind = ind + noise
    return np.clip(ind, lo, hi)


def run_ga(t, y_true, n_spikes_true, pop_size=None, generations=None, seed=None,
           verbose=True):
    """
    运行 NSGA-II 标定参数，返回字典包含：
      - best_theta          : 最终种群中 f2=0 且 f1 最小的个体
      - best_objectives     : 其目标向量
      - pareto_theta        : 末代非支配前沿的参数
      - pareto_objectives   : 末代非支配前沿的目标
      - final_population    : 末代种群参数 (P, 4)
      - final_objectives    : 末代种群目标 (P, 2)
      - history_f1          : 历代 f1 最小值序列（用于收敛曲线）
      - history_feasible    : 历代可行(f2=0)个体占比序列
    """
    pop_size = pop_size or config.GA_POP
    generations = generations or config.GA_GENERATIONS
    rng = np.random.default_rng(seed if seed is not None else config.GA_SEED)
    np.random.seed(seed if seed is not None else config.GA_SEED)

    lo, hi = config.bounds_as_array()

    # 初始化：在边界内均匀随机
    pop = rng.uniform(lo, hi, size=(pop_size, len(lo)))
    f = objectives(pop, t, y_true, n_spikes_true)

    history_f1 = []
    history_feasible = []

    for gen in range(generations):
        # 1) 选择 -> 交叉 -> 变异 生成子代
        sel = tournament_selection(pop, f)
        offspring = np.empty_like(pop)
        for i in range(0, pop_size, 2):
            p1 = pop[sel[i]]
            p2 = pop[sel[(i + 1) % pop_size]]
            c1, c2 = sbx_crossover(p1, p2, lo, hi,
                                   prob=config.GA_CROSSOVER_PROB)
            c1 = gaussian_mutation(c1, lo, hi,
                                   prob=config.GA_MUTATION_PROB,
                                   sigma_frac=config.GA_MUTATION_SIGMA)
            c2 = gaussian_mutation(c2, lo, hi,
                                   prob=config.GA_MUTATION_PROB,
                                   sigma_frac=config.GA_MUTATION_SIGMA)
            offspring[i] = c1
            if i + 1 < pop_size:
                offspring[i + 1] = c2

        # 2) 评估子代，合并父代+子代做精英保留
        f_off = objectives(offspring, t, y_true, n_spikes_true)
        merged = np.vstack([pop, offspring])
        f_merged = np.vstack([f, f_off])

        # 3) 非支配排序 + 拥挤度，截断保留 pop_size 个
        fronts = fast_non_dominated_sort(f_merged)
        new_pop = []
        new_f = []
        fidx = 0
        while fidx < len(fronts) and len(new_pop) + len(fronts[fidx]) <= pop_size:
            fr = fronts[fidx]
            idxs = list(fr)
            new_pop.append(merged[idxs])
            new_f.append(f_merged[idxs])
            fidx += 1
        if len(new_pop) < pop_size:
            fr = fronts[fidx]
            idxs = list(fr)
            cd = crowding_distance(idxs, f_merged)
            order = np.argsort(-cd)
            need = pop_size - len(new_pop)
            take = [idxs[o] for o in order[:need]]
            new_pop.append(merged[take])
            new_f.append(f_merged[take])

        pop = np.vstack(new_pop)
        f = np.vstack(new_f)

        # 记录
        feasible_mask = f[:, 1] == 0.0
        history_feasible.append(float(feasible_mask.mean()))
        if feasible_mask.any():
            history_f1.append(float(f[feasible_mask, 0].min()))
        else:
            history_f1.append(float(f[:, 0].min()))

        if verbose and (gen % max(1, generations // 10) == 0 or gen == generations - 1):
            print(f"  GA 代数 {gen:3d}/{generations}: "
                  f"最小f1={history_f1[-1]:.5f}, 可行占比={history_feasible[-1]:.2f}")

    # 结果整理
    all_theta = pop
    all_f = f
    feasible_mask = all_f[:, 1] == 0.0
    if feasible_mask.any():
        cand = np.where(feasible_mask)[0]
        best_idx = cand[np.argmin(all_f[feasible_mask, 0])]
        best_theta = all_theta[best_idx]
        best_objectives = all_f[best_idx]
    else:
        best_idx = np.argmin(all_f[:, 0])
        best_theta = all_theta[best_idx]
        best_objectives = all_f[best_idx]

    # 末代帕累托前沿（第一层非支配）
    fronts = fast_non_dominated_sort(all_f)
    pareto_idx = list(fronts[0])
    pareto_theta = all_theta[pareto_idx]
    pareto_objectives = all_f[pareto_idx]

    return {
        "best_theta": best_theta,
        "best_objectives": best_objectives,
        "pareto_theta": pareto_theta,
        "pareto_objectives": pareto_objectives,
        "final_population": all_theta,
        "final_objectives": all_f,
        "history_f1": np.array(history_f1),
        "history_feasible": np.array(history_feasible),
    }