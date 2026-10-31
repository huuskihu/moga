# -*- coding: utf-8 -*-
"""
真·代理模型（Surrogate）：输入 4 个模型参数 theta，输出整条电压轨迹 V(t) + 发放次数。

与 neural.py 的区别（关键）：
  neural.py   学的是  (t, 刺激) -> V(t)，只拟合「一条」固定参数的轨迹，
              换个参数就废了，不能替代 ODE 求解器。
  本文件       学的是  theta -> V(t)，一次训练覆盖整个参数空间，
              训练好之后可以当作 ODE 的「平替」，反复调参不再求解微分方程。

技术路线：
  1. 在参数边界内做拉丁超立方采样 N 组 theta；
  2. 用 model.fhn_simulate_batch 批量跑真实 ODE，得到 N 条轨迹（这一步仍然受刚性折磨）；
  3. 对轨迹做 PCA 降到 K 维（轨迹本质由 4 个参数决定，低维流形），
     代理模型只需预测 K 个系数，再线性重建整条曲线；
  4. 另接一个头预测「发放次数」（离散量，推理时四舍五入）。
"""
import numpy as np
import torch

import config
from model import fhn_simulate_batch, count_spikes_multi
from data import make_time_grid

K_PCA = 64       # PCA 主成分个数
HIDDEN = 256     # MLP 隐层宽度
N_TRAIN = 400    # 训练集采样点数（用户设想的 200~1000 组，取 400）
N_FEAT = 5       # 输入特征维度（4 个参数 + log(eps)，见 _features）


# ---------------------------------------------------------------------------
# 1. 采样 + 造数据（这一步是唯一需要真实 ODE 的地方）
# ---------------------------------------------------------------------------
def sample_theta(n, seed=0):
    """参数空间采样。优先拉丁超立方(LHS)，覆盖更均匀；scipy 缺失则退化为均匀随机。"""
    lo, hi = config.bounds_as_array()
    try:
        from scipy.stats import qmc
        u = qmc.LatinHypercube(d=len(lo), seed=seed).random(n)
    except Exception:
        u = np.random.default_rng(seed).random((n, len(lo)))
    return lo + u * (hi - lo)


def build_dataset(n=N_TRAIN, seed=0, t=None, verbose=True):
    """
    跑真实 ODE 造训练集。
    返回 theta (N,4)、traj (N,n)、spikes (N,)、耗时(秒)。
    """
    if t is None:
        t = make_time_grid()
    theta = sample_theta(n, seed)

    import time
    t0 = time.time()
    traj = fhn_simulate_batch(theta, t, config)
    cost = time.time() - t0

    spikes = count_spikes_multi(traj)
    if verbose:
        print(f"  [造数据] {n} 组参数 × 真实 ODE：{cost:.2f}s "
              f"（发放次数分布 {dict(zip(*np.unique(spikes, return_counts=True)))}）")
    return theta, traj, spikes, cost


# ---------------------------------------------------------------------------
# 2. 代理模型
# ---------------------------------------------------------------------------
class TrajSurrogate(torch.nn.Module):
    """theta -> (轨迹, 发放次数)。全程矩阵乘法 + tanh，没有任何时间积分，因此不存在刚性。"""

    def __init__(self, lo, hi, pca_mean, pca_comps, coef_std, k=K_PCA, hidden=HIDDEN):
        super().__init__()
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32))
        self.register_buffer("pca_mean", torch.tensor(pca_mean, dtype=torch.float32))
        # pca_comps: (k, n)，重建时 coef @ comps
        self.register_buffer("pca_comps", torch.tensor(pca_comps, dtype=torch.float32))
        self.register_buffer("coef_std", torch.tensor(coef_std, dtype=torch.float32))
        # log(eps) 的归一化常数：eps 在箱内跨约 20 倍，取对数后动力学才接近线性
        self.register_buffer("log_lo", torch.log(self.lo[0].clone()))
        self.register_buffer("log_hi", torch.log(self.hi[0].clone()))

        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(N_FEAT, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
        )
        self.head_coef = torch.nn.Linear(hidden, k)
        self.head_spike = torch.nn.Linear(hidden, 1)

    def _features(self, theta):
        """原始参数 -> 网络输入。额外加 log(eps)：刚性强度对 eps 是对数敏感的。"""
        u = (theta - self.lo) / (self.hi - self.lo)
        log_eps = (torch.log(theta[:, 0:1].clamp_min(1e-8)) - self.log_lo) / (self.log_hi - self.log_lo)
        return torch.cat([u, log_eps], dim=1)

    def encode(self, theta):
        """只算 PCA 系数和发放次数，不重建整条轨迹（用于快速算 MSE）。"""
        h = self.trunk(self._features(theta))
        coef = self.head_coef(h) * self.coef_std
        spike = self.head_spike(h).squeeze(-1)
        return coef, spike

    def forward(self, theta):
        """theta: (B,4) 原始量纲。返回 (B,n) 轨迹、(B,) 发放次数（连续值，未取整）。"""
        coef, spike = self.encode(theta)
        traj = self.pca_mean + coef @ self.pca_comps
        return traj, spike

    def mse_against(self, theta, y_true):
        """
        直接在 PCA 子空间里算与 y_true 的 MSE，避免重建 12001 维轨迹。
        因 PCA 基正交，||mean + c·U - y||² = ||c - proj||² + ||r||² - ||proj||²，与全维 MSE 完全等价。
        """
        n = len(y_true)
        r = y_true - self.pca_mean
        proj = r @ self.pca_comps.t()                      # (k,)
        const = (r @ r - proj @ proj) / n
        coef, spike = self.encode(theta)
        mse = torch.mean((coef - proj) ** 2, dim=1) * (len(proj) / n) + const
        return mse, spike


def fit_surrogate(theta, traj, spikes, n_val=60, epochs=2000, lr=3e-3,
                  batch=64, seed=0, verbose=True):
    """训练代理模型。按索引顺序切分训练/验证（LHS 采样本身与顺序无关，可直接切）。"""
    torch.manual_seed(seed)          # 固定随机性：保证同参数重复运行结果一致
    rng = np.random.default_rng(seed)
    N = len(theta)
    idx = rng.permutation(N)
    n_val = min(n_val, N // 5)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    # ---- PCA：只在训练集上拟合，避免信息泄露 ----
    from sklearn.decomposition import PCA
    pca = PCA(n_components=K_PCA, svd_solver="randomized", random_state=seed)
    pca.fit(traj[tr_idx])
    comps = pca.components_                       # (k, n)
    mean = pca.mean_                            # (n,)
    # 系数标准化：让 MLP 的输出尺度在 O(1)
    coef_tr = pca.transform(traj[tr_idx])
    coef_std = coef_tr.std(axis=0) + 1e-8

    expl = float(pca.explained_variance_ratio_.sum())
    if verbose:
        print(f"  [PCA] {K_PCA} 个主成分解释了轨迹方差的 {expl * 100:.4f}%")

    lo, hi = config.bounds_as_array()
    model = TrajSurrogate(lo, hi, mean, comps, coef_std)

    th_tr = torch.tensor(theta[tr_idx], dtype=torch.float32)
    y_tr = torch.tensor(traj[tr_idx], dtype=torch.float32)
    s_tr = torch.tensor(spikes[tr_idx], dtype=torch.float32)

    th_va = torch.tensor(theta[val_idx], dtype=torch.float32)
    y_va = torch.tensor(traj[val_idx], dtype=torch.float32)
    s_va = torch.tensor(spikes[val_idx], dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ntr = len(tr_idx)
    hist = []
    import time
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(ntr)
        tot = 0.0
        for i in range(0, ntr, batch):
            b = perm[i:i + batch]
            traj_p, sp_p = model(th_tr[b])
            loss = torch.mean((traj_p - y_tr[b]) ** 2) + 0.1 * torch.mean((sp_p - s_tr[b]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        hist.append(tot / ntr)
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print(f"  代理训练 {ep:4d}/{epochs}: train_loss={hist[-1]:.3e}")
    train_cost = time.time() - t0

    model.eval()
    with torch.no_grad():
        traj_v, sp_v = model(th_va)
        val_mse = float(torch.mean((traj_v - y_va) ** 2))
        val_sp_mae = float(torch.mean(torch.abs(sp_v - s_va)))
    if verbose:
        print(f"  [代理] 训练耗时 {train_cost:.2f}s | 验证集 MSE={val_mse:.3e} | "
              f"发放次数 MAE={val_sp_mae:.2f}")

    return model, {
        "val_idx": val_idx, "train_idx": tr_idx,
        "val_mse": val_mse, "val_spike_mae": val_sp_mae,
        "pca_explained": expl, "train_cost_sec": train_cost,
        "loss_history": np.array(hist),
    }


def predict(model, theta, chunk=512):
    """numpy 批量预测（自动分块，防止 N×n 太大）。返回 traj (B,n), spike (B,)。"""
    model.eval()
    theta = np.asarray(theta, dtype=np.float64).reshape(-1, 4)
    outs_t, outs_s = [], []
    with torch.no_grad():
        for i in range(0, len(theta), chunk):
            th = torch.tensor(theta[i:i + chunk], dtype=torch.float32)
            tr, sp = model(th)
            outs_t.append(tr.numpy())
            outs_s.append(sp.numpy())
    return np.vstack(outs_t), np.concatenate(outs_s)
