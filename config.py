# -*- coding: utf-8 -*-
"""
全局配置：模型超参数、真实(Ground Truth)参数、起搏协议、求解器设置与路径。

本项目的目标：用一个「极简但数学特性等效」的模型，复现 T-World 这类白盒 ODE
心肌细胞模型为什么不能用梯度下降、而要用多目标遗传算法(MOGA)来标定参数。
详见 train.py 顶部说明。
"""
import os

# ============ 路径 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ============ 模型 / 仿真 ============
# 使用 FitzHugh-Nagumo(FHN) 型可兴奋系统——神经/心肌可兴奋细胞的最小白盒模型：
#   eps * dV/dt = V - V^3/3 - w + I(t)     (快变量：类比快速 Na/Ca 电流，时间尺度 ~ eps)
#   dw/dt       = V + a - b*w              (慢变量：类比慢门控恢复，时间尺度 ~ 1)
# eps << 1 造成「刚性(stiffness)」：快慢时间尺度相差 ~1/eps 倍。
# 系统具有「阈值」：刺激低于阈值则不发放(亚阈值)，高于阈值则产生尖峰(动作电位)，
# 这正对应心肌细胞的「发作或是不发作 EAD/DAD」这类离散/不可导现象。
DT = 0.005          # 固定步长前向欧拉积分步长（保证 eps 尺度下的数值稳定，刚性系统需要小步长）
T_END = 60.0        # 总仿真时长（时间单位任意，看作 ms 即可）

# 起搏协议：周期性脉冲刺激（模拟实验中的电起搏）。脉冲幅度「是否作为待拟合参数」由
# THETA_TRUE / BOUNDS 中的 'amp' 决定——这里把它当作一个「可兴奋性门控强度」的集中描述。
PACE_CL = 15.0      # 起搏周期 (cycle length)
PACE_DUR = 1.0      # 每周期脉冲宽度（短脉冲 -> 健康细胞 1:1 发放）

# 动作电位(尖峰)检测阈值：V 上穿该值记为一次发放
SPIKE_THRESHOLD = 0.0

# ============ 真实(Ground Truth)参数 ============
# 待拟合参数顺序固定为 [eps, a, b, amp]
PARAM_NAMES = ["eps", "a", "b", "amp"]

THETA_TRUE = dict(eps=0.02, a=0.8, b=0.8, amp=0.8)

# 参数搜索边界（GA 用；也可用于对 GD 做 box 约束）。顺序与 PARAM_NAMES 一致。
# 注意 eps 下界 > DT/2，保证固定步长前向欧拉在边界内数值稳定。
BOUNDS = {
    "eps": (0.01, 0.2),
    "a":   (0.0, 1.5),
    "b":   (0.1, 1.5),
    "amp": (0.1, 1.5),
}

# 梯度下降「坏」初值：处于「自发放电(类心律失常)」状态，远离真实值，
# 用于展示 GD 收敛到「错误发放模式」的局部极小（无法跨越阈值切换到正确发放数）。
GD_INIT_BAD = dict(eps=0.05, a=0.3, b=0.4, amp=0.8)
# 梯度下降「好」初值：贴近真实值，用于对照展示「即便有好初值，刚性导致梯度爆炸/发散」。
GD_INIT_GOOD = dict(eps=0.02, a=0.75, b=0.75, amp=0.75)

# 合成数据的观测噪声标准差
NOISE_STD = 0.02

# ============ 梯度下降 (PyTorch Adam) ============
GD_LR = 0.01
GD_EPOCHS = 100         # 完整训练轮数。GD 在约 50 轮就卡死在局部极小（这正是要展示的现象），
                        # 更多轮数只是重复同样的平台，故默认 100（约 3 分钟）。可用 --epochs 调大。
# 目标缩放：多目标本应是一个向量，但 GD 天生是单目标，这里把它标量化。
# GD 只能优化「连续可导的 MSE」，无法处理「发放/不发放」这种阶跃式目标。
GD_LOSS_WEIGHTS = dict(mse=1.0)

# ============ 遗传算法 (手写 NSGA-II) ============
GA_POP = 100            # 种群大小
GA_GENERATIONS = 80     # 迭代代数（完整训练）。冒烟测试时可调小。
GA_CROSSOVER_PROB = 0.9 # 交叉概率
GA_MUTATION_PROB = 0.3  # 每个基因的变异概率
GA_MUTATION_SIGMA = 0.1 # 高斯变异标准差（占该参数搜索范围的 10%）
GA_SEED = 0
GD_SEED = 0


def theta_to_vec(theta_dict):
    """把 dict 形式的参数按 PARAM_NAMES 顺序转成 numpy 可用的顺序。"""
    import numpy as np
    return np.array([theta_dict[n] for n in PARAM_NAMES], dtype=np.float64)


def bounds_as_array():
    """返回 (lower, upper) 两个数组，顺序与 PARAM_NAMES 一致。"""
    import numpy as np
    lo = np.array([BOUNDS[n][0] for n in PARAM_NAMES], dtype=np.float64)
    hi = np.array([BOUNDS[n][1] for n in PARAM_NAMES], dtype=np.float64)
    return lo, hi