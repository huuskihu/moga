# moga — 梯度下降 vs 遗传算法：白盒 ODE 参数标定的极简复现

一个极简、可在个人电脑上复现的实验，回答 AI4Science 里的一个核心问题：

> 同样是「让模型拟合数据」，为什么 **T-World**（虚拟心肌细胞）用**多目标遗传算法**，
> 而 **手写数字识别器** 用**梯度下降**？

## 结论（一句话）

决定成败的是「**初值**」，不是「优化器」，也不是「**刚性**」。

白盒 ODE 上的梯度下降对初值极度敏感：**好初值下平滑收敛（MSE 4.00e-04，100 轮 121 s，无 NaN、无越界）**，
坏初值下则掉进「错误发放次数」的局部极小（28 次）并滑出参数盒（eps 0.04 → 0.00675）。
多目标遗传算法不依赖初值，所以稳健（MSE 4.32e-04，24 s）。
把「参数→输出」做成可微代理后，梯度下降 0.66 s 收敛到 1.53e-03——快 **184×**，精度受代理保真度限制。

## 模型

用 **FitzHugh–Nagumo 可兴奋系统**作为 T-World 的「极简等效替身」，保留三大特征：**白盒、刚性、阈值**。

```
eps·dV/dt = V − V³/3 − w + I(t)      # 快变量（类比快速电流）
dw/dt     = V + a − b·w              # 慢变量（类比慢门控恢复）
```

- `eps << 1` 引入刚性（快慢时间尺度差约 50 倍）。
- 系统具兴奋阈值，「发放/不发放」是二值事件，类比心肌细胞的早/延迟后除极（EAD/DAD）。
- 4 个待拟合参数 `[eps, a, b, amp]`，真实值 `[0.02, 0.8, 0.8, 0.8]` → 每次刺激 1 次发放（共 4 次）。
- 用已知参数生成合成数据 + 噪声作为 ground truth。

## 三种训练方式（同一份数据）

| 方式 | 结果 | 最终 MSE | 发放次数 |
| --- | --- | --- | --- |
| 梯度下降 + 白盒 ODE · **坏初值** | ❌ 错误发放 + 越界 | 2.06 | 28（应为 4） |
| 梯度下降 + 白盒 ODE · **好初值** | ✅ 平滑收敛 | **0.00040** | 4 |
| 多目标遗传算法 (NSGA-II) + 白盒 ODE | ✅ 恢复真值 | 0.00043 | 4 |
| 梯度下降 + 可微**代理模型**（θ→V，`run_surrogate.py`） | ✅ 0.66 s | 0.00153 | 4 |
| 梯度下降 + `neural.py`（t→V，仅拟合曲线） | ⚠️ 收敛但**不反推参数** | 0.061 | — |

> 前四行的 MSE 均由「参数丢回真实 ODE」复核；最后一行是曲线回归，输出 33,537 个网络权重而
> 非 4 个参数，**不能作为「GD 可行」的对照证据**。详见 `代理模型实验报告.md`。

## 文件结构

| 文件 | 说明 |
| --- | --- |
| `config.py` | 全局配置：模型、真实参数、边界、求解器超参数 |
| `model.py` | FHN 模型（numpy / 批量 / torch 三套前向） |
| `data.py` | 合成 ground truth 的生成与校验 |
| `objective.py` | 多目标损失（向量 `[f_MSE, f_fire]` + 硬约束） |
| `gd.py` | 梯度下降求解器（PyTorch Adam） |
| `ga.py` | 手写 NSGA-II（不依赖 deap/scipy） |
| `neural.py` | 可微代理（t→V 曲线回归）——**不反推参数，不能顶替 ODE 打分** |
| `surrogate.py` | 参数→轨迹代理模型（θ→V，PCA 64 维 + MLP，15 万参数） |
| `run_surrogate.py` | 代理实验驱动：保真度 / 误差归因 / GD×3 / 白盒 GD 对照 / NSGA-II / 外推 |
| `exp_support.py` | 支撑实验：计时重复 5 轮 + 地形多切片 |
| `exp_seeds.py` | 支撑实验：多随种子（GD 好初值 vs NSGA-II，5 组） |
| `visualize_surrogate.py` | 保真度四联图 |
| `make_matrix_fig.py` | 2×3 对照矩阵图 |
| `make_landscape_fig.py` | 势垒 + 断崖地形图 |
| `train.py` | 训练入口 / 启动器（代码与训练分离） |
| `visualize.py` | 原始三种方式的可视化 |
| `make_report.py` | 一键生成三张分图 + 11 页 PPTX |
| `results/` | 原始三种方式的输出（npz + 图） |
| `results_surrogate/` | 代理实验输出（json + 图）；`artifacts.npz` / `surrogate.pt` 不入库，可重生成 |
| `代理模型实验报告.md` | 代理实验完整报告，含「假设推翻」复盘 |
| `双解释_逐句有据.md` | 11 节「通俗 / 专业 / 支撑」三行对照，每句带出处 |
| `T-World_GD_vs_GA_报告.pptx` | 11 页报告 |
| `讲稿.md` | 配套讲稿（按 11 页组织，含备课附录） |

## 快速开始

```bash
pip install numpy scipy matplotlib torch python-pptx pillow

# 主线：GD / GA / 神经代理
python train.py                  # 三种方式，约 4 分钟
python train.py --quick          # 冒烟验证，几秒
python train.py --gd-init both   # 额外跑「好初值」对照（实测：收敛，并非 NaN）
python visualize.py              # 原始三合一可视化

# 代理模型（θ→V）
python run_surrogate.py --n 400 --epochs 3000 --gd-steps 600 --gen 80 --pop 100 --whitebox-gd
python visualize_surrogate.py    # 保真度 / 误差归因 / GD / 外推 四联图
python make_matrix_fig.py        # 2×3 对照矩阵
python make_landscape_fig.py     # 势垒 + 断崖地形图

# 支撑实验（复核用，README 里的数字都出自这里）
python exp_support.py            # 计时重复 + 地形多切片，约 1 分钟
python exp_seeds.py --n 5        # 多随种子，约 12 分钟

# 报告
python make_report.py            # 重新生成分图 + 11 页 PPT
```

## 依赖

- Python 3.10+
- `numpy`、`scipy`（拉丁超立方、Spearman）、`matplotlib`、`torch`
- `python-pptx`、`pillow`（仅报告生成需要）

## 结果解读

- `results/summary.json`：原始三种方式的最终参数与目标值；`results/*.npz` 含轨迹与损失曲线。
- `results_surrogate/summary_surrogate.json`：代理实验全部指标（保真度、误差归因、外推、2×3 对照）。
- `results_surrogate/exp_support.json`：加速比重复测量、地形多切片。
- `results_surrogate/seeds.json`：5 个随机种子的 GD / GA 复核。
- 核心证据链：`T-World_GD_vs_GA_报告.pptx`、`代理模型实验报告.md`、`双解释_逐句有据.md`。
