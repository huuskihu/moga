# moga — 梯度下降 vs 遗传算法：白盒 ODE 参数标定的极简复现

一个极简、可在个人电脑上复现的实验，回答 AI4Science 里的一个核心问题：

> 同样是「让模型拟合数据」，为什么 **T-World**（虚拟心肌细胞）用**多目标遗传算法**，
> 而 **ProteoFlow / UMA** 用**梯度下降**？

## 结论（一句话）

差异在「**参数化**」而不在「**优化器**」——刚性的白盒 ODE 让梯度下降失效（卡死 / 发散），
多目标遗传算法不依赖导数所以稳健；换成可微的神经参数化，梯度下降又立刻可用。

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
| 梯度下降 (Adam) + 白盒 ODE | ❌ 卡死 / 越界 | 2.06 | 28（错误，应为 4） |
| 多目标遗传算法 (NSGA-II) + 白盒 ODE | ✅ 恢复真值 | 0.0004 | 4 |
| 梯度下降 (Adam) + 可微神经代理 | ✅ 平滑收敛 | 0.06 | ~4 |

## 文件结构

| 文件 | 说明 |
| --- | --- |
| `config.py` | 全局配置：模型、真实参数、边界、求解器超参数 |
| `model.py` | FHN 模型（numpy / 批量 / torch 三套前向） |
| `data.py` | 合成 ground truth 的生成与校验 |
| `objective.py` | 多目标损失（向量 `[f_MSE, f_fire]` + 硬约束） |
| `gd.py` | 梯度下降求解器（PyTorch Adam） |
| `ga.py` | 手写 NSGA-II（不依赖 deap/scipy） |
| `neural.py` | 可微神经代理（ProteoFlow / UMA 范式） |
| `train.py` | 训练入口 / 启动器（代码与训练分离） |
| `visualize.py` | 结果可视化 |
| `make_report.py` | 一键生成三张分图 + PPTX 报告 |
| `results/` | 本次实验输出（npz + 图） |
| `T-World_GD_vs_GA_报告.pptx` | 8 页报告（通俗 × 专业双表述） |
| `讲稿.md` / `讲稿.docx` | 配套讲稿 |

## 快速开始

```bash
pip install numpy matplotlib torch python-pptx   # python-pptx 仅生成报告需要

python train.py                 # 完整训练（GD + GA + 神经代理，约 4 分钟）
python train.py --quick         # 冒烟验证（几秒）
python train.py --gd-init both  # 额外跑「好初值→NaN」对照
python visualize.py             # 出三合一可视化图
python make_report.py           # 重新生成报告 PPT 与分图
```

## 依赖

- Python 3.10+
- `numpy`、`matplotlib`、`torch`
- `python-pptx`（仅报告生成）

## 结果解读

`results/summary.json` 记录了各方法最终参数与目标值；`results/*.npz` 含完整轨迹与损失曲线。
核心证据链详见 `T-World_GD_vs_GA_报告.pptx` 与 `讲稿.md`。