# -*- coding: utf-8 -*-
"""修复报告第十节地形小节的 Markdown 结构，并把论文原文依据写进「事实性提醒」。"""
import io
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(BASE, "代理模型实验报告.md")
lines = io.open(p, encoding="utf-8").read().split("\n")

i = next(k for k, l in enumerate(lines) if l.startswith("**真正的地形结构**"))
j = next(k for k, l in enumerate(lines) if l.startswith("### 10.1"))
if i is None or j is None or j <= i:
    print("[FAIL] 找不到边界")
    sys.exit(1)

new = """**真正的地形结构**（`fig_landscape.png`，沿「GD 解 → 真值」切片）：

- GD 收敛解：MSE 2.66，24 次发放
- 小盆地底：MSE 1.74
- 势垒顶：MSE 2.22（比盆地底高 **27%**）
- 断崖：0.003 的参数区间内 MSE 从 2.22 掉到 0.10（**21 倍**）
- 崖底之后一路降到 4.0e-4

> 换两个爆发型起点重切（`exp_support.py`）：势垒 **+12% / +13%**，断崖 **6× / 16×**。
> 数值随切片变，但「盆地 → 抬价 → 跳崖」的结构三次都出现——这是结构性的，不是某一条切片的巧合。

所以准确的说法是：**MSE 自己也有断崖**（不止 f_fire），GD 卡住的原因是**要翻过一道 12%–27% 的势垒**才能到达崖口，
靠局部梯度翻不过去；GA 靠变异一步跨过。这与「是否可导」无关——代理模型把地形磨平之后，势垒还在。

**两处事实性提醒**：

1. `eps = 0.02` 是 FHN 替身的参数，**不是 T-World 的**，答辩时不要混用。
2. 不要说「T-World 放着 PyTorch 不用、非要用 MATLAB」。已核对论文原文（31 页中英对照 PDF，`pypdf` 全文检索）：
   - 「多目标遗传算法被用于拟合单个组分参数，也用于整合组分」——原文如此；
   - 「T-World is freely available via Matlab and CellML … 也可在 Python 中运行」——原文也支持 Python，所以「不用 PyTorch」的说法不成立；
   - `gamultiobj` **全文 0 命中**，`MATLAB`（大写）0 命中，只有 3 处 `Matlab` 且均指发布形式。不要写成「MATLAB gamultiobj」。

""".split("\n")

lines[i:j] = new
io.open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print("地形小节已重排，事实性提醒已更新")
