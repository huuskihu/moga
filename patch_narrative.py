# -*- coding: utf-8 -*-
"""一次性修正过时表述（config.py / README.md / 讲稿.md）。

所有替换都做命中校验：命中数不为 1 就报错退出，避免静默改坏文件。
统一按「行前缀」定位，避免中英文引号差异导致匹配失败。
"""
import io
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ok = []


def read(p):
    with io.open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def find_one(lines, prefix, tag):
    idx = [k for k, l in enumerate(lines) if l.startswith(prefix)]
    if len(idx) != 1:
        print(f"[FAIL] {tag}: 前缀 {prefix!r} 命中 {len(idx)} 行（期望 1）")
        sys.exit(1)
    return idx[0]


def rep_line(lines, prefix, new_text, tag, span=1):
    i = find_one(lines, prefix, tag)
    lines[i:i + span] = new_text.split("\n")
    ok.append(tag)


def rep_block(lines, start, end, new_text, tag):
    i = find_one(lines, start, tag + "·start")
    j = next((k for k, l in enumerate(lines) if k > i and l.startswith(end)), None)
    if j is None:
        print(f"[FAIL] {tag}: 找不到结束标记 {end!r}")
        sys.exit(1)
    lines[i:j] = new_text.split("\n")
    ok.append(tag)


def insert_before(lines, marker, new_text, tag):
    i = find_one(lines, marker, tag)
    lines[i:i] = new_text.split("\n")
    ok.append(tag)


# =====================================================================
# 1. config.py
# =====================================================================
p = os.path.join(BASE, "config.py")
lines = read(p).split("\n")
i = [k for k, l in enumerate(lines) if "梯度下降「好」初值" in l]
if len(i) != 1:
    print(f"[FAIL] config: 命中 {len(i)}")
    sys.exit(1)
lines[i[0]:i[0] + 1] = [
    "# 梯度下降「好」初值：贴近真实值。",
    "# 【2026-10-31 复核修正】原注释写「即便有好初值，刚性导致梯度爆炸/发散」——实测不成立：",
    "#   好初值下 GD 在白盒 ODE 上平滑收敛（3.06e-3 → 4.00e-4，100 轮 121 s），无 NaN、无越界，",
    "#   精度还优于 NSGA-II（4.32e-4），只是慢 5 倍。",
    "#   正确表述：GD 在白盒 ODE 上对「初值」极度敏感，而不是被刚性击穿。",
    "#   证据：python run_surrogate.py --whitebox-gd → results_surrogate/summary_surrogate.json",
]
ok.append("config.py: GD_INIT_GOOD 注释")
write(p, "\n".join(lines))

# =====================================================================
# 2. README.md
# =====================================================================
p = os.path.join(BASE, "README.md")
lines = read(p).split("\n")

rep_block(
    lines, "## 结论（一句话）", "## 模型",
    "## 结论（一句话）\n"
    "\n"
    "决定成败的是「**初值**」，不是「优化器」，也不是「**刚性**」。\n"
    "\n"
    "白盒 ODE 上的梯度下降对初值极度敏感：**好初值下平滑收敛（MSE 4.00e-04，100 轮 121 s，无 NaN、无越界）**，\n"
    "坏初值下则掉进「错误发放次数」的局部极小（28 次）并滑出参数盒（eps 0.04 → 0.00675）。\n"
    "多目标遗传算法不依赖初值，所以稳健（MSE 4.32e-04，24 s）。\n"
    "把「参数→输出」做成可微代理后，梯度下降 0.66 s 收敛到 1.53e-03——快 **184×**，精度受代理保真度限制。",
    "README: 结论段",
)

# 结果表：删掉旧三行，在第一行位置插入新五行
tbl_prefixes = ["| 梯度下降 (Adam) + 白盒 ODE |",
                "| 多目标遗传算法 (NSGA-II) + 白盒 ODE |",
                "| 梯度下降 (Adam) + 可微神经代理 |"]
rows = [k for k, l in enumerate(lines) if any(l.startswith(t) for t in tbl_prefixes)]
if len(rows) != 3:
    print(f"[FAIL] README 表格: 命中 {len(rows)} 行（期望 3）")
    sys.exit(1)
new_rows = "\n".join([
    "| 梯度下降 + 白盒 ODE · **坏初值** | ❌ 错误发放 + 越界 | 2.06 | 28（应为 4） |",
    "| 梯度下降 + 白盒 ODE · **好初值** | ✅ 平滑收敛 | **0.00040** | 4 |",
    "| 多目标遗传算法 (NSGA-II) + 白盒 ODE | ✅ 恢复真值 | 0.00043 | 4 |",
    "| 梯度下降 + 可微**代理模型**（θ→V，`run_surrogate.py`） | ✅ 0.66 s | 0.00153 | 4 |",
    "| 梯度下降 + `neural.py`（t→V，仅拟合曲线） | ⚠️ 收敛但**不反推参数** | 0.061 | — |",
    "",
    "> 前四行的 MSE 均由「参数丢回真实 ODE」复核；最后一行是曲线回归，输出 33,537 个网络权重而",
    "> 非 4 个参数，**不能作为「GD 可行」的对照证据**。详见 `代理模型实验报告.md`。",
]).split("\n")
first = rows[0]
for k in reversed(rows):
    del lines[k]
lines[first:first] = new_rows
ok.append("README: 结果表")

rep_line(lines, "python train.py --gd-init both",
         "python train.py --gd-init both  # 额外跑「好初值」对照（实测：收敛，并非 NaN）",
         "README: --gd-init 注释")
write(p, "\n".join(lines))

# =====================================================================
# 3. 讲稿.md
# =====================================================================
p = os.path.join(BASE, "讲稿.md")
lines = read(p).split("\n")

# 第5页：「发散」用词
for k, l in enumerate(lines):
    if "它卡住了，最终发散了 28 次" in l:
        lines[k] = l.replace("它卡住了，最终发散了 28 次",
                             "它卡住了，落在每周期自发多发放的病态模式上（28 次）")
        ok.append("讲稿: 第5页 发散→卡住")
        break
else:
    print("[FAIL] 讲稿: 第5页 未命中")
    sys.exit(1)

# 第6页：整段重写
rep_line(lines, "这张图更有说服力。",
         "这张图更有说服力。红色是「梯度下降 + 白盒 ODE」的坏初值：损失 2.77 降到 2.06 后形成平台，"
         "第 29 轮就平了、之后 70 轮不动——这是掉进局部极小，不是没跑够。绿色虚线是遗传算法的 0.0004。"
         "紫色那条要单独说明：它学的是 t→V 的曲线回归，不反推那四个参数，所以不能直接当作「GD 可行」的证据。"
         "真正公平的对照是第四次补跑——同一个优化器换好初值，121 秒收敛到 4.0e-04，比遗传算法还准。"
         "所以结论不是「GD 不行」，而是「GD 离不开好初值」。",
         "讲稿: 第6页 整段")

# D2 段
rep_block(
    lines, "**D2 · ", "**D3 ·",
    "**D2 · eps 方向的梯度尺度最大，但它没有「爆炸」**：\n"
    "\n"
    "损失对 eps 的依赖来自 `dV/dt = (…) / eps`，理论上 `∂V/∂eps` 是 `1/eps²` 量级。\n"
    "实测数值梯度（在 GD 收敛点）：eps 分量 **15.7**，其余三个参数只有 0.1–1.1——**eps 方向确实最大**。\n"
    "\n"
    "但它没有爆炸：`gd_bad` 的 100 轮里只有 **4 轮** loss 上升，第 **29** 轮就进入末值 ±0.1%，\n"
    "之后 70 轮是平线，全程无 NaN、无 inf。Adam 把这个大梯度按分量归一化吸收掉了。\n"
    "\n"
    "真正的后果不是崩溃，而是**方向**：GD 顺着 eps 的大梯度一路下调（0.04 → 0.00675），\n"
    "因为减小 eps 能让细胞更易自发发放、局部 MSE 更低（2.77 → 2.06）——代价是发放次数从 24 变成 28，\n"
    "最后**滑出参数下界 0.01**。所以这一条应改写成：\n"
    "**eps 方向的梯度最大，它没有炸掉 GD，而是把 GD 领进了错误区制并带出了边界。**",
    "讲稿: D2 段")

# B 节 稳定性说明（两行）
rep_line(lines, "- 这也就是 `config` 里写",
         "- 这也就是 `config` 里写「**eps 下界 > DT/2**」的原因：若 `eps` 被优化到太小，`dt/eps` 会逼近 2，欧拉失稳。\n"
         "  注意 GD 实际漂到的 0.00675 对应 `dt/eps = 0.74`，**仍在稳定域内**，所以那一次并没有数值爆炸——它只是越了界。",
         "讲稿: B 节 稳定性说明", span=2)

# 新增 D4：插在 E 节之前
insert_before(
    lines, "## E. 遗传算法为什么稳健",
    "**D4 · 补充复核（2026-10-31）：好初值下 GD 其实是成功的**\n"
    "\n"
    "原稿此处只测了坏初值，并把「好初值也会爆炸」写进了 `config.py` 注释，但那一臂从未执行。补跑结果：\n"
    "\n"
    "| 初值 | 耗时 | 真实 MSE | 发放次数 | 结局 |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| 坏 `[0.05, 0.3, 0.4, 0.8]` | 146 s | 2.06 | 28 | 局部极小 + 越界 |\n"
    "| 好 `[0.02, 0.75, 0.75, 0.75]` | 121 s | **4.00e-04** | 4 | 平滑收敛，优于 NSGA-II |\n"
    "\n"
    "结论：**GD 在白盒 ODE 上对初值极度敏感**，而不是「被刚性击穿」。\n"
    "T-World 这类模型有上百个参数、拿不到好初值，所以 GA 才是现实选择——原因是缺初值，不是梯度不存在。\n"
    "",
    "讲稿: 新增 D4")

# 第8页 结论行
rep_line(lines, "结论一句话：",
         "结论一句话：决定成败的是**初值**，不是优化器，也不是刚性。白盒 ODE 上的梯度下降，"
         "好初值 121 秒收敛到 4.0e-04（比遗传算法还准），坏初值则掉进错误发放的局部极小（28 次）"
         "并滑出参数下界（0.0067）。遗传算法不依赖初值，所以稳健，代价是慢。"
         "T-World 用 MOGA，是因为上百个参数根本给不出好初值——不是梯度不存在。",
         "讲稿: 结论行")

# H 因果链（代码块内容）
rep_block(
    lines, "白盒 ODE  = 显式方程", "所以：",
    "白盒 ODE  = 显式方程 + 刚性(1/eps) + 阈值(二值)\n"
    "   │ 阈值 → 发放次数是离散量，MSE 曲面在分岔处近乎断崖\n"
    "   ▼\n"
    "梯度下降（坏初值）→ 落在「28 次发放」的局部极小，并滑出参数盒\n"
    "梯度下降（好初值）→ 121 s 平滑收敛，MSE 4.0e-04（比 GA 更准）\n"
    "   │ 关键变量是「初值」，不是刚性\n"
    "   ▼\n"
    "遗传算法  → 不依赖初值，稳健恢复真值（0.0004, 4 spikes, 24 s）\n"
    "   │ 再把「参数→输出」做成可微代理\n"
    "   ▼\n"
    "梯度下降 + 代理 → 0.66 s 收敛到 1.5e-03（快 184×，精度受代理保真度限制）\n"
    "```\n",
    "讲稿: H 因果链")
write(p, "\n".join(lines))

print("全部替换成功：")
for t in ok:
    print("  ✓", t)
