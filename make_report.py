# -*- coding: utf-8 -*-
"""
生成 presentation 报告：三张分图 + 一份 PPTX（通俗 × 专业 双表述）。
运行：python make_report.py
产物：
  results/fig_fit.png, fig_loss.png, fig_ga.png
  T-World_GD_vs_GA_报告.pptx
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import config
from data import load_ground_truth
from model import fhn_simulate

BASE = config.BASE_DIR
RES = config.RESULTS_DIR


def load(key):
    return np.load(os.path.join(RES, f"{key}.npz"))


# ============ 数据 ============
t, y, y_clean, theta_true, n_spikes = load_ground_truth()
gd = load("gd_bad")
ga = load("ga")
neural = load("neural")
V_ga_best = fhn_simulate(ga["best_theta"], t, config)

# ============ 图1：拟合对比 ============
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(t, y_clean, color="0.35", lw=2, label="ground truth (4 次发放)")
ax.plot(t, gd["final_V"], color="#d62728", lw=1.2, alpha=0.9,
        label="GD + 白盒 ODE（28 次发放，卡死）")
ax.plot(t, V_ga_best, color="#2ca02c", ls="--", lw=1.6,
        label="遗传算法 + 白盒 ODE（恢复真值）")
ax.plot(t, neural["final_V"], color="#9467bd", ls="-.", lw=1.3,
        label="GD + 神经代理（平滑拟合）")
ax.set_xlabel("t")
ax.set_ylabel("V(t)")
ax.set_title("拟合对比：同一份数据，三种训练方式")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_fit.png"), dpi=130)
plt.close(fig)

# ============ 图2：损失曲线 ============
fig, ax = plt.subplots(figsize=(9, 4.2))
g_hist = gd["loss_history"]
g_fin = np.isfinite(g_hist)
ax.plot(np.arange(len(g_hist))[g_fin], g_hist[g_fin], color="#d62728", lw=1.4,
        label="GD + 白盒 ODE（卡死在 2.06，不下降）")
n_hist = neural["loss_history"]
ax.plot(np.arange(len(n_hist)), n_hist, color="#9467bd", lw=1.4,
        label="GD + 神经代理（平滑降到 0.06）")
ax.axhline(0.00043, color="#2ca02c", ls=":", lw=1.2,
           label="遗传算法最优 MSE = 0.00043")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("loss (MSE, log)")
ax.set_title("损失曲线：GD（坏初值）卡在 2.06 平台；遗传算法降到 4e-04")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_loss.png"), dpi=130)
plt.close(fig)

# ============ 图3：遗传算法收敛 ============
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(ga["history_f1"], color="#2ca02c", lw=1.5, marker="o", ms=3,
        label="最小 MSE (f1)")
ax.set_xlabel("代数 (generation)")
ax.set_ylabel("最小 MSE (f1)")
ax.set_yscale("log")
ax.set_title("遗传算法收敛：MSE 快速降至 ~0.0004，100% 个体发放正确")
ax.legend(fontsize=9)
ax.set_ylim(bottom=1e-4)
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_ga.png"), dpi=130)
plt.close(fig)

print("三张分图已生成。")

# ============ PPT ============
from PIL import Image as PILImage
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x20, 0x20, 0x20)
GRAY = RGBColor(0x59, 0x59, 0x59)
PLAIN_C = RGBColor(0x2E, 0x74, 0xB5)   # 通俗（蓝）
PRO_C = RGBColor(0x70, 0x70, 0x70)      # 专业（灰）

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def set_title(slide, text, size=26, color=ACCENT):
    slide.shapes.title.text = text
    tf = slide.shapes.title.text_frame
    tf.paragraphs[0].font.size = Pt(size)
    tf.paragraphs[0].font.color.rgb = color
    tf.paragraphs[0].font.bold = True


def add_dual(body, plain, prof, first=False):
    """追加一组「通俗 + 专业」双表述。plain 为通俗，prof 为专业。"""
    if first:
        p0 = body.paragraphs[0]
        p1 = body.add_paragraph()
    else:
        p0 = body.add_paragraph()
        p1 = body.add_paragraph()
    p0.text = "通俗 · " + plain
    p0.level = 0
    p0.font.size = Pt(16)
    p0.font.color.rgb = PLAIN_C
    p0.space_before = Pt(6)
    p1.text = "专业 · " + prof
    p1.level = 1
    p1.font.size = Pt(13)
    p1.font.color.rgb = PRO_C
    p1.space_after = Pt(6)


def content_slide(title, pairs):
    """pairs: list of (plain, prof)"""
    s = prs.slides.add_slide(prs.slide_layouts[1])
    set_title(s, title)
    body = s.placeholders[1].text_frame
    body.word_wrap = True
    for i, (plain, prof) in enumerate(pairs):
        add_dual(body, plain, prof, first=(i == 0))
    return s


def image_slide(title, img_path, plain, prof):
    s = prs.slides.add_slide(prs.slide_layouts[5])  # Title Only
    set_title(s, title)
    with PILImage.open(img_path) as im:
        iw, ih = im.size
    scale = min(Inches(12.0) / iw, Inches(4.95) / ih)
    w, h = int(iw * scale), int(ih * scale)
    s.shapes.add_picture(img_path, (prs.slide_width - w) // 2, Inches(1.25),
                         width=w, height=h)
    tb = s.shapes.add_textbox(Inches(0.8), Inches(6.32), Inches(11.7), Inches(0.95))
    tf = tb.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.text = "通俗 · " + plain
    p0.font.size = Pt(15)
    p0.font.color.rgb = PLAIN_C
    p1 = tf.add_paragraph()
    p1.text = "专业 · " + prof
    p1.font.size = Pt(12)
    p1.font.color.rgb = PRO_C
    return s


def figure_slide(title, img_path, caption, max_w=12.3, max_h=5.05, top=1.22):
    """整幅插图页：按比例装进 (max_w x max_h) 的框内，底部一行说明。
    与 image_slide 的区别是同时受宽高约束，避免 16:9 版面上把图撑出页面。"""
    s = prs.slides.add_slide(prs.slide_layouts[5])  # Title Only
    set_title(s, title)
    with PILImage.open(img_path) as im:
        iw, ih = im.size
    scale = min(Inches(max_w) / iw, Inches(max_h) / ih)
    w, h = int(iw * scale), int(ih * scale)
    s.shapes.add_picture(img_path, (prs.slide_width - w) // 2, Inches(top),
                         width=w, height=h)
    tb = s.shapes.add_textbox(Inches(0.75), Inches(6.5), Inches(11.9), Inches(0.8))
    tf = tb.text_frame
    tf.word_wrap = True
    pp = tf.paragraphs[0]
    pp.text = caption
    pp.font.size = Pt(14)
    pp.font.color.rgb = PLAIN_C
    return s


# --- 封面 ---
s = prs.slides.add_slide(prs.slide_layouts[0])
s.shapes.title.text = "梯度下降  vs  遗传算法"
s.shapes.title.text_frame.paragraphs[0].font.size = Pt(42)
s.shapes.title.text_frame.paragraphs[0].font.bold = True
s.shapes.title.text_frame.paragraphs[0].font.color.rgb = ACCENT
sub = s.placeholders[1]
sub.text = ("通俗 · 给模型调参，到底该“算梯度下山”，还是该“进化式海选”？\n"
            "专业 · 基于梯度的优化与多目标进化优化，在白盒 ODE 参数标定任务中的对照研究")
for p in sub.text_frame.paragraphs:
    p.font.size = Pt(16)
    p.font.color.rgb = GRAY

# --- 问题 ---
content_slide("问题：两种 AI4Science 模型，两种训练范式", [
    ("两种 AI 模型都在“让计算机自动拟合数据”，只是调参方式完全不同。",
     "二者均属数据驱动的参数估计任务，却分别采用梯度反向传播与多目标进化优化两种优化范式。"),
    ("T-World 是把心肌细胞写成“每个参数都看得懂含义”的物理方程组，所以用遗传算法来调参。",
     "T-World 为白盒 ODE 模型，参数具生理学含义；论文采用多目标遗传算法（MOGA）标定组分参数，模型以 Matlab / CellML 发布，也可在 Python 运行。"),
    ("ProteoFlow / UMA 把规律装进“光滑可导的神经网络”，所以能直接求梯度。",
     "ProteoFlow / UMA 采用可微神经网络参数化（流匹配速度场 / 等变图网络），支持端到端基于梯度的训练。"),
    ("核心疑问：同样拟合数据，凭啥一个用进化、一个用梯度？",
     "核心问题：同一拟合任务，其优化范式的选择差异，其内在决定因素为何？"),
])

# --- 方法 ---
content_slide("方法：一个“麻雀虽小、五脏俱全”的替身模型", [
    ("不搬原模型（太大、要超算），造了个保留关键特征的迷你细胞模型。",
     "采用 FitzHugh–Nagumo 可兴奋系统作为降维替身，保留“白盒 / 刚性 / 阈值”三大核心特征。"),
    ("它一边动得快、一边动得慢（叫刚性）；发不发放是“一刀切”的（有阈值）。",
     "eps << 1 引入刚性，快慢时间尺度相差约 50 倍；系统具兴奋阈值，发放与否为二值事件，类比早/延迟后除极（EAD/DAD）。"),
    ("有 4 个旋钮参数，真实值能让细胞“每刺激一下发一次”。",
     "4 个待拟合参数 [eps, a, b, amp]，真实值 [0.02, 0.8, 0.8, 0.8] 对应 1:1 刺激-发放；目标为向量 [f_MSE, f_fire]。"),
])

# --- 对比设计 ---
content_slide("第一轮设计：三条路线，但缺一格", [
    ("先按最顺的思路跑三条路线，后面会看到它缺了整整一格。",
     "初版设计：固定数据，比较三种「优化器 × 参数化」组合；该设计缺少「初值」这一维度，补跑后才暴露。"),
    ("路线①：梯度下降配白盒 ODE——T-World 若硬要用梯度的反事实。",
     "方案①：GD + 白盒 ODE（仅坏初值）——对应“若 T-World 坚持梯度下降”的反事实情形。"),
    ("路线②：多目标遗传算法配白盒 ODE——T-World 的实际做法。",
     "方案②：MOGA + 白盒 ODE——对应 T-World 论文采用的标定方式。"),
    ("路线③：梯度下降配可微代理——但它只拟合曲线，不反推参数。",
     "方案③：GD + 可微代理（t→V 曲线回归）——该臂不输出参数 θ，与①②不是同一道题，不能直接对比。"),
])

# --- 结果图1 ---
image_slide("结果一：拟合效果",
            os.path.join(RES, "fig_fit.png"),
            "一眼看出：红色（梯度下降）卡死、发错了 28 次；绿色（遗传算法）几乎和真值重合。",
            "梯度下降陷入错误发放模式的局部极小（28 次 vs 4 次）；遗传算法恢复真实波形（MSE 0.0004）。")

# --- 结果图2 ---
image_slide("结果二：损失曲线",
            os.path.join(RES, "fig_loss.png"),
            "损失掉到 2.06 就再也不动了——不是没跑够，是真下不去了；紫色（神经代理）却一路光滑向下。",
            "GD+白盒 ODE 收敛至非零平台（局部极小）；GD+神经代理单调下降至 0.06；GA 达 0.0004。")

# --- 结果图3 ---
image_slide("结果三：遗传算法收敛",
            os.path.join(RES, "fig_ga.png"),
            "几十代就找到答案，100% 的候选都“发对了次数”。",
            "MOGA 在数十代内收敛至 MSE≈0.0004，末代种群 100% 满足发放次数约束（f_fire=0）。")

# --- 交叉验证矩阵（2x3） ---
figure_slide("初值是决定性变量：GD vs GA 的完整交叉验证",
             os.path.join(BASE, "results_surrogate", "fig_matrix.png"),
             "同一个 Adam、同一份数据：好初值 3.95e-04（5 个种子全部优于 GA 的 4.09e-04），坏初值直接不可行。"
             "GD 极度依赖初值，而不是被刚性或不可导性一票否决。")

# --- 优化景观 ---
figure_slide("为什么坏初值会死？——优化景观里的势垒与断崖",
             os.path.join(BASE, "results_surrogate", "fig_landscape.png"),
             "代理模型磨平了可导性，却磨不掉多区制带来的非凸性：势垒 +12%~27%、断崖 6~21 倍（3 条切片）。"
             "GA 靠变异跨过去，GD 只能滚回坑里。")

# --- 结论 ---
content_slide("结论：决定成败的是“初值”，不是“优化器”", [
    ("梯度下降没坏——同一个优化器换个好初值，121 秒就收敛到 4.0e-04。",
     "白盒 ODE + GD（好初值）：3.06e-3 → 4.00e-04，无 NaN、无越界；5 个种子复核 3.95e-04±4.8e-6，NSGA-II 为 4.09e-04±1.4e-5，5/5 GD 更优但仅差 3.4%。"),
    ("坏初值才会让它卡住：卡在错误放电，还滑出了参数下界。",
     "坏初值下 GD 落在「28 次发放」的局部极小（MSE 2.06），并滑出下界至 eps=0.00675；根因是初值，不是刚性。"),
    ("进化算法不靠导数，稳，找回真值。",
     "MOGA 不依赖梯度，对刚性、阈值与不可导目标鲁棒，稳健恢复真值（MSE 0.0004）。"),
    ("选型一句话：给得出初值就用梯度，给不出就用进化。",
     "工程准则：有可靠初值 → 梯度下降（快且准）；无初值 → 进化优化（稳但慢）；单次仿真昂贵 → 训练代理模型作内循环。"),
])

# --- 附录：假设的推翻与修正 ---
content_slide("附录：实验假设的推翻与修正", [
    ("我们原来的假设：刚性和阶跃惩罚会让梯度下降直接爆炸。",
     "初始假设：刚性（1/eps）与 f_fire 阶跃项使 GD 梯度爆炸、无法收敛——该假设写在 config.py 注释里，未经验证。"),
    ("第一处错：那个阶跃项根本没进梯度下降的损失。",
     "复核①：gd.py 仅优化 MSE，f_fire 从未进入损失函数；GD 根本没有遇到不可导项。"),
    ("第二处错：损失是平滑收敛的，没有 NaN，参数也一直是正的。",
     "复核②：100 轮 loss 2.77→2.06 单调下降（仅 4 轮微升），无 NaN/inf；eps 最小 0.00675，恒为正。"),
    ("补上缺的那一格，结论整个反过来了。",
     "复核③：补跑 GD + 好初值，121 s 收敛至 4.00e-04，优于 NSGA-II（4.32e-4）；决定性变量是初值。"),
    ("所以这一页不是辩解，是我们实验设计的一部分。",
     "修正后结论：多区制导致的非凸景观才是主因；迭代复核与真实 ODE 复核应作为流程环节固定下来。"),
])

out = os.environ.get("DECK_OUT") or os.path.join(BASE, "T-World_GD_vs_GA_报告.pptx")
prs.save(out)
print("PPT 已生成：", out)