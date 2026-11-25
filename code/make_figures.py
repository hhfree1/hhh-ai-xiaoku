"""从 data/metrics.json 生成四张图，输出到 figures/。

所有图形都由实测数据绘制，不含任何示意性内容。
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

C0 = "#185FA5"
C1 = "#BA7517"
GRID = "#D3D1C7"
TEXT = "#2C2C2A"
MUTED = "#5F5E5A"

candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
installed = {f.name for f in font_manager.fontManager.ttflist}
chosen = [c for c in candidates if c in installed]
plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160
plt.rcParams["axes.edgecolor"] = GRID
plt.rcParams["axes.labelcolor"] = MUTED
plt.rcParams["text.color"] = TEXT

with open("data/metrics.json", encoding="utf-8") as f:
    m = json.load(f)

samples = m["samples"]
labels = m["labels"]
history = m["history"]
names = ["".join(str(v) for v in s) for s in samples]
before = history[0]
after = history[-1]

os.makedirs("figures", exist_ok=True)


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


epochs = [h["epoch"] for h in history]
losses = [h["loss"] for h in history]

fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.plot(epochs, losses, color=C0, linewidth=1.8)
ax.scatter([epochs[0]], [losses[0]], color=C1, zorder=3, s=26)
ax.scatter([epochs[-1]], [losses[-1]], color=C0, zorder=3, s=26)
ax.annotate(
    f"第 0 轮  {losses[0]:.3f}",
    (epochs[0], losses[0]),
    textcoords="offset points",
    xytext=(8, 4),
    fontsize=9,
    color=MUTED,
)
ax.annotate(
    f"第 {epochs[-1]} 轮  {losses[-1]:.4f}",
    (epochs[-1], losses[-1]),
    textcoords="offset points",
    xytext=(-14, 12),
    fontsize=9,
    color=MUTED,
    ha="right",
)
ax.set_xlabel("训练轮数 epoch")
ax.set_ylabel("损失 loss")
ax.set_title("损失从 %.2f 降到 %.4f" % (losses[0], losses[-1]), fontsize=11, color=TEXT)
ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
finish(fig, "figures/loss_curve.png")

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
for ax, snap, title in zip(
    axes, [before, after], [f"训练前（第 0 轮）", f"训练后（第 {after['epoch']} 轮）"]
):
    pos = range(4)
    h1 = [row[0] for row in snap["hidden"]]
    h2 = [row[1] for row in snap["hidden"]]
    ax.bar([p - 0.19 for p in pos], h1, width=0.36, color=C0, label="h1")
    ax.bar([p + 0.19 for p in pos], h2, width=0.36, color=C1, label="h2")
    ax.set_xticks(list(pos))
    ax.set_xticklabels(names)
    ax.set_title(title, fontsize=10, color=MUTED)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("隐藏单元输出")
axes[1].legend(frameon=False, fontsize=9)
finish(fig, "figures/hidden_activations.png")

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
for ax, snap, title in zip(
    axes, [before, after], ["训练前：两点几乎重合", "训练后：两点完全重合"]
):
    groups = {}
    for (h1, h2), name, y in zip(snap["hidden"], names, labels):
        key = (round(h1, 2), round(h2, 2))
        groups.setdefault(key, {"names": [], "y": y, "xy": (h1, h2)})
        groups[key]["names"].append(name)
    for key, g in groups.items():
        h1, h2 = g["xy"]
        ax.scatter(h1, h2, s=70, color=C0 if g["y"] == 0 else C1, zorder=3)
        label = " / ".join(g["names"])
        dy = 10 if len(g["names"]) > 1 else 6
        ax.annotate(
            f"{label} (y={g['y']})",
            (h1, h2),
            textcoords="offset points",
            xytext=(9, dy),
            fontsize=8.5,
            color=C0 if g["y"] == 0 else C1,
        )
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel("h1")
    ax.set_ylabel("h2")
    ax.set_title(title, fontsize=10, color=MUTED)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
finish(fig, "figures/hidden_space.png")

fig, ax = plt.subplots(figsize=(6.2, 3.4))
colors = [C0 if y == 0 else C1 for y in labels]
bars = ax.bar(names, after["prob"], color=colors, width=0.55)
ax.axhline(0.5, color="#A32D2D", linewidth=1.2, linestyle="--")
ax.text(3.45, 0.515, "阈值 0.5", color="#A32D2D", fontsize=8.5, ha="right")
for bar, p, pred, y in zip(bars, after["prob"], after["pred"], labels):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        p + 0.03,
        f"{p:.3f}\n预测 {pred}" + (" 正确" if pred == y else " 错误"),
        ha="center",
        fontsize=8.5,
        color=TEXT,
    )
ax.set_ylim(0, 1.18)
ax.set_ylabel("输出概率")
ax.set_xlabel("输入样本")
ax.set_title("四个样本的最终概率与判定", fontsize=11, color=TEXT)
ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
finish(fig, "figures/probability_bars.png")
