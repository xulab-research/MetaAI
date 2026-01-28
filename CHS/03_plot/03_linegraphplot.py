import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

CSV_PATH = Path("epistasis_pred.csv")
ROW_INDEXES = [5, 24]
MARKERS = ["D", "^"]
COLORS = ["#d62728", "#2ca02c"]
baseline_value = 0.0

sns.set_theme(style="white", context="paper", font_scale=1.35)


def setup_axes(ax, axr):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    axr.spines["top"].set_visible(False)
    axr.spines["left"].set_visible(False)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["WT", "Mut.A", "Mut.B", "Mut.A+B"], fontsize=9)
    ax.set_ylabel("Experimental fitness", fontsize=10)
    axr.set_ylabel("Predictive fitness", fontsize=10, rotation=-90, va="center", rotation_mode="anchor", labelpad=6)
    exp = 3
    scale = 10**exp
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: f"{y/scale:.1f}"))
    ax.yaxis.offsetText.set_visible(False)
    ax.text(-0.02, 1.02, rf"$\times 10^{{{exp}}}$", transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
    ax.set_xlim(-0.03, 3.03)
    ax.set_box_aspect(1)


df = pd.read_csv(CSV_PATH)
for row_idx, marker, color in zip(ROW_INDEXES, MARKERS, COLORS):
    r = df.iloc[row_idx]
    yl = [baseline_value, float(r["MutationALabel"]), float(r["MutationBLabel"]), float(r["CombinedLabel"])]
    yp = [baseline_value, float(r["MutationAPred"]), float(r["MutationBPred"]), float(r["CombinedMutationPred"])]
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=1000)
    axr = ax.twinx()
    setup_axes(ax, axr)
    sns.lineplot(x=[0, 1, 3], y=[yl[0], yl[1], yl[3]], ax=ax, color=color, lw=1.35, marker=None)
    sns.lineplot(x=[0, 2, 3], y=[yl[0], yl[2], yl[3]], ax=ax, color=color, lw=1.35, marker=None)
    sns.scatterplot(x=[0, 1, 2, 3], y=yl, ax=ax, color=color, s=28, marker=marker, edgecolor="none")
    sns.lineplot(x=[0, 1, 3], y=[yp[0], yp[1], yp[3]], ax=axr, color=color, lw=1.35, linestyle="--", marker=None)
    sns.lineplot(x=[0, 2, 3], y=[yp[0], yp[2], yp[3]], ax=axr, color=color, lw=1.35, linestyle="--", marker=None)
    sns.scatterplot(x=[0, 1, 2, 3], y=yp, ax=axr, color=color, s=28, marker=marker, edgecolor="none")
    ax.set_ylim(0, max(yl) * 1.04)
    axr.set_ylim(0, max(yp) * 1.04)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.18)
    fig.savefig(Path(f"plot/linegraph_epistasis_{row_idx}.svg"), format="svg", bbox_inches="tight")
    plt.show()
