import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, rankdata


def plot_reg_ci(ax, x, y, color="#74add1", ci_alpha=0.15, line_alpha=0.85):
    X = sm.add_constant(x)
    m = sm.OLS(y, X).fit()
    xp = np.linspace(x.min(), x.max(), 100)
    yp = m.predict(sm.add_constant(xp))
    ci = m.get_prediction(sm.add_constant(xp)).conf_int(alpha=0.05)
    ax.plot(xp, yp, color=color, ls="--", alpha=line_alpha, zorder=3)
    ax.fill_between(xp, ci[:, 0], ci[:, 1], color=color, alpha=ci_alpha, lw=0, zorder=2)


def plot_scatter_rank_label_pred(csv_path):
    df = pd.read_csv(csv_path).reset_index(drop=True)
    x = rankdata(df["epistasis"].to_numpy(), method="average")
    y = rankdata(df["epistasisPred"].to_numpy(), method="average")
    _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, color="#74add1", alpha=0.9, s=30, zorder=4)
    plot_reg_ci(ax, x, y, color="#74add1")
    sr, _ = spearmanr(x, y)
    pr, _ = pearsonr(x, y)
    ax.text(0.05, 0.96, f"Spearman $\\rho$={sr:.2f}\nPearson $r$={pr:.2f}", transform=ax.transAxes, va="top", ha="left", fontsize=12)
    ax.set_xlabel("Experimental Rank", fontsize=14)
    ax.set_ylabel("Predicted Rank", fontsize=14)
    plt.tight_layout()
    plt.savefig("plot/scatter_epistasis_rank.svg", bbox_inches="tight")
    plt.close()


def plot_boxplot_blind_test_top10_mut234(df_box, wt_value):
    data = df_box["label+wt"] / wt_value
    np.random.seed(2)
    plt.figure(figsize=(3.3, 6))
    box = plt.boxplot([data], positions=[0], patch_artist=True, widths=0.225, showfliers=False, medianprops=dict(color="k", linewidth=1))
    color = "#74add1"
    box["boxes"][0].set_facecolor(color)
    box["boxes"][0].set_edgecolor("k")
    box["boxes"][0].set_alpha(0.4)
    y = data.values
    x = np.random.normal(loc=0, scale=0.08, size=len(y))
    plt.scatter(x, y, color=color, alpha=0.9, s=30, zorder=2)
    plt.axhline(1.0, ls="--", color="red", label="wt_value = 1", lw=1.5)
    plt.xlabel("All Mutations Combined", fontsize=14)
    plt.ylabel("fold change", fontsize=14)
    plt.xticks([0], ["All"])
    plt.legend(frameon=True, framealpha=0.5, loc="lower right")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("plot/boxplot_blind_test_top10.svg", bbox_inches="tight")
    plt.close()


plot_scatter_rank_label_pred("epistasis_pred.csv")
df_box = pd.read_csv("raw_test_top10_label+wt.csv")
wt_value = (df_box["label+wt"] - df_box["label"]).iloc[0]
plot_boxplot_blind_test_top10_mut234(df_box, wt_value)
