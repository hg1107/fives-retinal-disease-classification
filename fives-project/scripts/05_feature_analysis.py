"""05 - Feature analysis on the TRAIN split only (the test split is never read here).

Figures: results/figures/feat_*.png. Tables: results/tables/feat_*.csv.
The vessel sanity check is the one place ground-truth masks are read (to validate our image-derived
vessel features); the masks do not feed any feature or model.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.cluster.hierarchy import leaves_list, linkage  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.feature_selection import f_classif  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import features as F  # noqa: E402
from src.preprocess import load_dataset  # noqa: E402

CLASSES = config.CLASS_NAMES
PAL = dict(zip(CLASSES, ["#4C78A8", "#E45756", "#72B7B2", "#F2A93B"]))  # same class colours as the EDA
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({"savefig.dpi": config.FIG_DPI, "axes.titleweight": "bold"})
T = config.TABLES_DIR
DOMINANT_SHARE = 0.95  # near-constant: one value holds >= 95% of the training rows
HI_CORR = 0.95


def savefig(fig, name):
    fig.savefig(config.FIGURES_DIR / f"feat_{name}.png", dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


Xtr, ytr, _, _ = load_dataset("handcrafted")  # test split deliberately discarded
cls = pd.Series(ytr).map(config.IDX_TO_CLASS)
names = list(Xtr.columns)
group = pd.Series([c.split("_")[0] for c in names], index=names)
print(f"Train: {Xtr.shape[0]} images x {Xtr.shape[1]} handcrafted features")

# ---------------------------------------------------------------- 1. missing / constant / near-constant
miss = Xtr.isna().sum()
nonfinite = np.isinf(Xtr).sum()
top_share = Xtr.apply(lambda s: s.round(10).value_counts(normalize=True, dropna=False).iloc[0])
nuniq = Xtr.nunique()
drop = pd.DataFrame({"feature": names, "n_missing": miss.values, "n_inf": nonfinite.values, "n_unique": nuniq.values,
                     "top_value_share": top_share.values})
drop["reason"] = np.where(drop.n_unique <= 1, "constant", np.where(drop.top_value_share >= DOMINANT_SHARE, "near_constant", ""))
drop_list = drop[drop.reason != ""]
drop.sort_values("top_value_share", ascending=False).to_csv(T / "feat_missing_constant.csv", index=False)
drop_list.to_csv(T / "feat_drop_list.csv", index=False)
print(f"\nMissing values: {int(miss.sum())}; inf: {int(nonfinite.sum())}")
print(f"Constant / near-constant (top value >= {DOMINANT_SHARE:.0%}): {len(drop_list)}")
print(drop_list.to_string(index=False) if len(drop_list) else "(none)")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
t = drop.sort_values("top_value_share", ascending=False).head(15)
ax[0].barh(t.feature[::-1], t.top_value_share[::-1], color="#4C78A8")
ax[0].axvline(DOMINANT_SHARE, color="r", ls="--", label=f"near-constant cutoff {DOMINANT_SHARE:.0%}")
ax[0].set(title="Share of rows held by the most common value (top 15)", xlabel="share")
ax[0].legend(loc="lower right")
ax[1].hist(drop.n_unique / len(Xtr), bins=20, color="#72B7B2")
ax[1].set(title=f"Distinct values / rows (missing: {int(miss.sum())}, inf: {int(nonfinite.sum())})", xlabel="unique fraction", ylabel="features")
fig.tight_layout()
savefig(fig, "01_missing_constant")

# ---------------------------------------------------------------- 2. top-12 features by ANOVA F
Fs, ps = f_classif(Xtr.fillna(Xtr.median()), ytr)
rank = pd.DataFrame({"feature": names, "group": group.values, "F": Fs, "p": ps}).sort_values("F", ascending=False)
rank.to_csv(T / "feat_anova_ranking.csv", index=False)
print("\nTop 12 features by ANOVA F:\n", rank.head(12).to_string(index=False))
print("Mean F by group:", rank.groupby("group").F.mean().round(1).to_dict(), "| in top 30:", rank.head(30).group.value_counts().to_dict())
fig, axes = plt.subplots(3, 4, figsize=(16, 9))
for a, (_, r) in zip(axes.ravel(), rank.head(12).iterrows()):
    d = pd.DataFrame({"v": Xtr[r.feature].values, "cls": pd.Categorical(cls, CLASSES)})
    sns.kdeplot(d, x="v", hue="cls", hue_order=CLASSES, palette=PAL, fill=True, alpha=0.2, common_norm=False, legend=False, ax=a, warn_singular=False)
    a.set(title=f"{r.feature}\nF={r.F:.0f}", xlabel="", ylabel="")
    a.title.set_fontsize(9)
fig.legend([plt.Rectangle((0, 0), 1, 1, color=PAL[c], alpha=0.5) for c in CLASSES], CLASSES, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Top 12 handcrafted features by ANOVA F-score (train)", fontweight="bold")
fig.tight_layout()
savefig(fig, "02_top12_distributions")

# ---------------------------------------------------------------- 3. correlation (clustered) + pairs > 0.95
keep_cols = [c for c in names if c not in set(drop_list.feature)]
C = Xtr[keep_cols].fillna(Xtr[keep_cols].median()).corr()
order = leaves_list(linkage(1 - np.abs(C.values)[np.triu_indices(len(C), 1)], method="average"))
Co = C.iloc[order, order]
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(Co, cmap="vlag", vmin=-1, vmax=1, square=True, xticklabels=False, yticklabels=False, cbar_kws={"label": "Pearson r", "shrink": 0.6}, ax=ax)
ax.set_title("Clustered correlation of handcrafted features (train)")
# group colour strip along the left edge
gcol = dict(zip(sorted(group.unique()), sns.color_palette("Set2", group.nunique())))
for i, f in enumerate(Co.index):
    ax.add_patch(plt.Rectangle((-3, i), 2.5, 1, color=gcol[group[f]], clip_on=False))
ax.legend([plt.Rectangle((0, 0), 1, 1, color=v) for v in gcol.values()], list(gcol), title="group", loc="upper left", bbox_to_anchor=(1.12, 1.0))
savefig(fig, "03_correlation_clustered")
iu = np.triu_indices(len(C), 1)
pairs = pd.DataFrame({"feature_a": C.index[iu[0]], "feature_b": C.columns[iu[1]], "r": C.values[iu]})
pairs = pairs[pairs.r.abs() > HI_CORR].assign(abs_r=lambda d: d.r.abs()).sort_values("abs_r", ascending=False).drop(columns="abs_r")
pairs.to_csv(T / "feat_high_corr_pairs.csv", index=False)
from src.preprocess import CorrelationFilter  # noqa: E402
kept_after_filter = int(CorrelationFilter(HI_CORR).fit(Xtr[keep_cols].fillna(0)).support_.sum())
print(f"\nPairs with |r| > {HI_CORR}: {len(pairs)} (features left after the greedy correlation filter: {kept_after_filter}/{len(keep_cols)})")
print(pairs.head(15).to_string(index=False))

# ---------------------------------------------------------------- 4. outliers (IQR) and skew
q1, q3 = Xtr.quantile(0.25), Xtr.quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out_frac = ((Xtr < lo) | (Xtr > hi)).mean()
out_frac[iqr == 0] = np.nan  # IQR rule undefined for zero-IQR features
sk = Xtr.apply(stats.skew)
outl = pd.DataFrame({"feature": names, "group": group.values, "iqr": iqr.values, "outlier_frac": out_frac.values,
                     "n_outliers": ((Xtr < lo) | (Xtr > hi)).sum().values, "skew": sk.values, "abs_skew_gt1": (sk.abs() > 1).values})
outl.sort_values("outlier_frac", ascending=False).to_csv(T / "feat_outlier_summary.csv", index=False)
print(f"\nOutliers (1.5 IQR): median per-feature fraction {out_frac.median():.1%}, max {out_frac.max():.1%}; "
      f"{(out_frac > 0.05).sum()} features have >5% outliers; zero-IQR features: {int((iqr == 0).sum())}")
print(f"Skew: median |skew| {sk.abs().median():.2f}; {(sk.abs() > 1).sum()}/{len(sk)} features have |skew| > 1; {(sk.abs() > 2).sum()} have |skew| > 2")
print("Outlier fraction by group (median):", outl.groupby("group").outlier_frac.median().round(3).to_dict())
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
t = outl.dropna(subset=["outlier_frac"]).sort_values("outlier_frac", ascending=False).head(20)
ax[0].barh(t.feature[::-1], t.outlier_frac[::-1], color="#E45756")
ax[0].set(title="Top 20 features by IQR-outlier fraction", xlabel="fraction of train rows")
sns.histplot(outl.outlier_frac.dropna(), bins=25, ax=ax[1], color="#4C78A8")
ax[1].set(title="Outlier fraction across features", xlabel="fraction", ylabel="features")
sns.histplot(sk, bins=30, ax=ax[2], color="#72B7B2")
for v in (-1, 1):
    ax[2].axvline(v, color="k", ls="--", lw=1)
ax[2].set(title=f"Skewness ({(sk.abs() > 1).sum()} of {len(sk)} features |skew|>1)", xlabel="skew", ylabel="features")
fig.tight_layout()
savefig(fig, "04_outliers_skew")

# ---------------------------------------------------------------- 5. PCA and t-SNE (standardised train features)
Z = StandardScaler().fit_transform(Xtr[keep_cols].fillna(Xtr[keep_cols].median()))
Z = np.clip(Z, -5, 5)  # keep extreme outliers from dominating the 2-D embeddings (visualisation only)
pca = PCA(2, random_state=config.SEED).fit(Z)
P = pca.transform(Z)
TS = TSNE(2, perplexity=30, init="pca", random_state=config.SEED).fit_transform(Z)
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for a, E, ttl, lab in [(ax[0], P, f"PCA (explained variance {pca.explained_variance_ratio_.sum():.0%})", "PC"), (ax[1], TS, "t-SNE (perplexity 30)", "t-SNE ")]:
    sns.scatterplot(x=E[:, 0], y=E[:, 1], hue=pd.Categorical(cls, CLASSES), palette=PAL, s=22, alpha=0.8, linewidth=0, ax=a)
    a.set(title=ttl, xlabel=f"{lab}1", ylabel=f"{lab}2")
ax[0].legend_.remove()
sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1), title="")
fig.tight_layout()
savefig(fig, "05_pca_tsne")
print(f"\nPCA: first 5 PCs explain {PCA(5, random_state=config.SEED).fit(Z).explained_variance_ratio_.round(3).tolist()}")

# ---------------------------------------------------------------- 6. vessel sanity check vs ground-truth masks
m = pd.read_csv(config.DATA_DIR / "manifest.csv")
mt = m[m.official_split == "train"].reset_index(drop=True)
assert (mt.image_id.values == pd.read_csv(config.DATA_DIR / "features_handcrafted.csv").query("split=='train'").image_id.values).all()


def gt_density(img_rel, mask_rel):
    p = F.preprocess(config.DATASET_PATH / img_rel, config.DATASET_PATH / mask_rel)
    inner = F.vessel_region(p["mask"])  # same region our own vessel_density is measured in
    return float(((p["aux"] > 127) & inner).sum() / inner.sum())


gt = np.array(Parallel(n_jobs=8)(delayed(gt_density)(a, b) for a, b in zip(mt.image_path, mt.mask_path)))
vcols = ["vessel_density", "vessel_skeleton_density", "vessel_frangi_mean"]
rows = []
fig, ax = plt.subplots(1, 3, figsize=(15, 4.7))
for a, v in zip(ax, vcols):
    x = Xtr[v].values
    rp, rs = stats.pearsonr(gt, x)[0], stats.spearmanr(gt, x)[0]
    rows.append({"feature": v, "pearson_r": rp, "spearman_rho": rs})
    sns.scatterplot(x=gt, y=x, hue=pd.Categorical(cls, CLASSES), palette=PAL, s=18, alpha=0.75, linewidth=0, ax=a, legend=v == vcols[-1])
    sl, ic = np.polyfit(gt, x, 1)
    a.plot(np.sort(gt), sl * np.sort(gt) + ic, "k--", lw=1)
    a.set(title=f"{v}\nPearson r={rp:.2f}, Spearman={rs:.2f}", xlabel="ground-truth mask vessel density", ylabel="image-derived")
sns.move_legend(ax[-1], "upper left", bbox_to_anchor=(1, 1), title="")
fig.tight_layout()
savefig(fig, "06_vessel_vs_groundtruth")
vs = pd.DataFrame(rows)
by = pd.DataFrame({"gt": gt, "ours": Xtr.vessel_density.values, "cls": cls}).groupby("cls").mean().reindex(CLASSES)
vs.to_csv(T / "feat_vessel_sanity.csv", index=False)
print("\nVessel sanity check vs ground-truth density (train):\n", vs.round(3).to_string(index=False))
print("Class means (GT vs image-derived vessel_density):\n", by.round(4).to_string())
print(f"Scale: GT mean {gt.mean():.3f} vs ours {Xtr.vessel_density.mean():.3f}")
