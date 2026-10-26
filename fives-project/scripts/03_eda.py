"""03 - Exploratory data analysis on data/processed/manifest.csv.

Writes results/figures/eda_*.png, results/tables/eda_summary.csv and results/eda_insights.md.
Choices: the field of view (FOV) is estimated as grayscale > FOV_THRESH so colour histograms
and vessel density ignore the black border; mean images / sample grids use 256 px thumbnails.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FOV_THRESH = 10
CORNER = 120  # px; top-left corner lies outside the FOV circle, any bright pixels there are an overlay glyph/text
CORNER_BRIGHT = 40
THUMB = 256
CLASSES = config.CLASS_NAMES
PAL = dict(zip(CLASSES, ["#4C78A8", "#E45756", "#72B7B2", "#F2A93B"]))  # one colour per class everywhere
QUAL = ["blur_score", "brightness", "contrast"]
QLAB = {"blur_score": "Blur (Laplacian variance)", "brightness": "Brightness (mean gray)", "contrast": "Contrast (std gray)",
        "fov_fraction": "FOV fraction"}
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": config.FIG_DPI, "axes.titleweight": "bold"})


def savefig(fig, name):
    fig.savefig(config.FIGURES_DIR / f"eda_{name}.png", dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


m = pd.read_csv(config.DATA_DIR / "manifest.csv")
m["label_name"] = pd.Categorical(m["label_name"], CLASSES, ordered=True)
D = config.DATASET_PATH


# ---------------------------------------------------------------- pixel pass (parallel)
def pixel_features(img_rel, mask_rel):
    rgb = cv2.cvtColor(cv2.imread(str(D / img_rel), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fov = gray > FOV_THRESH
    hist = np.stack([np.bincount(rgb[..., c][fov], minlength=256) for c in range(3)]).astype(np.float64)
    msk = cv2.imread(str(D / mask_rel), cv2.IMREAD_GRAYSCALE)
    vessel = msk > 127
    return {
        "hist": hist / hist.sum(1, keepdims=True),
        "thumb": cv2.resize(rgb, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32),
        "fov_fraction": float(fov.mean()),
        "corner_glyph": float((gray[:CORNER, :CORNER] > CORNER_BRIGHT).mean() > 0.001),
        "vessel_ratio_image": float(vessel.mean()),
        "vessel_ratio_fov": float(vessel[fov].mean()),
        "mask_values": len(np.unique(msk)),
        **{f"mean_{k}": float(rgb[..., i][fov].mean()) for i, k in enumerate("rgb")},
    }


print("Reading images and masks...")
feats = Parallel(n_jobs=-1)(delayed(pixel_features)(a, b) for a, b in zip(m.image_path, m.mask_path))
hists = np.stack([f["hist"] for f in feats])
thumbs = np.stack([f["thumb"] for f in feats])
for k in ["corner_glyph", "fov_fraction", "vessel_ratio_image", "vessel_ratio_fov", "mean_r", "mean_g", "mean_b", "mask_values"]:
    m[k] = [f[k] for f in feats]
m["file_mb"] = [(D / p).stat().st_size / 1e6 for p in m.image_path]
print("distinct gray values per mask (max):", m.mask_values.max())

# ---------------------------------------------------------------- 1. class distribution
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.countplot(m, x="label_name", hue="label_name", palette=PAL, order=CLASSES, legend=False, ax=ax[0])
ax[0].set(title="Class distribution (all)", xlabel="", ylabel="Images")
sns.countplot(m, x="label_name", hue="official_split", order=CLASSES, palette=["#555555", "#BBBBBB"], hue_order=["train", "test"], ax=ax[1])
ax[1].set(title="Class distribution by split", xlabel="", ylabel="")
for a in ax:
    a.tick_params(axis="x", rotation=20)
    for c in a.containers:
        a.bar_label(c, fontsize=9)
ax[1].legend(title="Split")
fig.tight_layout()
savefig(fig, "01_class_distribution")

# ---------------------------------------------------------------- 2. sample grid (image | mask)
rng = np.random.RandomState(config.SEED)
fig, axes = plt.subplots(4, 8, figsize=(16, 8.4))
for r, cls in enumerate(CLASSES):
    idx = rng.choice(m.index[m.label_name == cls], 4, replace=False)
    for k, i in enumerate(idx):
        img = cv2.cvtColor(cv2.imread(str(D / m.image_path[i])), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(str(D / m.mask_path[i]), cv2.IMREAD_GRAYSCALE)
        for j, (arr, cmap) in enumerate([(cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA), None),
                                         (cv2.resize(msk, (384, 384), interpolation=cv2.INTER_AREA), "gray")]):
            a = axes[r, 2 * k + j]
            a.imshow(arr, cmap=cmap)
            a.axis("off")
            if j == 0:
                a.set_title(m.image_id[i], fontsize=9)
    axes[r, 0].text(-0.06, 0.5, cls, transform=axes[r, 0].transAxes, rotation=90, ha="right", va="center",
                    fontsize=12, fontweight="bold", color=PAL[cls])
fig.suptitle("Random samples per class: fundus image (left) and vessel mask (right)", fontweight="bold", y=0.95)
fig.subplots_adjust(wspace=0.04, hspace=0.12)
savefig(fig, "02_sample_grid")

# ---------------------------------------------------------------- 3. image size distribution
sz = m.assign(size=m.width.astype(str) + "x" + m.height.astype(str)).groupby(["size", "official_split"]).size().reset_index(name="n")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.barplot(sz, x="size", y="n", hue="official_split", ax=ax[0], palette=["#BBBBBB", "#555555"])
ax[0].set(title="Pixel dimensions", xlabel="width x height", ylabel="Images")
for c in ax[0].containers:
    ax[0].bar_label(c)
sns.histplot(m, x="file_mb", hue="label_name", hue_order=CLASSES, palette=PAL, element="step", bins=30, ax=ax[1])
ax[1].set(title="PNG file size on disk", xlabel="MB", ylabel="Images")
sns.move_legend(ax[1], "upper right", title="")
fig.tight_layout()
savefig(fig, "03_image_size")

# ---------------------------------------------------------------- 4. quality metrics by class
fig, ax = plt.subplots(2, 3, figsize=(14, 8))
for j, q in enumerate(QUAL):
    sns.violinplot(m, x="label_name", y=q, hue="label_name", palette=PAL, order=CLASSES, inner="quartile", cut=0, legend=False, ax=ax[0, j])
    sns.boxplot(m, x="label_name", y=q, hue="label_name", palette=PAL, order=CLASSES, legend=False, fliersize=2, ax=ax[1, j])
    for a in ax[:, j]:
        a.set(xlabel="", ylabel=QLAB[q] if j == 0 else "")
        a.tick_params(axis="x", rotation=25)
    ax[0, j].set_title(QLAB[q])
ax[0, 0].set_ylabel("Violin\n" + QLAB[QUAL[0]])
ax[1, 0].set_ylabel("Boxplot\n" + QLAB[QUAL[0]])
fig.tight_layout()
savefig(fig, "04_quality_by_class")

# ---------------------------------------------------------------- 5. mean RGB histograms per class
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
x = np.arange(256)
for a, cls in zip(ax, CLASSES):
    h = hists[(m.label_name == cls).values].mean(0)
    for c, col in enumerate(["#D62728", "#2CA02C", "#1F77B4"]):
        a.plot(x, h[c], color=col, lw=1.6, label="RGB"[c])
        a.fill_between(x, h[c], color=col, alpha=0.12)
    a.set_title(cls, color=PAL[cls])
    a.set_xlabel("Pixel value (FOV only)")
ax[0].set_ylabel("Mean fraction of pixels")
ax[0].legend(title="Channel")
fig.tight_layout()
savefig(fig, "05_rgb_histograms")

# ---------------------------------------------------------------- 6. mean image per class
fig, ax = plt.subplots(1, 4, figsize=(14, 3.9))
for a, cls in zip(ax, CLASSES):
    a.imshow(np.clip(thumbs[(m.label_name == cls).values].mean(0), 0, 255).astype(np.uint8))
    a.set_title(f"{cls} (n={(m.label_name == cls).sum()})", color=PAL[cls])
    a.axis("off")
fig.suptitle("Mean image per class", fontweight="bold")
fig.tight_layout()
savefig(fig, "06_mean_image")

# ---------------------------------------------------------------- 7. vessel density
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.kdeplot(m, x="vessel_ratio_fov", hue="label_name", hue_order=CLASSES, palette=PAL, fill=True, alpha=0.2, common_norm=False, cut=0, ax=ax[0])
ax[0].set(title="Vessel density distribution", xlabel="Vessel pixels / FOV pixels", ylabel="Density")
sns.boxplot(m, x="label_name", y="vessel_ratio_fov", hue="label_name", palette=PAL, order=CLASSES, legend=False, fliersize=2, ax=ax[1])
sns.stripplot(m, x="label_name", y="vessel_ratio_fov", order=CLASSES, color="k", size=1.8, alpha=0.35, ax=ax[1])
ax[1].set(title="Vessel density by class", xlabel="", ylabel="")
ax[1].tick_params(axis="x", rotation=20)
sns.move_legend(ax[0], "upper right", title="")
fig.tight_layout()
savefig(fig, "07_vessel_density")

# ---------------------------------------------------------------- 8. per-class correlation of quality metrics
CORR = QUAL + ["fov_fraction"]
fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
corrs = {}
for a, cls in zip(ax, CLASSES):
    c = m.loc[m.label_name == cls, CORR].corr(method="spearman")
    corrs[cls] = c
    sns.heatmap(c, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="vlag", square=True, cbar=a is ax[-1], ax=a,
                xticklabels=["blur", "bright", "contrast", "FOV"], yticklabels=["blur", "bright", "contrast", "FOV"])
    a.set_title(cls, color=PAL[cls])
fig.suptitle("Spearman correlation among quality metrics, per class", fontweight="bold")
fig.tight_layout()
savefig(fig, "08_quality_correlation")

# ---------------------------------------------------------------- summary table
SUMCOLS = QUAL + ["corner_glyph", "fov_fraction", "vessel_ratio_fov", "vessel_ratio_image", "mean_r", "mean_g", "mean_b", "file_mb"]
summ = m.groupby("label_name", observed=True)[SUMCOLS].describe().T.reset_index()
summ.columns = ["metric", "stat"] + CLASSES
summ.to_csv(config.TABLES_DIR / "eda_summary.csv", index=False)


# ---------------------------------------------------------------- insights
def kw(col):
    g = [m.loc[m.label_name == c, col].values for c in CLASSES]
    H, p = stats.kruskal(*g)
    return H / (len(m) - 1), p  # epsilon-squared effect size, p-value


def per_class(col, fmt="{:.1f}"):
    mu = m.groupby("label_name", observed=True)[col].mean()
    return ", ".join(f"{c} {fmt.format(mu[c])}" for c in CLASSES)


def pstr(p):
    return "p<1e-10" if p < 1e-10 else f"p={p:.1e}"


eff = {c: kw(c) for c in QUAL + ["fov_fraction", "vessel_ratio_fov", "mean_r", "mean_g", "mean_b"]}
# shortcut probe: can quality metrics alone predict the class? (train split only, 5-fold CV, chance = 25%)
tr = m[m.official_split == "train"]
probe_cols = QUAL + ["fov_fraction"]
skf = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
probe = lambda cols: cross_val_score(make_pipeline(StandardScaler(), RandomForestClassifier(300, random_state=config.SEED, n_jobs=-1)),
                                     tr[cols], tr.label_id, cv=skf, scoring="accuracy")
acc_q, acc_c = probe(probe_cols), probe(["mean_r", "mean_g", "mean_b"])
off = {c: corrs[c].values[np.triu_indices(4, 1)] for c in CLASSES}
bc = {c: corrs[c].loc["brightness", "contrast"] for c in CLASSES}
ct = m.groupby(["label_name", "official_split"], observed=True).size().unstack()
shift = {q: stats.mannwhitneyu(m.loc[m.official_split == "train", q], m.loc[m.official_split == "test", q]).pvalue for q in QUAL}
size_n = m.groupby(["width", "height"]).ngroups
hi_b = m.groupby("label_name", observed=True).brightness.mean()
v_rank = m.groupby("label_name", observed=True).vessel_ratio_fov.mean().sort_values()
g_share = (m.mean_g / (m.mean_r + m.mean_g + m.mean_b)).groupby(m.label_name, observed=True).mean()

nb = m[m.label_name == "Normal"].groupby("official_split").blur_score.mean()
ob = {c: m[m.label_name == c].groupby("official_split").blur_score.mean() for c in CLASSES}
gl = {c: m[m.label_name == c].groupby("official_split").corner_glyph.mean() for c in CLASSES}
ins = [
    f"**Class balance:** {len(m)} images; train {ct['train'].to_dict()}, test {ct['test'].to_dict()}. "
    f"Imbalance ratio (max/min) is {ct['train'].max() / ct['train'].min():.2f} in train and {ct['test'].max() / ct['test'].min():.2f} in test "
    "(only AMD lost images to de-duplication), so accuracy is a fair metric but macro-F1 will be reported too.",
    f"**Uniform geometry:** {size_n} distinct resolution(s) ({m.width.iloc[0]}x{m.height.iloc[0]}), all RGB PNG; "
    f"file size {m.file_mb.min():.1f}-{m.file_mb.max():.1f} MB. No resizing is needed to unify inputs, only to reduce cost.",
    f"**Brightness and contrast differ strongly by class** (Kruskal-Wallis {pstr(eff['brightness'][1])}, e2={eff['brightness'][0]:.2f} and "
    f"{pstr(eff['contrast'][1])}, e2={eff['contrast'][0]:.2f}). Mean gray: {per_class('brightness')}; std gray: {per_class('contrast')}. "
    f"AMD images are ~{hi_b['AMD'] / hi_b['Glaucoma']:.1f}x brighter than Glaucoma.",
    f"**Sharpness differs by class, driven by the Normal training images** ({pstr(eff['blur_score'][1])}, e2={eff['blur_score'][0]:.2f}). "
    f"Laplacian variance: {per_class('blur_score', '{:.2f}')}. Normal is {nb['train']:.2f} in train but {nb['test']:.2f} in test, "
    f"while the other classes agree across splits (train/test: " + ", ".join(f"{c} {ob[c]['train']:.2f}/{ob[c]['test']:.2f}" for c in CLASSES[1:]) +
    ") - the train Normal images (ids 451-600) look like a different acquisition batch.",
    f"**Overlay glyph in the image corner is a source marker:** bright pixels in the top-left {CORNER}px corner (outside the FOV) appear in "
    + ", ".join(f"{c} {gl[c]['train']:.0%}/{gl[c]['test']:.0%}" for c in CLASSES) +
    " of images (train/test). Normal is the outlier: 3% of train but 86% of test Normal images carry it, so a model can learn "
    "'no glyph (and soft focus) = Normal' on train and fail on test. Mask or crop the corner (use a circular FOV mask) before feature extraction, "
    "and expect a Normal train/test shift to show up as Normal recall errors.",
    f"**Black border differs by class:** FOV fraction of the frame {per_class('fov_fraction', '{:.3f}')} "
    f"({pstr(eff['fov_fraction'][1])}, e2={eff['fov_fraction'][0]:.2f}). Whole-image brightness/contrast/blur therefore mix scene content with border size; "
    "mask the FOV before computing features.",
    f"**Shortcut-learning risk is real:** a random forest on only the 4 quality numbers (brightness, contrast, blur, FOV fraction) reaches "
    f"{acc_q.mean():.1%} +/- {acc_q.std():.1%} 5-fold CV accuracy on train (chance 25%); mean R/G/B alone reaches {acc_c.mean():.1%} +/- {acc_c.std():.1%}. "
    "Global intensity features must be normalised (e.g. per-image colour/illumination normalisation) and results interpreted with this baseline in mind.",
    f"**Vessel density** (vessel px / FOV px) by class: {per_class('vessel_ratio_fov', '{:.3f}')}; ordered {' < '.join(v_rank.index)} "
    f"(Kruskal-Wallis {pstr(eff['vessel_ratio_fov'][1])}, e2={eff['vessel_ratio_fov'][0]:.2f}). "
    "Masks are binary (" + f"{m.mask_values.max()} gray levels), but they exist only for dataset images: using density as a feature in the app would need a vessel segmenter.",
    f"**Colour balance differs:** mean channel values inside the FOV (R/G/B) differ across classes "
    f"(R {pstr(eff['mean_r'][1])}, e2={eff['mean_r'][0]:.2f}; G e2={eff['mean_g'][0]:.2f}; B e2={eff['mean_b'][0]:.2f}); "
    f"green share of total intensity: {', '.join(f'{c} {g_share[c]:.3f}' for c in CLASSES)}. In the histograms blue is concentrated at low values, green sits mid-low and red is broad and highest; AMD shows a red saturation spike at 255.",
    f"**Metric structure:** brightness and contrast are strongly coupled (Spearman {', '.join(f'{c} {v:.2f}' for c, v in bc.items())}); "
    f"all off-diagonal |rho| range {min(np.abs(v).min() for v in off.values()):.2f}-{max(np.abs(v).max() for v in off.values()):.2f} across classes, so correlated "
    f"features are expected and regularisation/feature selection inside the CV pipeline is warranted. Train vs test distributions of the quality metrics "
    f"{'show no significant shift' if min(shift.values()) > 0.01 else 'differ'} (Mann-Whitney p: " + ", ".join(f"{k} {v:.2g}" for k, v in shift.items()) + ").",
]
(config.ROOT / "results" / "eda_insights.md").write_text("# EDA insights\n\n" + "\n".join(f"- {i}" for i in ins) + "\n")
print("\n".join(f"- {i}" for i in ins))
print("\nFigures:", *sorted(p.name for p in config.FIGURES_DIR.glob("eda_*.png")), sep="\n  ")
