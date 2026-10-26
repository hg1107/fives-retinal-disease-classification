"""07 - Compare the baseline models using ONLY the saved outputs of 06 (tables + predictions).

Model selection rule: best model = highest mean CV macro-F1 on the train split. Test numbers are reported, never used to choose.
Figures: results/figures/cmp_*.png; tables: results/tables/cmp_*.csv; text: results/comparison_summary.md.
The only data read besides the saved results is a descriptive Normal-class train-vs-test shift diagnostic (section 11).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.preprocess import load_dataset  # noqa: E402

T = config.TABLES_DIR
PRED = config.ROOT / "results" / "predictions"
CLASSES = config.CLASS_NAMES
EXPS = ["handcrafted", "handcrafted_norm", "resnet", "both"]
SHORT = {"LogisticRegression": "LogReg", "RandomForest": "RF", "SVM_RBF": "SVM", "KNN": "KNN", "XGBoost": "XGB", "GaussianNB": "GNB"}
MODELS = list(SHORT)
CPAL = dict(zip(CLASSES, ["#4C78A8", "#E45756", "#72B7B2", "#F2A93B"]))  # same class colours as before
MPAL = dict(zip(MODELS, sns.color_palette("tab10", 6)))
EPAL = dict(zip(EXPS, ["#4C78A8", "#9ECAE9", "#59A14F", "#B07AA1"]))
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({"savefig.dpi": config.FIG_DPI, "axes.titleweight": "bold"})


def savefig(fig, name):
    fig.savefig(config.FIGURES_DIR / f"cmp_{name}.png", dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


cv = pd.read_csv(T / "baseline_cv.csv")
te = pd.read_csv(T / "baseline_test.csv")
pc = pd.read_csv(T / "baseline_per_class.csv")
comp = cv.merge(te.drop(columns=["cv_f1_macro", "cv_accuracy"]), on=["experiment", "model"])
comp["experiment"] = pd.Categorical(comp.experiment, EXPS, ordered=True)
comp["model"] = pd.Categorical(comp.model, MODELS, ordered=True)
comp = comp.sort_values(["experiment", "model"]).reset_index(drop=True)
comp["cv_f1"] = comp.f1_macro_mean
comp["test_f1"] = comp.test_f1_macro


def predictions(exp, model, which):
    d = pd.read_csv(PRED / f"{exp}__{model}.csv")
    return d[d.set == which].reset_index(drop=True)


# ---------------------------------------------------------------- selection (CV only)
ranked = comp.sort_values("cv_f1", ascending=False).reset_index(drop=True)
best = ranked.iloc[0]
best_exp = best.experiment
top3 = ranked.head(3)
best_per_exp = {e: comp[comp.experiment == e].sort_values("cv_f1", ascending=False).iloc[0] for e in EXPS}
print(f"Best by CV F1: {best.model} on {best.experiment} (CV F1 {best.cv_f1:.3f}, test F1 {best.test_f1:.3f})")

# ---------------------------------------------------------------- 1. comparison table (CSV + PNG)
METRICS = [("accuracy", "Acc"), ("precision_macro", "Prec"), ("recall_macro", "Rec"), ("f1_macro", "F1"), ("roc_auc_ovr", "AUC")]
table = comp[["experiment", "model"]].copy()
for k, _ in METRICS:
    table[f"cv_{k}_mean"], table[f"cv_{k}_std"] = comp[f"{k}_mean"], comp[f"{k}_std"]
for k, _ in METRICS:
    table[f"test_{k}"] = comp[f"test_{k}"]
table["gap_f1_cv_minus_test"] = comp.gap_f1_cv_minus_test
table["gap_acc_cv_minus_test"] = comp.gap_acc_cv_minus_test
table["cv_fit_time_s"], table["fit_time_s"], table["predict_time_s"] = comp.cv_fit_time_s, comp.fit_time_s, comp.predict_time_s
table.to_csv(T / "cmp_comparison_table.csv", index=False)

cols = [f"CV {s}" for _, s in METRICS] + [f"Test {s}" for _, s in METRICS] + ["Gap F1"]
cells = pd.DataFrame({f"CV {s}": table[f"cv_{k}_mean"].map("{:.3f}".format) + "±" + table[f"cv_{k}_std"].map("{:.3f}".format) for k, s in METRICS})
for k, s in METRICS:
    cells[f"Test {s}"] = table[f"test_{k}"].map("{:.3f}".format)
cells["Gap F1"] = table.gap_f1_cv_minus_test.map("{:+.3f}".format)
vals = pd.DataFrame({f"CV {s}": table[f"cv_{k}_mean"] for k, s in METRICS} | {f"Test {s}": table[f"test_{k}"] for k, s in METRICS})
fig, ax = plt.subplots(figsize=(21, 11))
ax.axis("off")
rowlab = [f"{e} | {SHORT[m]}" for e, m in zip(table.experiment, table.model)]
tb = ax.table(cellText=cells[cols].values, rowLabels=rowlab, colLabels=cols, loc="center", cellLoc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(8.5)
tb.scale(1, 1.35)
for j, c in enumerate(cols):
    tb[0, j].set_facecolor("#DDDDDD")
    tb[0, j].set_text_props(fontweight="bold")
    if c in vals:
        for e in EXPS:  # light green = best of the experiment block, dark green + bold = best overall (per metric)
            idx = np.where(table.experiment == e)[0]
            tb[idx[np.argmax(vals[c].values[idx])] + 1, j].set_facecolor("#C7E9C0")
        i = int(np.argmax(vals[c].values))
        tb[i + 1, j].set_facecolor("#41AB5D")
        tb[i + 1, j].set_text_props(fontweight="bold")
ax.set_title("All models x experiments: 5-fold CV on train (mean±std) vs official test\n"
             "dark green = best overall per metric, light green = best within experiment; Gap F1 = CV F1 - test F1", fontsize=11, fontweight="bold")
savefig(fig, "01_comparison_table")

# ---------------------------------------------------------------- 2. grouped bars (test metrics)
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
long = comp.melt(id_vars=["experiment", "model"], value_vars=[f"test_{k}" for k, _ in METRICS[:4]], var_name="metric", value_name="v")
long["metric"] = long.metric.map({f"test_{k}": s for k, s in METRICS[:4]})
long["model"] = long.model.astype(str).map(SHORT)
for a, e in zip(axes.ravel(), EXPS):
    sns.barplot(long[long.experiment == e], x="model", y="v", hue="metric", order=list(SHORT.values()), palette="Set2", ax=a)
    a.set(title=e, xlabel="", ylabel="test score" if e in (EXPS[0], EXPS[2]) else "", ylim=(0, 0.85))
    a.axhline(0.25, color="gray", ls=":", lw=1)
    a.legend(loc="upper right", ncol=4, fontsize=8, title="")
fig.suptitle("Test performance (macro precision / recall / F1, accuracy); dotted line = chance (0.25)", fontweight="bold")
fig.tight_layout()
savefig(fig, "02_test_metrics_bars")

# ---------------------------------------------------------------- 3. CV-vs-test gap
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
g = comp.assign(m=comp.model.astype(str).map(SHORT))
for a, col, ttl in [(ax[0], "gap_f1_cv_minus_test", "Macro-F1 gap (CV - test)"), (ax[1], "gap_acc_cv_minus_test", "Accuracy gap (CV - test)")]:
    sns.barplot(g, x="m", y=col, hue="experiment", hue_order=EXPS, order=list(SHORT.values()), palette=EPAL, ax=a)
    a.axhline(0, color="k", lw=1)
    a.set(title=ttl, xlabel="", ylabel="gap")
ax[0].legend(title="", fontsize=8)
ax[1].legend_.remove()
fig.suptitle("Optimism of CV relative to the official test split (positive = test is worse)", fontweight="bold")
fig.tight_layout()
savefig(fig, "03_cv_test_gap")

# ---------------------------------------------------------------- 4. per-class recall heatmaps (test), Normal highlighted
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
for a, e in zip(axes.ravel(), EXPS):
    d = pc[pc.experiment == e]
    R = d.pivot(index="model", columns="class", values="test_recall").loc[MODELS, CLASSES]
    C = d.pivot(index="model", columns="class", values="cv_recall_mean").loc[MODELS, CLASSES]
    ann = R.map("{:.2f}".format)
    ann["Normal"] = R["Normal"].map("{:.2f}".format) + "\n(cv " + C["Normal"].map("{:.2f}".format) + ")"
    sns.heatmap(R, annot=ann, fmt="", cmap="YlGnBu", vmin=0, vmax=1, ax=a, cbar=False, annot_kws={"fontsize": 9}, yticklabels=[SHORT[m] for m in MODELS])
    a.add_patch(plt.Rectangle((0, 0), 1, len(MODELS), fill=False, ec="#D62728", lw=3))
    a.set(title=e, xlabel="", ylabel="")
    a.tick_params(axis="x", rotation=15)
fig.suptitle("Per-class recall on test (Normal column outlined in red, CV recall in brackets)", fontweight="bold")
fig.tight_layout()
savefig(fig, "04_per_class_recall")


# ---------------------------------------------------------------- 5. confusion matrices
def cm_of(exp, model):
    return pd.read_csv(T / "baseline_confusion" / f"{exp}__{model}.csv", index_col=0).values


picks = [(r.experiment, r.model, f"best of '{r.experiment}' by CV F1") for r in best_per_exp.values()]
for i, r in enumerate(top3.itertuples(), 1):
    key = (r.experiment, r.model)
    if key in [(a, b) for a, b, _ in picks]:
        picks = [(a, b, t + f"; top-{i} overall") if (a, b) == key else (a, b, t) for a, b, t in picks]
    else:
        picks.append((r.experiment, r.model, f"top-{i} overall by CV F1"))
fig, axes = plt.subplots(len(picks), 2, figsize=(11, 4.1 * len(picks)))
for row, (e, m, why) in zip(axes, picks):
    cm = cm_of(e, m)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=row[0], xticklabels=[c[:6] for c in CLASSES], yticklabels=CLASSES)
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=row[1],
                xticklabels=[c[:6] for c in CLASSES], yticklabels=False)
    row[0].set_title(f"{e} | {SHORT[m]} (counts)\n{why}", fontsize=9)
    row[1].set_title("row-normalised (recall)", fontsize=9)
    for a in row:
        a.set(xlabel="predicted", ylabel="true" if a is row[0] else "")
fig.tight_layout()
savefig(fig, "05_confusion_matrices")

# most confused pairs: mean row-normalised off-diagonal rate over all 24 models, and for the selected model
rates, pair_rows = [], []
for e in EXPS:
    for m in MODELS:
        cm = cm_of(e, m)
        rates.append(cm / cm.sum(1, keepdims=True))
mean_rate = np.mean(rates, axis=0)
bcm = cm_of(best.experiment, best.model)
brate = bcm / bcm.sum(1, keepdims=True)
for i in range(4):
    for j in range(4):
        if i != j:
            pair_rows.append({"true": CLASSES[i], "predicted": CLASSES[j], "mean_rate_all_24_models": mean_rate[i, j],
                              "rate_selected_model": brate[i, j], "count_selected_model": int(bcm[i, j])})
pairs = pd.DataFrame(pair_rows).sort_values("mean_rate_all_24_models", ascending=False)
pairs.to_csv(T / "cmp_confused_pairs.csv", index=False)

# ---------------------------------------------------------------- 6. ROC (one-vs-rest) for the best experiment, one panel per model
fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=True)
grid = np.linspace(0, 1, 201)
for a, m in zip(axes.ravel(), MODELS):
    d = predictions(best_exp, m, "test")
    tprs = []
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(d.y_true == i, d[f"proba_{i}"])
        tprs.append(np.interp(grid, fpr, tpr))
        a.plot(fpr, tpr, color=CPAL[c], lw=1.6, label=f"{c} (AUC {auc(fpr, tpr):.2f})")
    mt = np.mean(tprs, axis=0)
    a.plot(grid, mt, "k--", lw=2, label=f"macro-avg (AUC {auc(grid, mt):.2f})")
    a.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    a.set(title=SHORT[m], xlabel="false positive rate" if a in axes[1] else "", ylabel="true positive rate" if a in axes[:, 0] else "")
    a.legend(fontsize=7.5, loc="lower right")
fig.suptitle(f"One-vs-rest ROC on test: experiment '{best_exp}' (best by CV F1)", fontweight="bold")
fig.tight_layout()
savefig(fig, "06_roc_curves")

# ---------------------------------------------------------------- 7. fold-wise CV scores
fold_rows = []
for e in EXPS:
    for m in MODELS:
        d = predictions(e, m, "cv_oof")
        for k, g_ in d.groupby("fold"):
            fold_rows.append({"experiment": e, "model": SHORT[m], "fold": k, "f1_macro": f1_score(g_.y_true, g_.y_pred, average="macro"),
                              "accuracy": accuracy_score(g_.y_true, g_.y_pred)})
folds = pd.DataFrame(fold_rows)
folds.to_csv(T / "cmp_cv_fold_scores.csv", index=False)
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for a, col, ttl in [(axes[0], "f1_macro", "Macro-F1"), (axes[1], "accuracy", "Accuracy")]:
    sns.boxplot(folds, x="model", y=col, hue="experiment", hue_order=EXPS, order=list(SHORT.values()), palette=EPAL, fliersize=0, ax=a)
    sns.stripplot(folds, x="model", y=col, hue="experiment", hue_order=EXPS, order=list(SHORT.values()), dodge=True, palette=["k"] * 4, size=2.5, alpha=0.6, legend=False, ax=a)
    a.set(ylabel=f"{ttl} per CV fold", xlabel="")
    a.legend(title="", ncol=4, fontsize=8, loc="lower right")
axes[1].legend_.remove()
fig.suptitle("Fold-wise CV scores (5 folds, train split)", fontweight="bold")
fig.tight_layout()
savefig(fig, "07_cv_fold_boxplots")

# ---------------------------------------------------------------- 8. time vs macro-F1
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey="row")
mk = dict(zip(EXPS, ["o", "s", "^", "D"]))
for r, (ycol, ylab) in enumerate([("cv_f1", "CV macro-F1"), ("test_f1", "test macro-F1")]):
    for c, (xcol, xlab) in enumerate([("fit_time_s", "fit time on full train (s)"), ("predict_time_s", "predict time on test (s)")]):
        a = axes[r, c]
        for e in EXPS:
            for m in MODELS:
                row = comp[(comp.experiment == e) & (comp.model == m)].iloc[0]
                a.scatter(row[xcol], row[ycol], color=MPAL[m], marker=mk[e], s=60, edgecolor="k", lw=0.5)
        a.set(xscale="log", xlabel=xlab if r == 1 else "", ylabel=ylab if c == 0 else "")
h = [plt.Line2D([], [], color=MPAL[m], marker="o", ls="", label=SHORT[m]) for m in MODELS]
h += [plt.Line2D([], [], color="gray", marker=mk[e], ls="", label=e) for e in EXPS]
axes[0, 1].legend(handles=h, fontsize=8, ncol=2, loc="lower right")
fig.suptitle("Cost vs quality", fontweight="bold")
fig.tight_layout()
savefig(fig, "08_time_vs_f1")

# ---------------------------------------------------------------- 9. McNemar: top-2 models by CV F1
a_, b_ = ranked.iloc[0], ranked.iloc[1]
pa, pb = predictions(a_.experiment, a_.model, "test"), predictions(b_.experiment, b_.model, "test")
assert (pa.image_id.values == pb.image_id.values).all()
ca, cb = (pa.y_pred == pa.y_true).values, (pb.y_pred == pb.y_true).values
n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
nd = n10 + n01
p_exact = stats.binomtest(min(n10, n01), nd, 0.5).pvalue if nd else 1.0
chi = (abs(n10 - n01) - 1) ** 2 / nd if nd else 0.0
mc = pd.DataFrame([{"model_A": f"{a_.experiment} | {a_.model}", "model_B": f"{b_.experiment} | {b_.model}", "both_correct": int((ca & cb).sum()),
                    "only_A_correct": n10, "only_B_correct": n01, "both_wrong": int((~ca & ~cb).sum()), "chi2_cc": chi,
                    "p_chi2": stats.chi2.sf(chi, 1), "p_exact": p_exact, "acc_A": ca.mean(), "acc_B": cb.mean()}])
mc.to_csv(T / "cmp_mcnemar.csv", index=False)
print("\nMcNemar (top-2 by CV F1):\n", mc.T.to_string(header=False))

# ---------------------------------------------------------------- 10. handcrafted vs handcrafted_norm
nrm = pc[pc["class"] == "Normal"].set_index(["experiment", "model"])
rows = []
for m in MODELS:
    r = {"model": SHORT[m]}
    for e, tag in [("handcrafted", "base"), ("handcrafted_norm", "norm")]:
        c = comp[(comp.experiment == e) & (comp.model == m)].iloc[0]
        r[f"cv_f1_{tag}"], r[f"test_f1_{tag}"] = c.cv_f1, c.test_f1
        r[f"normal_recall_test_{tag}"] = nrm.loc[(e, m), "test_recall"]
    for k in ("cv_f1", "test_f1", "normal_recall_test"):
        r[f"d_{k}"] = r[f"{k}_norm"] - r[f"{k}_base"]
    rows.append(r)
nt = pd.DataFrame(rows)
mean_row = {"model": "MEAN", **{c: nt[c].mean() for c in nt.columns[1:]}}
nt = pd.concat([nt, pd.DataFrame([mean_row])], ignore_index=True)
nt.to_csv(T / "cmp_norm_effect.csv", index=False)
fig, ax = plt.subplots(figsize=(14, 3.6))
ax.axis("off")
show = nt[["model", "cv_f1_base", "cv_f1_norm", "d_cv_f1", "test_f1_base", "test_f1_norm", "d_test_f1", "normal_recall_test_base", "normal_recall_test_norm", "d_normal_recall_test"]]
hdr = ["model", "CV F1 base", "CV F1 norm", "Δ", "test F1 base", "test F1 norm", "Δ", "Normal rec. base", "Normal rec. norm", "Δ"]
tb = ax.table(cellText=[[r.model] + [f"{v:+.3f}" if h == "Δ" else f"{v:.3f}" for v, h in zip(r[1:], hdr[1:])] for r in show.itertuples(index=False)],
              colLabels=hdr, loc="center", cellLoc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(9)
tb.scale(1, 1.5)
for j in range(len(hdr)):
    tb[0, j].set_facecolor("#DDDDDD")
    tb[0, j].set_text_props(fontweight="bold")
    if hdr[j] == "Δ":
        for i in range(1, len(show) + 1):
            v = show.iloc[i - 1, j]
            tb[i, j].set_facecolor("#C7E9C0" if v > 0.005 else "#FDD0CE" if v < -0.005 else "white")
ax.set_title("Did per-image normalisation help? (green = better, red = worse)", fontweight="bold")
savefig(fig, "10_norm_effect")

# ---------------------------------------------------------------- 11. Normal-class shift diagnostic (descriptive only)
shift = {}
for e in ("handcrafted", "handcrafted_norm"):
    Xtr, ytr, Xte, yte = load_dataset(e)
    n = {}
    for k, c in enumerate(CLASSES):
        p = np.array([stats.mannwhitneyu(Xtr[ytr == k][f], Xte[yte == k][f]).pvalue for f in Xtr.columns])
        n[c] = int((p < 1e-3).sum())
    shift[e] = n
man = pd.read_csv(config.DATA_DIR / "manifest.csv")
blur = man[man.label_name == "Normal"].groupby("official_split").blur_score.mean()
nr_cv = pc[pc["class"] == "Normal"].cv_recall_mean.mean()
nr_te = pc[pc["class"] == "Normal"].test_recall.mean()
nrm_exp = pc[pc["class"] == "Normal"].groupby("experiment").agg(cv=("cv_recall_mean", "mean"), test=("test_recall", "mean")).reindex(EXPS)
normal_to = {CLASSES[j]: mean_rate[0, j] for j in range(4)}
other_cv = pc[pc["class"] != "Normal"].cv_recall_mean.mean()
other_te = pc[pc["class"] != "Normal"].test_recall.mean()
pd.DataFrame(shift).T.to_csv(T / "cmp_normal_shift_features.csv")

# ---------------------------------------------------------------- summary markdown
se = np.sqrt(best.test_accuracy * (1 - best.test_accuracy) / 199)
sel = pc[(pc.experiment == best.experiment) & (pc.model == best.model)].set_index("class")
top_pairs = pairs.head(4)
md = f"""# Model comparison summary

All numbers come from `results/tables/baseline_*.csv` (script 06). CV = 5-fold stratified CV on the train split (594 images);
test = the official test split (199 images), used once. Identical preprocessing (median impute, RobustScaler, variance filter,
correlation filter 0.95, SelectKBest k=30) and identical folds for all 24 model x experiment combinations. Hyperparameters are still at defaults.

## Selected model (by CV macro-F1, not test)

**{best.model}** on **{best.experiment}** features: CV macro-F1 {best.cv_f1:.3f} ± {best.f1_macro_std:.3f}, CV accuracy {best.accuracy_mean:.3f} ± {best.accuracy_std:.3f}.

Test result (single evaluation): accuracy {best.test_accuracy:.3f}, macro-F1 {best.test_f1:.3f}, macro precision {best.test_precision_macro:.3f},
macro recall {best.test_recall_macro:.3f}, ROC-AUC (OvR) {best.test_roc_auc_ovr:.3f}. Gap CV - test macro-F1: {best.gap_f1_cv_minus_test:+.3f}.

| class | test precision | test recall | test F1 | CV recall |
|---|---|---|---|---|
""" + "\n".join(f"| {c} | {sel.loc[c, 'test_precision']:.2f} | {sel.loc[c, 'test_recall']:.2f} | {sel.loc[c, 'test_f1']:.2f} | {sel.loc[c, 'cv_recall_mean']:.2f} |" for c in CLASSES) + f"""

The CV lead of this model over the runner-up ({ranked.iloc[1].model}, {ranked.iloc[1].experiment}, CV F1 {ranked.iloc[1].cv_f1:.3f}) is {best.cv_f1 - ranked.iloc[1].cv_f1:.3f},
{'smaller' if best.cv_f1 - ranked.iloc[1].cv_f1 < best.f1_macro_std else 'larger'} than one CV standard deviation ({best.f1_macro_std:.3f}){', so the ranking among the top models is not statistically meaningful' if best.cv_f1 - ranked.iloc[1].cv_f1 < best.f1_macro_std else ''}.
McNemar on test predictions of the top-2 by CV F1 ({mc.model_A[0]} vs {mc.model_B[0]}): only-A-correct {n10}, only-B-correct {n01},
exact p = {p_exact:.3f}, chi-square (continuity-corrected) p = {mc.p_chi2[0]:.3f}; {'no significant difference' if p_exact > 0.05 else 'a significant difference'} at 0.05.

Best model per experiment by CV F1: """ + "; ".join(f"{e}: {r.model} (CV {r.cv_f1:.3f}, test {r.test_f1:.3f})" for e, r in best_per_exp.items()) + f""".

## Most confused class pairs (test, mean over all 24 models; selected model in brackets)

""" + "\n".join(f"- true {r.true} predicted as {r.predicted}: {r.mean_rate_all_24_models:.1%} of that class [{r.rate_selected_model:.1%}, {r.count_selected_model} images]" for r in top_pairs.itertuples()) + f"""

Where true Normal test images go on average: """ + ", ".join(f"{c} {v:.0%}" for c, v in normal_to.items()) + f""".

## Normal-class source shift

Normal images in the train split come from a different source than the Normal images in the test split (EDA: a corner overlay glyph is present in 3% of
train Normal but 86% of test Normal images; the other classes show it at 80-98% in both splits).
- Mean blur score of Normal images: {blur['train']:.2f} in train vs {blur['test']:.2f} in test.
- Features whose Normal distribution differs between train and test (Mann-Whitney p < 0.001): {shift['handcrafted']['Normal']}/116 (handcrafted), {shift['handcrafted_norm']['Normal']}/116 (handcrafted_norm).
  For AMD / DR / Glaucoma the counts are {shift['handcrafted']['AMD']} / {shift['handcrafted']['Diabetic Retinopathy']} / {shift['handcrafted']['Glaucoma']} (handcrafted) and {shift['handcrafted_norm']['AMD']} / {shift['handcrafted_norm']['Diabetic Retinopathy']} / {shift['handcrafted_norm']['Glaucoma']} (normalised).
- Consequence: mean Normal recall over all 24 models is {nr_cv:.2f} in CV but {nr_te:.2f} on test; the other three classes average {other_cv:.2f} in CV vs {other_te:.2f} on test.
  By experiment (CV / test): """ + "; ".join(f"{e} {r.cv:.2f}/{r.test:.2f}" for e, r in nrm_exp.iterrows()) + f""".
  Models trained on the train Normal source learn "that look" as Normal; test Normal images look different and are mostly assigned to other classes.
- Per-image normalisation (illumination correction + channel standardisation) did not remove the shift (see `cmp_norm_effect.csv`): mean change over the six models in CV F1 {nt.iloc[-1].d_cv_f1:+.3f}, test F1 {nt.iloc[-1].d_test_f1:+.3f}, Normal test recall {nt.iloc[-1].d_normal_recall_test:+.3f}.
- CV folds are drawn from the train split only, so every CV Normal image comes from the same source as the Normal images the model trained on; CV cannot see this shift; the CV-to-test gap (macro-F1) ranges from {comp.gap_f1_cv_minus_test.min():+.3f} to {comp.gap_f1_cv_minus_test.max():+.3f} across the 24 combinations.

## Limitations

- Test set size is 199 images; the standard error of the selected model's test accuracy is about ±{se:.3f}. Differences of a few points between models are within noise.
- One train/test partition and one CV seed; no repeated CV. Fold-wise CV F1 std is about {comp.f1_macro_std.mean():.3f}.
- The shift is a property of the data source, not something the models can learn from this training set: reported test scores are therefore a pessimistic-but-honest estimate for images from the test source, and CV scores are optimistic for it. Which one matters for deployment depends on which source future images resemble.
- Hyperparameters, feature count (k=30) and the ResNet/both feature sets were not tuned here, so ResNet features are probably under-used.
- Earlier diagnostics (the Normal shift test, a quick random-forest probe) looked at test data and motivated creating the `handcrafted_norm` variant; no model or hyperparameter was selected from test results, and the selected model above is chosen on CV only, but the existence of that experiment is not test-blind.
- Vessel features are only moderately correlated with ground-truth vessel density (Pearson 0.56), colour features dominate feature rankings, and global intensity alone predicted class at ~66% in EDA, so part of the performance may be acquisition-style shortcuts rather than pathology.

## Note on regression metrics

MAE, MSE, RMSE and R² are regression metrics and do not apply to this 4-class classification task; accuracy, macro precision/recall/F1, per-class recall, confusion matrices and one-vs-rest ROC-AUC are used instead.
"""
(config.ROOT / "results" / "comparison_summary.md").write_text(md)
print("\n" + md)
