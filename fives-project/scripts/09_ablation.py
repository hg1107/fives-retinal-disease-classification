"""09 - Ablation, source-shift analysis, robustness and error analysis for the final model.

The final model (models/best_model.joblib, a soft-voting ensemble of tuned pipelines on `handcrafted_norm`) defines the
"pipeline settings": ablations clone its (unfitted) hyperparameters and refit on the feature subset. Nothing here is used to
choose a model; test numbers are for reporting. Analyses 2 (train-vs-test classifier, per-feature AUC) look at test FEATURES to
quantify covariate shift; they never feed back into training.
"""
import json
import os
import sys
import warnings
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support, precision_score, recall_score,  # noqa: E402
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline as skl_pipeline  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import features as F  # noqa: E402
from src.preprocess import load_dataset  # noqa: E402

SEED = config.SEED
T = config.TABLES_DIR
CLASSES = config.CLASS_NAMES
LABELS = list(range(4))
CPAL = dict(zip(CLASSES, ["#4C78A8", "#E45756", "#72B7B2", "#F2A93B"]))
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({"savefig.dpi": config.FIG_DPI, "axes.titleweight": "bold"})

final = joblib.load(config.MODELS_DIR / "best_model.joblib")
meta = json.loads((config.MODELS_DIR / "best_model_meta.json").read_text())
EXP = meta["feature_set"]
print(f"Final model: {meta['candidate']} on {EXP}")
Xtr, ytr, Xte, yte = load_dataset(EXP)
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr))
ids = pd.read_csv(config.DATA_DIR / "features_resnet_ids.csv")
ids_te = ids.image_id[ids.split == "test"].values
man = pd.read_csv(config.DATA_DIR / "manifest.csv").set_index("image_id")
group = pd.Series([c.split("_")[0] for c in Xtr.columns], index=Xtr.columns)


def macro_f1(y, p, labels=LABELS):
    return f1_score(y, p, labels=labels, average="macro", zero_division=0)


def cv_and_test(Xa, ya, Xb, yb, est=None, labels=LABELS):
    """5-fold CV on (Xa, ya) with clones of the final model, then a fit on all of (Xa, ya) and one evaluation on (Xb, yb)."""
    est = est if est is not None else final
    fl = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xa, ya))

    def one(a, b):
        m = clone(est).fit(Xa.iloc[a], ya[a])
        return b, m.predict(Xa.iloc[b]), m.predict_proba(Xa.iloc[b])
    out = Parallel(n_jobs=5)(delayed(one)(a, b) for a, b in fl)
    f1s = [macro_f1(ya[b], p, labels) for b, p, _ in out]
    oof_p = np.zeros(len(ya), int)
    oof_pr = np.zeros((len(ya), len(np.unique(ya))))
    for b, p, pr in out:
        oof_p[b], oof_pr[b] = p, pr
    m = clone(est).fit(Xa, ya)
    pt, prt = m.predict(Xb), m.predict_proba(Xb)
    return {"cv_f1_mean": np.mean(f1s), "cv_f1_std": np.std(f1s, ddof=1), "test_f1": macro_f1(yb, pt, labels),
            "test_acc": accuracy_score(yb, pt), "oof_pred": oof_p, "oof_proba": oof_pr, "test_pred": pt, "test_proba": prt}


# ================================================================ 1. feature-group ablation
print("\n[1] Feature-group ablation")
cfgs = {"all (handcrafted_norm)": list(Xtr.columns)}
for g in ["color", "tex", "vessel", "disc"]:
    cfgs[f"only {g}"] = list(group.index[group == g])
for g in ["color", "tex", "vessel", "disc"]:
    cfgs[f"without {g}"] = list(group.index[group != g])
rows = []
full = None
for name, cols in cfgs.items():
    r = cv_and_test(Xtr[cols], ytr, Xte[cols], yte)
    if full is None:
        full = r
    rows.append({"config": name, "n_features": len(cols), **{k: r[k] for k in ("cv_f1_mean", "cv_f1_std", "test_f1", "test_acc")}})
    print(f"  {name:24s} n={len(cols):4d} CV F1 {r['cv_f1_mean']:.3f}±{r['cv_f1_std']:.3f} | test F1 {r['test_f1']:.3f}")
Rtr, ytr_r, Rte, yte_r = load_dataset("resnet")
r = cv_and_test(Rtr, ytr_r, Rte, yte_r)
rows.append({"config": "ResNet50 only", "n_features": Rtr.shape[1], **{k: r[k] for k in ("cv_f1_mean", "cv_f1_std", "test_f1", "test_acc")}})
print(f"  {'ResNet50 only':24s} n={Rtr.shape[1]:4d} CV F1 {r['cv_f1_mean']:.3f}±{r['cv_f1_std']:.3f} | test F1 {r['test_f1']:.3f}")
abl = pd.DataFrame(rows)
abl.to_csv(T / "abl_feature_groups.csv", index=False)
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(abl))
ax.bar(x - 0.2, abl.cv_f1_mean, 0.4, yerr=abl.cv_f1_std, capsize=3, color="#4C78A8", edgecolor="k", label="CV (train)")
ax.bar(x + 0.2, abl.test_f1, 0.4, color="#F2A93B", edgecolor="k", hatch="//", label="test")
ax.axhline(0.25, color="gray", ls=":", label="chance")
ax.set_xticks(x, abl.config, rotation=30, ha="right")
ax.set(ylabel="macro-F1", ylim=(0, 0.9), title="Feature-group ablation (final model settings)")
ax.legend()
fig.savefig(config.FIGURES_DIR / "abl_01_feature_groups.png", dpi=config.FIG_DPI, bbox_inches="tight")
plt.close(fig)

# ================================================================ 2. shortcut / source-shift analysis
print("\n[2] Source shift")
pt, pp = final.predict(Xte), final.predict_proba(Xte)  # the saved final model, already fitted on the full train split
nn = yte != 0
# (a) final 4-class model scored on non-Normal test images (predicting Normal counts as an error)
rec_all = recall_score(yte, pt, labels=LABELS, average=None, zero_division=0)
f1_non_normal_4cls = macro_f1(yte[nn], pt[nn], [1, 2, 3])
# (b) 3-class model trained without Normal at all
tr_nn = ytr != 0
r3 = cv_and_test(Xtr[tr_nn].reset_index(drop=True), ytr[tr_nn], Xte[nn].reset_index(drop=True), yte[nn], labels=[1, 2, 3])
rec3 = recall_score(yte[nn], r3["test_pred"], labels=[1, 2, 3], average=None, zero_division=0)
cvrec3 = [recall_score(ytr[tr_nn][b], r3["oof_pred"][b], labels=[1, 2, 3], average=None, zero_division=0) for b in range(0)] or None
rec3_cv = recall_score(ytr[tr_nn], r3["oof_pred"], labels=[1, 2, 3], average=None, zero_division=0)
rec4_cv = recall_score(ytr, full["oof_pred"], labels=LABELS, average=None, zero_division=0)
nonnormal = pd.DataFrame([
    {"model": "final 4-class, scored on non-Normal test images", "macro_f1_3cls": f1_non_normal_4cls,
     **{f"recall_{CLASSES[i]}": rec_all[i] for i in (1, 2, 3)}, "cv_recall_" + CLASSES[1]: rec4_cv[1], "cv_recall_" + CLASSES[2]: rec4_cv[2], "cv_recall_" + CLASSES[3]: rec4_cv[3]},
    {"model": "3-class (Normal removed from train and test)", "macro_f1_3cls": r3["test_f1"],
     **{f"recall_{CLASSES[i + 1]}": rec3[i] for i in range(3)}, "cv_recall_" + CLASSES[1]: rec3_cv[0], "cv_recall_" + CLASSES[2]: rec3_cv[1], "cv_recall_" + CLASSES[3]: rec3_cv[2],
     "cv_macro_f1_3cls": r3["cv_f1_mean"]}])
nonnormal.to_csv(T / "abl_normal_excluded.csv", index=False)
print(nonnormal.round(3).to_string(index=False))

# (c) Normal-vs-rest from the 4-class probabilities, CV (out-of-fold) vs test
def binary_row(name, y, proba, pred):
    yb = (y == 0).astype(int)
    pb = (pred == 0).astype(int)
    return {"split": name, "auc_normal_vs_rest": roc_auc_score(yb, proba[:, 0]), "precision": precision_score(yb, pb, zero_division=0),
            "recall": recall_score(yb, pb), "f1": f1_score(yb, pb), "n_normal": int(yb.sum()), "n_pred_normal": int(pb.sum())}


binary = pd.DataFrame([binary_row("CV (train, out-of-fold)", ytr, full["oof_proba"], full["oof_pred"]), binary_row("test", yte, pp, pt)])
binary.to_csv(T / "abl_normal_vs_rest.csv", index=False)
print(binary.round(3).to_string(index=False))

# (d) train-vs-test classifier on images of one class (Normal; other classes as controls)
FS = {"handcrafted": load_dataset("handcrafted"), "handcrafted_norm": load_dataset("handcrafted_norm"), "resnet": load_dataset("resnet")}
cvs = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)


def split_auc(Xa, Xb, kind):
    X = pd.concat([Xa, Xb], ignore_index=True)
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    clf = (LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced", random_state=SEED) if kind == "LogReg"
           else RandomForestClassifier(300, class_weight="balanced", random_state=SEED, n_jobs=-1))
    pipe = skl_pipeline(SimpleImputer(strategy="median"), RobustScaler(), clf)
    s = cross_val_score(pipe, X, y, cv=cvs, scoring="roc_auc", n_jobs=5)
    return s.mean(), s.std()


shift_rows = []
for k, c in enumerate(CLASSES):
    for fs, (a, ya, b, yb) in FS.items():
        for kind in ("LogReg", "RandomForest"):
            m_, s_ = split_auc(a[ya == k], b[yb == k], kind)
            shift_rows.append({"class": c, "feature_set": fs, "model": kind, "train_vs_test_auc": m_, "auc_std": s_, "n_train": int((ya == k).sum()), "n_test": int((yb == k).sum())})
shift = pd.DataFrame(shift_rows)
shift.to_csv(T / "abl_train_vs_test_auc.csv", index=False)
print("\nTrain-vs-test AUC (0.5 = no shift):\n", shift.pivot_table(index=["class", "feature_set"], columns="model", values="train_vs_test_auc").round(3).to_string())

# (e) which feature groups carry the shift (Normal only)
grp_rows = []
for fs in ("handcrafted", "handcrafted_norm"):
    a, ya, b, yb = FS[fs]
    gi = pd.Series([c.split("_")[0] for c in a.columns], index=a.columns)
    for g in ["color", "tex", "vessel", "disc"]:
        cols = list(gi.index[gi == g])
        for kind in ("LogReg", "RandomForest"):
            m_, s_ = split_auc(a.loc[ya == 0, cols], b.loc[yb == 0, cols], kind)
            grp_rows.append({"feature_set": fs, "group": g, "n_features": len(cols), "model": kind, "train_vs_test_auc": m_, "auc_std": s_})
grp = pd.DataFrame(grp_rows)
grp.to_csv(T / "abl_shift_by_group.csv", index=False)
print("\nShift by feature group (Normal only):\n", grp.pivot_table(index=["feature_set", "group"], columns="model", values="train_vs_test_auc").round(3).to_string())
uni = []
for fs in ("handcrafted", "handcrafted_norm"):
    a, ya, b, yb = FS[fs]
    for f in a.columns:
        u = mannwhitneyu(a.loc[ya == 0, f], b.loc[yb == 0, f]).statistic / ((ya == 0).sum() * (yb == 0).sum())
        uni.append({"feature_set": fs, "feature": f, "group": f.split("_")[0], "auc_train_vs_test": u, "abs_dev_from_0.5": abs(u - 0.5)})
uni = pd.DataFrame(uni).sort_values(["feature_set", "abs_dev_from_0.5"], ascending=[True, False])
uni.to_csv(T / "abl_shift_single_feature_auc.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
pv = shift[shift.model == "LogReg"].pivot(index="class", columns="feature_set", values="train_vs_test_auc").loc[CLASSES]
pv.plot.bar(ax=ax[0], edgecolor="k", color=["#9ECAE9", "#4C78A8", "#59A14F"])
ax[0].axhline(0.5, color="k", ls=":")
ax[0].set(title="Train-vs-test classifier AUC per class (LogReg, 0.5 = no shift)", ylabel="AUC", xlabel="", ylim=(0.4, 1.02))
ax[0].tick_params(axis="x", rotation=15)
gp = grp[grp.model == "LogReg"].pivot(index="group", columns="feature_set", values="train_vs_test_auc").loc[["color", "tex", "vessel", "disc"]]
gp.plot.bar(ax=ax[1], edgecolor="k", color=["#9ECAE9", "#4C78A8"])
ax[1].axhline(0.5, color="k", ls=":")
ax[1].set(title="Normal images: shift AUC by feature group (LogReg)", ylabel="AUC", xlabel="", ylim=(0.4, 1.02))
ax[1].tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "abl_02_source_shift.png", dpi=config.FIG_DPI, bbox_inches="tight")
plt.close(fig)

# ================================================================ 3. robustness (re-extract features from perturbed test images)
print("\n[3] Robustness")
PERT = {"clean": None, "blur sigma=2": partial(F.perturb_blur, sigma=2), "blur sigma=4": partial(F.perturb_blur, sigma=4),
        "brightness x0.7": partial(F.perturb_brightness, gain=0.7), "brightness x1.3": partial(F.perturb_brightness, gain=1.3),
        "noise sigma=10": partial(F.perturb_noise, sigma=10), "noise sigma=20": partial(F.perturb_noise, sigma=20)}
te_paths = [config.DATASET_PATH / man.loc[i, "image_path"] for i in ids_te]
rob = []
for name, fn in PERT.items():
    feats = Parallel(n_jobs=8)(delayed(F.extract_handcrafted_norm)(p, fn) for p in te_paths)
    Xp = pd.DataFrame(feats)[list(Xtr.columns)]
    assert np.isfinite(Xp.values).all(), f"non-finite features under {name}"
    pr = final.predict(Xp)
    rec = recall_score(yte, pr, labels=LABELS, average=None, zero_division=0)
    rob.append({"perturbation": name, "macro_f1": macro_f1(yte, pr), "accuracy": accuracy_score(yte, pr), **{f"recall_{c}": rec[i] for i, c in enumerate(CLASSES)},
                "share_predicted_Normal": float((pr == 0).mean()), "largest_predicted_class_share": float(np.bincount(pr, minlength=4).max() / len(pr))})
    print(f"  {name:16s} macro-F1 {rob[-1]['macro_f1']:.3f} acc {rob[-1]['accuracy']:.3f} Normal recall {rec[0]:.2f}")
rob = pd.DataFrame(rob)
rob["f1_drop_vs_clean"] = rob.macro_f1.iloc[0] - rob.macro_f1
rob.to_csv(T / "abl_robustness.csv", index=False)
assert abs(rob.macro_f1.iloc[0] - macro_f1(yte, pt)) < 1e-9, "clean re-extraction must reproduce the cached test result"
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].bar(rob.perturbation, rob.macro_f1, color=["#4C78A8"] + ["#E45756"] * (len(rob) - 1), edgecolor="k")
ax[0].set(title="Test macro-F1 under image perturbations", ylabel="macro-F1", ylim=(0, 0.8))
ax[0].tick_params(axis="x", rotation=35)
for i, v in enumerate(rob.macro_f1):
    ax[0].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
rec_cols = [f"recall_{c}" for c in CLASSES]
rob.set_index("perturbation")[rec_cols].plot.bar(ax=ax[1], color=list(CPAL.values()), edgecolor="k")
ax[1].set(title="Per-class test recall under perturbations", ylabel="recall", xlabel="", ylim=(0, 1))
ax[1].tick_params(axis="x", rotation=35)
ax[1].legend([c.replace("recall_", "") for c in rec_cols], fontsize=8)
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "abl_03_robustness.png", dpi=config.FIG_DPI, bbox_inches="tight")
plt.close(fig)

# ================================================================ 4. error analysis: 12 most confident wrong predictions
print("\n[4] Error analysis")
conf = pp.max(1)
wrong = np.where(pt != yte)[0]
top = wrong[np.argsort(-conf[wrong])][:12]
err = pd.DataFrame({"image_id": ids_te[top], "true": [CLASSES[i] for i in yte[top]], "predicted": [CLASSES[i] for i in pt[top]],
                    "confidence": conf[top], "p_true_class": pp[top, yte[top]]})
err["involves_Normal"] = (err.true == "Normal") | (err.predicted == "Normal")
err.to_csv(T / "abl_top_errors.csv", index=False)
print(err.round(3).to_string(index=False))
fig, axes = plt.subplots(3, 4, figsize=(15, 11.5))
for a, r in zip(axes.ravel(), err.itertuples()):
    img = cv2.cvtColor(cv2.imread(str(config.DATASET_PATH / man.loc[r.image_id, "image_path"])), cv2.COLOR_BGR2RGB)
    a.imshow(cv2.resize(img, (420, 420), interpolation=cv2.INTER_AREA))
    a.set_title(f"{r.image_id}\ntrue: {r.true} -> pred: {r.predicted} ({r.confidence:.2f})", fontsize=9, color="#C0392B" if r.involves_Normal else "k")
    a.axis("off")
fig.suptitle("12 most confident wrong test predictions (red title = involves Normal)", fontweight="bold")
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "abl_04_top_errors.png", dpi=config.FIG_DPI, bbox_inches="tight")
plt.close(fig)
n_wrong = len(wrong)
wrong_normal_true = int((yte[wrong] == 0).sum())
normal_to = pd.Series(pt[yte == 0]).map(config.IDX_TO_CLASS).value_counts()

# ================================================================ summary
ab = abl.set_index("config")
best_only = abl[abl.config.str.startswith("only")].sort_values("cv_f1_mean", ascending=False).iloc[0]
worst_wo = abl[abl.config.str.startswith("without")].sort_values("cv_f1_mean").iloc[0]
sh_n = shift[(shift["class"] == "Normal") & (shift.model == "LogReg")].set_index("feature_set").train_vs_test_auc
sh_o = shift[(shift["class"] != "Normal") & (shift.model == "LogReg")].groupby("feature_set").train_vs_test_auc.max()
sh_rf = shift[(shift["class"] == "Normal") & (shift.model == "RandomForest")].set_index("feature_set").train_vs_test_auc
gpn = grp[(grp.feature_set == "handcrafted_norm") & (grp.model == "LogReg")].set_index("group").train_vs_test_auc.sort_values(ascending=False)
gph = grp[(grp.feature_set == "handcrafted") & (grp.model == "LogReg")].set_index("group").train_vs_test_auc.sort_values(ascending=False)
rb = rob.set_index("perturbation")
blur_note = (f"Blur raises Normal recall from {rb.loc['clean', 'recall_Normal']:.2f} to {rb.loc['blur sigma=2', 'recall_Normal']:.2f} (sigma 2) and {rb.loc['blur sigma=4', 'recall_Normal']:.2f} (sigma 4) while lowering macro-F1, "
             f"consistent with the model having learned 'blurrier image = Normal' from the blurrier train Normal images (blur score 4.1 vs 8.4 in test).")
noise_note = (f"Noise destroys the texture features: at sigma=20 macro-F1 is {rb.loc['noise sigma=20', 'macro_f1']:.2f} (chance is 0.25) and {rb.loc['noise sigma=20', 'largest_predicted_class_share']:.0%} of predictions fall into one class; "
              f"the model is not robust to sensor noise and should not be trusted on noisy or heavily compressed images.")
conf_wrong_normal = float(conf[(yte == 0) & (pt != 0)].mean())
nw = int(((yte == 0) & (pt != 0)).sum())
worst_p = rob.iloc[1:].sort_values("f1_drop_vs_clean", ascending=False).iloc[0]
n3 = nonnormal.iloc[1]
n4 = nonnormal.iloc[0]
b_cv, b_te = binary.iloc[0], binary.iloc[1]
top_f = uni[uni.feature_set == "handcrafted_norm"].head(5)

md = f"""# Ablation, source-shift and robustness summary

Final model: **{meta['candidate']}** (soft-voting ensemble of tuned LogReg / SVM / RF pipelines) on **{EXP}** features; CV macro-F1 {meta['cv_f1_macro']:.3f} (nested), test macro-F1 {macro_f1(yte, pt):.3f}.
Ablation refits clones of the final model's hyperparameters. Those were tuned on all features and on this same train data, so the ablation CV scores are not nested (the full-feature CV here, {full['cv_f1_mean']:.3f}, is about {full['cv_f1_mean'] - meta['cv_f1_macro']:.3f} above the nested {meta['cv_f1_macro']:.3f}); compare configurations with each other, not with the nested figure, and subsets are slightly disadvantaged. Test numbers are reported, never used for selection.

## 1. Feature-group ablation (CV / test macro-F1)

""" + abl.assign(CV=lambda d: d.cv_f1_mean.map("{:.3f}".format) + " ± " + d.cv_f1_std.map("{:.3f}".format), test=lambda d: d.test_f1.map("{:.3f}".format))[["config", "n_features", "CV", "test"]].to_markdown(index=False) + f"""

- Strongest single group by CV: **{best_only.config}** (CV {best_only.cv_f1_mean:.3f}); removing a group hurts most for **{worst_wo.config}** (CV {worst_wo.cv_f1_mean:.3f} vs {ab.loc['all (handcrafted_norm)', 'cv_f1_mean']:.3f} with everything).
- Differences between most configurations are within the CV standard deviation (about {abl.cv_f1_std.mean():.3f}); only large drops should be read as real.
- ResNet50 alone: CV {ab.loc['ResNet50 only', 'cv_f1_mean']:.3f}, test {ab.loc['ResNet50 only', 'test_f1']:.3f}.

## 2. Source shift of the Normal class

**Normal excluded.** Scoring the final 4-class model on the {int(nn.sum())} non-Normal test images (predicting Normal counts as an error) gives macro-F1 over the three classes of {n4.macro_f1_3cls:.3f}
(recall AMD {n4['recall_AMD']:.2f}, DR {n4['recall_Diabetic Retinopathy']:.2f}, Glaucoma {n4['recall_Glaucoma']:.2f}; CV recall {n4['cv_recall_AMD']:.2f} / {n4['cv_recall_Diabetic Retinopathy']:.2f} / {n4['cv_recall_Glaucoma']:.2f}).
A 3-class model trained without any Normal images reaches CV macro-F1 {n3.cv_macro_f1_3cls:.3f} and test macro-F1 {n3.macro_f1_3cls:.3f} (test recall {n3['recall_AMD']:.2f} / {n3['recall_Diabetic Retinopathy']:.2f} / {n3['recall_Glaucoma']:.2f}),
so the three disease classes transfer from train to test much better than Normal does.

**Normal-vs-rest** (final model, P(Normal) as score): CV (out-of-fold) AUC {b_cv.auc_normal_vs_rest:.3f}, precision {b_cv.precision:.2f}, recall {b_cv.recall:.2f}, F1 {b_cv.f1:.2f}; test AUC {b_te.auc_normal_vs_rest:.3f}, precision {b_te.precision:.2f}, recall {b_te.recall:.2f}, F1 {b_te.f1:.2f}
(the model predicts Normal for {int(b_te.n_pred_normal)} test images; {int(b_te.n_normal)} are truly Normal). True Normal test images are predicted as: {', '.join(f'{k} {v}' for k, v in normal_to.items())}.

**Train-vs-test classifier (images of a single class; AUC 0.5 = train and test are indistinguishable).** For Normal, a LogReg separates train from test with AUC
{sh_n['handcrafted']:.3f} (handcrafted), {sh_n['handcrafted_norm']:.3f} (handcrafted_norm), {sh_n['resnet']:.3f} (ResNet50); RandomForest {sh_rf['handcrafted']:.3f} / {sh_rf['handcrafted_norm']:.3f} / {sh_rf['resnet']:.3f}.
The highest AUC among the other three classes (controls) is {sh_o['handcrafted']:.3f} / {sh_o['handcrafted_norm']:.3f} / {sh_o['resnet']:.3f}. Full table: `abl_train_vs_test_auc.csv`.

**Which feature groups carry the shift** (Normal, LogReg AUC): handcrafted_norm: {', '.join(f'{g} {v:.3f}' for g, v in gpn.items())}; handcrafted: {', '.join(f'{g} {v:.3f}' for g, v in gph.items())}.
Most shifted single features (handcrafted_norm, single-feature AUC of train vs test; below 0.5 means lower in train): {', '.join(f'{r.feature} ({r.auc_train_vs_test:.2f})' for r in top_f.itertuples())}.

## 3. Robustness (test images degraded, features re-extracted, final model unchanged)

""" + rob[["perturbation", "macro_f1", "accuracy", "recall_Normal", "share_predicted_Normal", "f1_drop_vs_clean"]].round(3).to_markdown(index=False) + f"""

Largest drop: **{worst_p.perturbation}** ({worst_p.f1_drop_vs_clean:+.3f} macro-F1). Perturbation definitions: Gaussian blur at 512 px scale; multiplicative brightness (so the black border stays black);
Gaussian noise on the retina only. The model was never trained with such perturbations. {blur_note} {noise_note}

## 4. Error analysis

{n_wrong} of {len(yte)} test images are misclassified; {wrong_normal_true} of those errors are true-Normal images. Of the 12 most confident errors, {int(err.involves_Normal.sum())} involve Normal
(true Normal: {int((err.true == 'Normal').sum())}, predicted Normal: {int((err.predicted == 'Normal').sum())}). See `abl_04_top_errors.png` and `abl_top_errors.csv`.

## What this means for trusting the test numbers

- The test macro-F1 ({macro_f1(yte, pt):.3f}) is an average of two different things: the three disease classes, whose train and test images are far more alike (control AUC at most {sh_o['handcrafted_norm']:.2f}; Glaucoma shows a mild shift, AMD and DR almost none) and whose test performance is close to CV, and Normal,
  whose train and test images are strongly separable (AUC {sh_n['handcrafted_norm']:.2f}) and whose test recall ({rec_all[0]:.2f}) is far below CV recall ({rec4_cv[0]:.2f}). Test numbers for the disease classes can be trusted at the level of the test-set noise (n=50 per class, about ±0.07 on a recall).
- The Normal test figures measure how well a model trained on the train Normal source transfers to a different source; they are not an estimate of how well the model detects Normal fundi in general, and CV cannot reveal them.
  Both CV and test numbers are therefore biased for Normal, in opposite directions: CV is optimistic for images from the test source, while the test (one source, 50 images) may be pessimistic or optimistic for any other source.
- The deployment risk is concrete: {nw} of {int((yte == 0).sum())} test Normal images ({nw / (yte == 0).sum():.0%}) are labelled as a disease (most often {normal_to.drop('Normal').index[0]}), with mean confidence {conf_wrong_normal:.2f} on those wrong predictions. Treat Normal predictions from this model as unreliable until Normal training images from more than one source are available.
- Honest headline numbers to report: CV macro-F1 {meta['cv_f1_macro']:.3f} (train source), test macro-F1 {macro_f1(yte, pt):.3f}, test macro-F1 over the three disease classes only {n4.macro_f1_3cls:.3f}, Normal test recall {rec_all[0]:.2f}.
"""
(config.ROOT / "results" / "ablation_summary.md").write_text(md)
print("\n" + md)
