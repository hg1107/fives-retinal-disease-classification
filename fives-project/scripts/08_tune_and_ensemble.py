"""08 - Hyperparameter tuning and ensembles. ALL selection uses CV on the train split only; test is used once, for reporting.

Design
  * Experiment = the feature set with the best baseline CV macro-F1; models = the 3 best model types within it (CV macro-F1).
  * Tuning: RandomizedSearchCV (40 draws, 5-fold stratified, macro-F1) over the model's hyperparameters plus
    select__k, the scaler (Robust/Standard) and the correlation-filter threshold. (The optional follow-up grid search is
    skipped: with 594 rows its gain would be below CV noise.)
  * Ensembles from the 3 tuned models: soft Voting, Stacking (logistic-regression meta-learner, internal 5-fold CV) and
    Bagging of the best tuned base model (best = highest inner-CV score).
  * Honest CV estimate for everything tuned: NESTED CV. In each of the 5 outer folds (identical to the baseline folds) the
    3 searches are re-run on the outer-train part only, ensembles are rebuilt from those tuned models and scored on the outer
    validation part. Search `best_score_` alone would be optimistic (best of 40 draws on the same folds).
  * Final model = highest nested-CV (or baseline-CV, for untuned candidates) macro-F1. Test numbers never enter the choice.
  * Gain vs noise: paired fold differences vs baseline (same folds) and a paired bootstrap CI of the test macro-F1 difference.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import loguniform, randint, uniform  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import BaggingClassifier, StackingClassifier, VotingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, learning_curve  # noqa: E402
from sklearn.preprocessing import RobustScaler, StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.models import baseline_models, make_pipeline  # noqa: E402
from src.preprocess import load_dataset  # noqa: E402

warnings.filterwarnings("ignore")  # sklearn FutureWarnings (SVC probability) and xgboost device notices
import os  # noqa: E402

os.environ["PYTHONWARNINGS"] = "ignore"  # propagate to joblib worker processes

SEED = config.SEED
T = config.TABLES_DIR
CLASSES = config.CLASS_NAMES
LABELS = list(range(4))
N_ITER = 40
BOOT = 2000
t_start = time.time()

# ---------------------------------------------------------------- 1. choose experiment and models from baseline CV
base_cv = pd.read_csv(T / "baseline_cv.csv")
base_pc = pd.read_csv(T / "baseline_per_class.csv")
EXP = base_cv.loc[base_cv.f1_macro_mean.idxmax(), "experiment"]
TOP = base_cv[base_cv.experiment == EXP].nlargest(3, "f1_macro_mean").model.tolist()
print(f"Experiment (best baseline CV F1): {EXP}; models to tune: {TOP}")
Xtr, ytr, Xte, yte = load_dataset(EXP)
ids = pd.read_csv(config.DATA_DIR / "features_resnet_ids.csv")
ids_te = ids.image_id[ids.split == "test"].values
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr))  # same folds as the baselines

# ---------------------------------------------------------------- 2. search spaces
PREP = {"select__k": [10, 15, 20, 30, 40, 60, "all"], "scale": [RobustScaler(), StandardScaler()],
        "corr__threshold": [0.85, 0.9, 0.95, 0.98, 0.995]}
SPACES = {
    "LogisticRegression": {"clf__C": loguniform(1e-3, 1e2), "clf__class_weight": [None, "balanced"]},
    "SVM_RBF": {"clf__C": loguniform(1e-1, 1e3), "clf__gamma": loguniform(1e-4, 1e0), "clf__class_weight": [None, "balanced"]},
    "RandomForest": {"clf__n_estimators": [200, 300, 500, 800], "clf__max_depth": [None, 4, 6, 8, 12, 20],
                     "clf__min_samples_leaf": [1, 2, 3, 5], "clf__min_samples_split": [2, 4, 8],
                     "clf__max_features": ["sqrt", "log2", 0.3, 0.5]},
    "XGBoost": {"clf__n_estimators": randint(100, 600), "clf__max_depth": randint(2, 7), "clf__learning_rate": loguniform(0.01, 0.3),
                "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.4, 0.6), "clf__reg_lambda": loguniform(0.1, 20)},
    "KNN": {"clf__n_neighbors": randint(3, 31), "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]},
    "GaussianNB": {"clf__var_smoothing": loguniform(1e-11, 1e-3)},
}


def tunable(name):
    """Fresh classifier for searching. Search runs in parallel processes, so RF/XGB use 1 thread and XGB runs on CPU."""
    clf = clone(baseline_models()[name])
    if name == "RandomForest":
        clf.set_params(n_jobs=1)
    if name == "XGBoost":
        clf.set_params(device="cpu", n_jobs=1)
    return clf


def run_search(name, X, y):
    rs = RandomizedSearchCV(make_pipeline(tunable(name)), {**PREP, **SPACES[name]}, n_iter=N_ITER, scoring="f1_macro",
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1, random_state=SEED, refit=True)
    return rs.fit(X, y)


def ensembles(searches, X=None):
    """Voting / Stacking / Bagging built from clones of the tuned pipelines (unfitted)."""
    est = [(n, clone(s.best_estimator_)) for n, s in searches.items()]
    best_name = max(searches, key=lambda n: searches[n].best_score_)
    return {
        "voting": VotingClassifier(est, voting="soft"),
        "stacking": StackingClassifier(est, final_estimator=LogisticRegression(max_iter=5000, random_state=SEED), stack_method="predict_proba",
                                       cv=StratifiedKFold(5, shuffle=True, random_state=SEED)),
        "bagging": BaggingClassifier(estimator=clone(searches[best_name].best_estimator_), n_estimators=30, bootstrap=True,
                                     random_state=SEED, n_jobs=-1),
    }, best_name


def scores(y, pred, proba):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=LABELS, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision_macro": p, "recall_macro": r, "f1_macro": f,
            "roc_auc_ovr": roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=LABELS)}


# ---------------------------------------------------------------- 3. nested CV (honest CV estimate for tuned + ensembles)
print("\nNested CV (5 outer folds x 3 searches x 40 draws x 5 inner folds)...")
nested = []
for k, (a, b) in enumerate(folds):
    t = time.time()
    xa, ya, xb, yb = Xtr.iloc[a], ytr[a], Xtr.iloc[b], ytr[b]
    s = {n: run_search(n, xa, ya) for n in TOP}
    cands = {f"tuned_{n}": s[n].best_estimator_ for n in TOP}
    ens, bag_base = ensembles(s)
    for n, e in ens.items():
        cands[n] = e.fit(xa, ya)
    for n, e in cands.items():
        pr, pp = e.predict(xb), e.predict_proba(xb)
        rec = precision_recall_fscore_support(yb, pr, labels=LABELS, zero_division=0)[1]
        nested.append({"fold": k, "candidate": n, **scores(yb, pr, pp), **{f"recall_{c}": rec[i] for i, c in enumerate(CLASSES)}})
    print(f"  fold {k}: inner-best {{ {', '.join(f'{n}: {s[n].best_score_:.3f}' for n in TOP)} }}, bagging base {bag_base} ({time.time() - t:.0f}s)")
nested = pd.DataFrame(nested)
nested.to_csv(T / "tune_nested_fold_scores.csv", index=False)

# ---------------------------------------------------------------- 4. final searches on the full train split
print("\nFull-train searches...")
final_search = {n: run_search(n, Xtr, ytr) for n in TOP}
for n, s in final_search.items():
    cr = pd.DataFrame(s.cv_results_)
    keep = [c for c in cr.columns if c.startswith("param_") or c in ("mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time")]
    cr[keep].sort_values("rank_test_score").to_csv(T / f"tune_cv_results_{n}.csv", index=False)
    print(f"  {n}: best inner-CV F1 {s.best_score_:.3f} (baseline CV {base_cv[(base_cv.experiment == EXP) & (base_cv.model == n)].f1_macro_mean.iloc[0]:.3f})")
ens_final, bag_base_final = ensembles(final_search)
cand = {f"tuned_{n}": final_search[n].best_estimator_ for n in TOP}
for n, e in ens_final.items():
    cand[n] = e.fit(Xtr, ytr)
for n in TOP:  # untuned baselines refit on full train, for the before/after comparison and as selection candidates
    cand[f"baseline_{n}"] = make_pipeline(clone(baseline_models()[n])).fit(Xtr, ytr)
print(f"Bagging base model (best inner CV): {bag_base_final}")


def jsonable(v):
    if isinstance(v, (RobustScaler, StandardScaler)):
        return type(v).__name__
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


params_json = {"experiment": EXP, "n_iter": N_ITER, "bagging_base_model": bag_base_final,
               "models": {n: {"best_inner_cv_f1": float(s.best_score_), "params": {k: jsonable(v) for k, v in s.best_params_.items()}} for n, s in final_search.items()}}

# ---------------------------------------------------------------- 5. test evaluation (once) + per-fold baseline scores
rows, test_pred = [], {}
base_fold = {}
for n in TOP:
    d = pd.read_csv(config.ROOT / "results" / "predictions" / f"{EXP}__{n}.csv")
    d = d[d.set == "cv_oof"]
    base_fold[n] = pd.DataFrame([{"fold": k, "f1_macro": f1_score(g.y_true, g.y_pred, average="macro"),
                                  **dict(zip([f"recall_{c}" for c in CLASSES], precision_recall_fscore_support(g.y_true, g.y_pred, labels=LABELS, zero_division=0)[1]))}
                                 for k, g in d.groupby("fold")])
for name, est in cand.items():
    pr, pp = est.predict(Xte), est.predict_proba(Xte)
    test_pred[name] = pr
    sc = scores(yte, pr, pp)
    rec = precision_recall_fscore_support(yte, pr, labels=LABELS, zero_division=0)[1]
    stage = "baseline" if name.startswith("baseline_") else "tuned" if name.startswith("tuned_") else "ensemble"
    if stage == "baseline":
        m = name[len("baseline_"):]
        b = base_cv[(base_cv.experiment == EXP) & (base_cv.model == m)].iloc[0]
        pcm = base_pc[(base_pc.experiment == EXP) & (base_pc.model == m)].set_index("class")
        cvf, cvs, cva = b.f1_macro_mean, b.f1_macro_std, b.accuracy_mean
        cvr = {c: pcm.loc[c, "cv_recall_mean"] for c in CLASSES}
        fold_f1 = base_fold[m].f1_macro.values
    else:
        nf = nested[nested.candidate == name]
        cvf, cvs, cva = nf.f1_macro.mean(), nf.f1_macro.std(), nf.accuracy.mean()
        cvr = {c: nf[f"recall_{c}"].mean() for c in CLASSES}
        fold_f1 = nf.f1_macro.values
    rows.append({"stage": stage, "candidate": name, "cv_f1_mean": cvf, "cv_f1_std": cvs, "cv_accuracy_mean": cva,
                 **{f"cv_recall_{c}": v for c, v in cvr.items()},
                 **{f"test_{k}": v for k, v in sc.items()}, **{f"test_recall_{c}": rec[i] for i, c in enumerate(CLASSES)}, "_fold_f1": fold_f1})
    pd.DataFrame({"image_id": ids_te, "y_true": yte, "y_pred": pr, **{f"proba_{i}": pp[:, i] for i in LABELS}}).to_csv(
        config.ROOT / "results" / "predictions" / f"tune__{name}.csv", index=False)
res = pd.DataFrame(rows)

# gain vs reference (tuned -> its own baseline; ensembles -> the best baseline by CV), with fold-paired and bootstrap-paired tests
best_base = max(TOP, key=lambda n: res.set_index("candidate").loc[f"baseline_{n}", "cv_f1_mean"])
rng = np.random.RandomState(SEED)
boot_idx = rng.randint(0, len(yte), size=(BOOT, len(yte)))
fcache = {}


def boot_f1(pred):
    return np.array([f1_score(yte[i], pred[i], average="macro") for i in boot_idx])


gains = []
for r in res[res.stage != "baseline"].itertuples():
    ref = f"baseline_{r.candidate[len('tuned_'):]}" if r.stage == "tuned" else f"baseline_{best_base}"
    rr = res.set_index("candidate").loc[ref]
    diff_folds = r._asdict()["_fold_f1"] if False else res.loc[res.candidate == r.candidate, "_fold_f1"].iloc[0] - res.loc[res.candidate == ref, "_fold_f1"].iloc[0]
    for n in (r.candidate, ref):
        fcache.setdefault(n, boot_f1(test_pred[n]))
    bd = fcache[r.candidate] - fcache[ref]
    pt = r.test_f1_macro - rr.test_f1_macro
    lo, hi = np.percentile(bd, [2.5, 97.5])
    gains.append({"candidate": r.candidate, "reference": ref, "cv_gain_f1": r.cv_f1_mean - rr.cv_f1_mean, "reference_cv_std": rr.cv_f1_std,
                  "gain_over_ref_std": (r.cv_f1_mean - rr.cv_f1_mean) / rr.cv_f1_std, "paired_fold_diff_mean": diff_folds.mean(),
                  "paired_fold_diff_std": diff_folds.std(ddof=1), "folds_improved": int((diff_folds > 0).sum()),
                  "test_gain_f1": pt, "test_gain_ci_low": lo, "test_gain_ci_high": hi,
                  "normal_recall_gain_test": r.test_recall_Normal - rr.test_recall_Normal})
gains = pd.DataFrame(gains)


def verdict(g):
    if g.test_gain_ci_low > 0:
        return "survives on test (CI > 0)"
    if g.test_gain_f1 > 0:
        return "positive on test but CI includes 0"
    return "does not survive on test"


gains["test_verdict"] = gains.apply(verdict, axis=1)
gains["cv_verdict"] = np.where(gains.gain_over_ref_std > 1, "gain > 1 CV std", np.where(gains.cv_gain_f1 > 0, "gain within CV noise", "no CV gain"))
out = res.drop(columns="_fold_f1").merge(gains[["candidate", "reference", "cv_gain_f1", "gain_over_ref_std", "folds_improved", "test_gain_f1", "test_gain_ci_low", "test_gain_ci_high", "cv_verdict", "test_verdict"]],
                                         on="candidate", how="left")
out["stage"] = pd.Categorical(out.stage, ["baseline", "tuned", "ensemble"], ordered=True)
out = out.sort_values(["stage", "candidate"]).reset_index(drop=True)
out.to_csv(T / "tune_before_after.csv", index=False)
gains.to_csv(T / "tune_gain_vs_noise.csv", index=False)

# ---------------------------------------------------------------- 6. final model (by CV only)
final_name = out.loc[out.cv_f1_mean.idxmax(), "candidate"]
final_est = cand[final_name]
print(f"\nFinal model by CV macro-F1: {final_name} (CV F1 {out.cv_f1_mean.max():.3f})")
params_json["final_model"] = {"candidate": final_name, "cv_f1": float(out.cv_f1_mean.max()), "cv_f1_std": float(out.loc[out.candidate == final_name, "cv_f1_std"].iloc[0]),
                              "cv_estimate": "baseline 5-fold CV" if final_name.startswith("baseline_") else "nested 5-fold CV"}
(config.ROOT / "results" / "best_hyperparameters.json").write_text(json.dumps(params_json, indent=2))
config.MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(final_est, config.MODELS_DIR / "best_model.joblib")
(config.MODELS_DIR / "classes.json").write_text(json.dumps({str(i): c for i, c in config.IDX_TO_CLASS.items()}, indent=2))
(config.MODELS_DIR / "best_model_meta.json").write_text(json.dumps({
    "candidate": final_name, "feature_set": EXP, "feature_columns": list(Xtr.columns),
    "feature_extractor": {"handcrafted": "src.features.extract_handcrafted", "handcrafted_norm": "src.features.extract_handcrafted_norm"}.get(EXP, "see scripts/04_extract_features.py"),
    "feature_version": {"handcrafted": "v1", "handcrafted_norm": "norm_v1"}.get(EXP), "input": "DataFrame with feature_columns, in this order",
    "cv_f1_macro": params_json["final_model"]["cv_f1"], "cv_estimate": params_json["final_model"]["cv_estimate"]}, indent=2))

# learning curve (final model's fixed hyperparameters, CV on train only)
sizes, trs, vas = learning_curve(clone(final_est), Xtr, ytr, train_sizes=np.linspace(0.2, 1.0, 6), cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                 scoring="f1_macro", n_jobs=-1, random_state=SEED, shuffle=True)  # shuffle: rows are ordered by class, so un-shuffled subsets lack classes
lc = pd.DataFrame({"train_size": sizes, "train_f1_mean": trs.mean(1), "train_f1_std": trs.std(1), "cv_f1_mean": vas.mean(1), "cv_f1_std": vas.std(1)})
lc.to_csv(T / "tune_learning_curve.csv", index=False)

# permutation importance computed on the validation part of each TRAIN CV fold (never on test)
imp = []
for k, (a, b) in enumerate(folds):
    m = clone(final_est).fit(Xtr.iloc[a], ytr[a])
    pi = permutation_importance(m, Xtr.iloc[b], ytr[b], scoring="f1_macro", n_repeats=5, random_state=SEED, n_jobs=-1)
    imp.append(pi.importances_mean)
imp = np.array(imp)
pimp = pd.DataFrame({"feature": Xtr.columns, "importance_mean": imp.mean(0), "importance_std_across_folds": imp.std(0)}).sort_values("importance_mean", ascending=False)
pimp.to_csv(T / "tune_permutation_importance.csv", index=False)

# ---------------------------------------------------------------- 7. figures
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({"savefig.dpi": config.FIG_DPI, "axes.titleweight": "bold"})
STAGE_C = {"baseline": "#9E9E9E", "tuned": "#4C78A8", "ensemble": "#F2A93B"}
order = out.candidate.tolist()
fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), sharey=False)
x = np.arange(len(out))
for a, (cvc, tec, ttl) in zip(ax, [("cv_f1_mean", "test_f1_macro", "Macro-F1"), ("cv_recall_Normal", "test_recall_Normal", "Normal recall")]):
    a.bar(x - 0.2, out[cvc], 0.4, color=[STAGE_C[s] for s in out.stage], edgecolor="k", label="CV (train)")
    if cvc == "cv_f1_mean":
        a.errorbar(x - 0.2, out[cvc], yerr=out.cv_f1_std, fmt="none", ecolor="k", capsize=3)
    a.bar(x + 0.2, out[tec], 0.4, color=[STAGE_C[s] for s in out.stage], edgecolor="k", hatch="//", alpha=0.7, label="test")
    a.set_xticks(x, [c.replace("baseline_", "base: ").replace("tuned_", "tuned: ") for c in order], rotation=40, ha="right")
    a.set(title=f"{ttl}: CV (solid) vs test (hatched)", ylabel=ttl, ylim=(0, 1))
handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="k") for c in STAGE_C.values()]
ax[0].legend(handles, STAGE_C, title="stage", loc="lower left")
fig.suptitle(f"Before vs after tuning/ensembling ({EXP}); tuned/ensemble CV = nested CV", fontweight="bold")
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "tune_01_before_after.png", dpi=config.FIG_DPI, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for col, lab, c in [("train_f1", "train", "#E45756"), ("cv_f1", "CV (validation)", "#4C78A8")]:
    ax.plot(lc.train_size, lc[f"{col}_mean"], "o-", color=c, label=lab)
    ax.fill_between(lc.train_size, lc[f"{col}_mean"] - lc[f"{col}_std"], lc[f"{col}_mean"] + lc[f"{col}_std"], color=c, alpha=0.15)
ax.set(title=f"Learning curve: {final_name}", xlabel="training images", ylabel="macro-F1", ylim=(0, 1.02))
ax.legend()
fig.savefig(config.FIGURES_DIR / "tune_02_learning_curve.png", dpi=config.FIG_DPI, bbox_inches="tight")
plt.close(fig)

top = pimp.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top.feature, top.importance_mean, xerr=top.importance_std_across_folds, color="#4C78A8", ecolor="k", capsize=2)
ax.set(title=f"Permutation importance (macro-F1 drop, train CV folds): {final_name}", xlabel="mean decrease in macro-F1 (± std across folds)")
fig.savefig(config.FIGURES_DIR / "tune_03_permutation_importance.png", dpi=config.FIG_DPI, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 8. summary
pd.set_option("display.width", 220)
show = out[["stage", "candidate", "cv_f1_mean", "cv_f1_std", "test_f1_macro", "test_accuracy", "cv_recall_Normal", "test_recall_Normal", "cv_verdict", "test_verdict"]]
print("\n", show.round(3).to_string(index=False))
print("\n", gains.round(3).to_string(index=False))
tuned_g, ens_g = gains[gains.candidate.str.startswith("tuned_")], gains[~gains.candidate.str.startswith("tuned_")]
bg = gains.loc[gains.cv_gain_f1.idxmax()]
fin = out[out.candidate == final_name].iloc[0]
from scipy.stats import mannwhitneyu  # noqa: E402

# descriptive diagnostic only (not used for any choice): do the most important features shift between train and test Normal images?
Xtr_n, Xte_n = Xtr[ytr == 0], Xte[yte == 0]
top10 = pimp.head(10).feature.tolist()
shifted = [f for f in top10 if mannwhitneyu(Xtr_n[f], Xte_n[f]).pvalue < 1e-3]
top_perm = ", ".join(f"{r.feature} ({r.importance_mean:.3f})" for r in pimp.head(5).itertuples())
ref_base = out[out.candidate == f"baseline_{best_base}"].iloc[0]


def sentence(g):
    return (f"{g.candidate}: CV F1 {g.cv_gain_f1:+.3f} ({g.gain_over_ref_std:+.2f} reference CV std; {g.folds_improved}/5 folds improved), "
            f"test F1 {g.test_gain_f1:+.3f} (95% bootstrap CI {g.test_gain_ci_low:+.3f} to {g.test_gain_ci_high:+.3f}) -> {g.cv_verdict}; {g.test_verdict}")


md = f"""# Tuning and ensembling summary

Experiment: **{EXP}** (best baseline CV macro-F1). Tuned models (top 3 by baseline CV F1 in that experiment): {', '.join(TOP)}.
Each was tuned with RandomizedSearchCV ({N_ITER} draws, 5-fold stratified CV, macro-F1) over its hyperparameters, `select__k`, the scaler and the
correlation-filter threshold. Ensembles (soft voting, stacking with a logistic-regression meta-learner, bagging of {bag_base_final}) were built from the tuned models.
CV for every tuned model and ensemble is **nested CV** (searches are re-run inside each outer training fold), so it is not inflated by the search;
baseline CV numbers are the plain 5-fold CV from script 06 on the same folds. Test was used once at the end.

## Did tuning help?

""" + "\n".join("- " + sentence(g) for g in tuned_g.itertuples()) + f"""

Mean CV gain of tuning over the three baselines: {tuned_g.cv_gain_f1.mean():+.3f} macro-F1 (mean reference CV std {tuned_g.reference_cv_std.mean():.3f}); mean test gain {tuned_g.test_gain_f1.mean():+.3f}.

## Did ensembling help?

Reference for ensembles = best baseline by CV ({best_base}, CV F1 {ref_base.cv_f1_mean:.3f} ± {ref_base.cv_f1_std:.3f}, test F1 {ref_base.test_f1_macro:.3f}).

""" + "\n".join("- " + sentence(g) for g in ens_g.itertuples()) + f"""

## Verdict on noise

Largest CV gain: {bg.candidate} ({bg.cv_gain_f1:+.3f}, {bg.gain_over_ref_std:+.2f} reference CV std). """ + (
    "At least one gain exceeds one CV standard deviation." if (gains.gain_over_ref_std > 1).any() else
    "No gain exceeds one CV standard deviation of the baseline, so every improvement is within CV noise.") + " " + (
    f"{int((gains.test_verdict == 'survives on test (CI > 0)').sum())} of {len(gains)} candidates show a test gain whose bootstrap CI excludes 0."
    ) + (" That result is borderline (lower CI bound close to 0) and is one of six comparisons at 95%, where about 0.3 false positives are expected by chance; it is not backed by a CV gain beyond noise." if (gains.test_verdict == 'survives on test (CI > 0)').any() else "") + f"""

## Final model (selected by CV only)

**{final_name}** with {fin.cv_f1_mean:.3f} ± {fin.cv_f1_std:.3f} CV macro-F1 ({params_json['final_model']['cv_estimate']}). Test (single evaluation): accuracy {fin.test_accuracy:.3f},
macro-F1 {fin.test_f1_macro:.3f}, ROC-AUC {fin.test_roc_auc_ovr:.3f}; per-class test recall: """ + ", ".join(f"{c} {fin[f'test_recall_{c}']:.2f}" for c in CLASSES) + f""".
Saved: `models/best_model.joblib` (fitted estimator; every member includes the full preprocessing Pipeline), `models/classes.json`,
`models/best_model_meta.json` (feature set `{EXP}`, feature columns and extractor `{'src.features.extract_handcrafted_norm' if EXP == 'handcrafted_norm' else 'src.features.extract_handcrafted'}`), `results/best_hyperparameters.json`.
Top permutation-importance features (train CV folds): {top_perm}.

## Caveats

- **Normal source shift.** Normal training images come from a different source than Normal test images (see `comparison_summary.md`). CV cannot see this, so any CV gain
  may not transfer to test: CV Normal recall of the final model is {fin.cv_recall_Normal:.2f} versus {fin.test_recall_Normal:.2f} on test. Selection by CV therefore cannot fix, and may even
  reward, fitting the train-source Normal look.
- The baseline CV score used as reference for LogisticRegression/ensembles is itself the maximum of 24 baseline CV scores, so it is slightly optimistic (winner's curse); the nested estimates of tuned models are not.
- Of the 10 most important features (permutation importance on train CV folds), {len(shifted)} differ significantly between train and test Normal images (p < 0.001): {', '.join(shifted) if shifted else 'none'}. Importance computed on train therefore partly reflects the train-source Normal look.
- Tuned models are chosen from {N_ITER} draws on 594 images; differences between top configurations are small relative to fold-to-fold variation (inner CV std about {pd.concat([pd.DataFrame(s.cv_results_) for s in final_search.values()]).std_test_score.mean():.3f}).
- The test set has 199 images; a one-point change in macro-F1 is a few images. Bootstrap CIs above are wide for that reason.
- The CV selection of the final model compares candidates whose CV estimates differ by less than one standard deviation, so the choice among them is close to arbitrary.
- Bagging a stable linear model or SVM rarely helps: it reduces variance, which these models have little of.
"""
(config.ROOT / "results" / "tuning_summary.md").write_text(md)
print("\n" + md)
print(f"Total time {(time.time() - t_start) / 60:.1f} min")
