"""06 - Baseline models: 5-fold CV on train, then ONE evaluation on the official test split.

4 experiments (feature sets) x 6 models, all with the same preprocessing Pipeline and the same CV folds.
The test split is used only for the final report; no choice (features, thresholds, model) depends on it.
Default preprocessing (build_pipeline defaults: median impute, RobustScaler, variance filter,
correlation filter 0.95, SelectKBest k=30) is identical across models and experiments.
"""
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.models import baseline_models, make_pipeline  # noqa: E402
from src.preprocess import load_dataset  # noqa: E402

warnings.filterwarnings("ignore", message=".*mismatched devices.*")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", message=".*probability. parameter was deprecated.*")  # SVC(probability=True) is requested explicitly
EXPERIMENTS = ["handcrafted", "handcrafted_norm", "resnet", "both"]
CLASSES = config.CLASS_NAMES
LABELS = list(range(config.N_CLASSES))
PRED_DIR = config.ROOT / "results" / "predictions"
CM_DIR = config.TABLES_DIR / "baseline_confusion"
PRED_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)
SEED = config.SEED

ids = pd.read_csv(config.DATA_DIR / "features_resnet_ids.csv")  # same row order as load_dataset
ids_tr = ids.image_id[ids.split == "train"].values
ids_te = ids.image_id[ids.split == "test"].values


def metrics(y, pred, proba):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=LABELS, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision_macro": p, "recall_macro": r, "f1_macro": f,
            "roc_auc_ovr": roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=LABELS)}


cv_rows, test_rows, class_rows = [], [], []
for exp in EXPERIMENTS:
    Xtr, ytr, Xte, yte = load_dataset(exp)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)  # identical folds for every model/experiment
    folds = list(skf.split(Xtr, ytr))
    print(f"\n=== {exp}: train {Xtr.shape}, test {Xte.shape} ===")
    for name, clf in baseline_models().items():
        pipe = make_pipeline(clf)
        # ---- 5-fold CV on train
        fm, frec, ftimes = [], [], []
        oof_pred, oof_proba, oof_fold = np.zeros(len(ytr), int), np.zeros((len(ytr), len(LABELS))), np.zeros(len(ytr), int)
        for k, (a, b) in enumerate(folds):
            m = clone(pipe)
            t = time.perf_counter()
            m.fit(Xtr.iloc[a], ytr[a])
            ftimes.append(time.perf_counter() - t)
            pred, proba = m.predict(Xtr.iloc[b]), m.predict_proba(Xtr.iloc[b])
            fm.append(metrics(ytr[b], pred, proba))
            frec.append(precision_recall_fscore_support(ytr[b], pred, labels=LABELS, zero_division=0)[1])
            oof_pred[b], oof_proba[b], oof_fold[b] = pred, proba, k
        fm = pd.DataFrame(fm)
        cv_rec = np.array(frec)
        cv_rows.append({"experiment": exp, "model": name, **{f"{c}_mean": fm[c].mean() for c in fm}, **{f"{c}_std": fm[c].std() for c in fm},
                        "cv_fit_time_s": np.mean(ftimes)})
        # ---- final fit on full train, ONE evaluation on test
        m = clone(pipe)
        t = time.perf_counter()
        m.fit(Xtr, ytr)
        fit_t = time.perf_counter() - t
        t = time.perf_counter()
        pred = m.predict(Xte)
        pred_t = time.perf_counter() - t
        proba = m.predict_proba(Xte)
        tm = metrics(yte, pred, proba)
        test_rows.append({"experiment": exp, "model": name, **{f"test_{k}": v for k, v in tm.items()},
                          "fit_time_s": fit_t, "predict_time_s": pred_t,
                          "cv_f1_macro": fm.f1_macro.mean(), "gap_f1_cv_minus_test": fm.f1_macro.mean() - tm["f1_macro"],
                          "cv_accuracy": fm.accuracy.mean(), "gap_acc_cv_minus_test": fm.accuracy.mean() - tm["accuracy"]})
        p, r, f, s = precision_recall_fscore_support(yte, pred, labels=LABELS, zero_division=0)
        for i, c in enumerate(CLASSES):
            class_rows.append({"experiment": exp, "model": name, "class": c, "test_precision": p[i], "test_recall": r[i], "test_f1": f[i],
                               "test_support": s[i], "cv_recall_mean": cv_rec[:, i].mean(), "cv_recall_std": cv_rec[:, i].std(ddof=1)})
        pd.DataFrame(confusion_matrix(yte, pred, labels=LABELS), index=[f"true_{c}" for c in CLASSES],
                     columns=[f"pred_{c}" for c in CLASSES]).to_csv(CM_DIR / f"{exp}__{name}.csv")
        out = pd.concat([
            pd.DataFrame({"image_id": ids_tr, "set": "cv_oof", "fold": oof_fold, "y_true": ytr, "y_pred": oof_pred,
                          **{f"proba_{i}": oof_proba[:, i] for i in LABELS}}),
            pd.DataFrame({"image_id": ids_te, "set": "test", "fold": -1, "y_true": yte, "y_pred": pred,
                          **{f"proba_{i}": proba[:, i] for i in LABELS}})])
        out.to_csv(PRED_DIR / f"{exp}__{name}.csv", index=False)
        print(f"{name:19s} CV F1 {fm.f1_macro.mean():.3f}+/-{fm.f1_macro.std():.3f} | test F1 {tm['f1_macro']:.3f} acc {tm['accuracy']:.3f} "
              f"| Normal recall {r[0]:.2f} | fit {fit_t:.2f}s")

cv = pd.DataFrame(cv_rows)
test = pd.DataFrame(test_rows)
per_class = pd.DataFrame(class_rows)
cv.to_csv(config.TABLES_DIR / "baseline_cv.csv", index=False)
test.to_csv(config.TABLES_DIR / "baseline_test.csv", index=False)
per_class.to_csv(config.TABLES_DIR / "baseline_per_class.csv", index=False)
summary = cv[["experiment", "model", "accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std", "roc_auc_ovr_mean"]].merge(
    test[["experiment", "model", "test_accuracy", "test_f1_macro", "test_roc_auc_ovr", "gap_f1_cv_minus_test", "gap_acc_cv_minus_test"]],
    on=["experiment", "model"])
summary.to_csv(config.TABLES_DIR / "baseline_summary.csv", index=False)

# ---------------------------------------------------------------- final report
pd.set_option("display.width", 200)
print("\n\n##### RESULTS (CV mean +/- std | test) #####")
for exp in EXPERIMENTS:
    s = summary[summary.experiment == exp].copy()
    s["CV F1"] = s.f1_macro_mean.map("{:.3f}".format) + " +/- " + s.f1_macro_std.map("{:.3f}".format)
    s["CV acc"] = s.accuracy_mean.map("{:.3f}".format) + " +/- " + s.accuracy_std.map("{:.3f}".format)
    print(f"\n[{exp}]")
    print(s[["model", "CV acc", "CV F1", "test_accuracy", "test_f1_macro", "gap_f1_cv_minus_test"]].round(3).to_string(index=False))

print("\n##### Best model per experiment #####")
for exp in EXPERIMENTS:
    s = summary[summary.experiment == exp]
    a, b = s.loc[s.f1_macro_mean.idxmax()], s.loc[s.test_f1_macro.idxmax()]
    print(f"{exp:17s} by CV F1: {a.model} ({a.f1_macro_mean:.3f}; its test F1 {a.test_f1_macro:.3f}) | by test F1: {b.model} ({b.test_f1_macro:.3f})")

print("\n##### 5 largest CV-to-test macro-F1 gaps #####")
print(summary.nlargest(5, "gap_f1_cv_minus_test")[["experiment", "model", "f1_macro_mean", "test_f1_macro", "gap_f1_cv_minus_test"]].round(3).to_string(index=False))

print("\n##### Normal test recall (CV recall in brackets) #####")
nr = per_class[per_class["class"] == "Normal"].pivot(index="model", columns="experiment", values="test_recall")[EXPERIMENTS]
nc = per_class[per_class["class"] == "Normal"].pivot(index="model", columns="experiment", values="cv_recall_mean")[EXPERIMENTS]
print(pd.DataFrame({e: nr[e].map("{:.2f}".format) + " (" + nc[e].map("{:.2f}".format) + ")" for e in EXPERIMENTS}).to_string())
