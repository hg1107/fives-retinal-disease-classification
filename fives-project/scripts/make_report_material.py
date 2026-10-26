"""Assemble results/final_report_material.md from the saved results (no model is fitted here).

Run after scripts/run_all.py. Every number comes from results/tables/*.csv, results/*.json|md, data/processed/manifest.csv
or the code constants, so the file regenerates with the pipeline.
"""
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
import xgboost  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import features as F  # noqa: E402
from src.models import baseline_models  # noqa: E402

R = config.ROOT / "results"
T = config.TABLES_DIR
FIG = "results/figures"
CL = config.CLASS_NAMES


def tab(name):
    return pd.read_csv(T / name)


def md(df, digits=3, index=False):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    return d.to_markdown(index=index)


def text(path):
    return (config.ROOT / path).read_text().strip()


def downgrade(s, n=2):
    """Push embedded markdown headings down n levels so they nest under the report sections."""
    return "\n".join(("#" * n + l) if l.startswith("#") else l for l in s.splitlines())


def figs(items):
    out = ["| figure | what it shows |", "|---|---|"]
    for f, cap in items:
        assert (config.FIGURES_DIR / f).exists(), f"missing figure {f}"
        out.append(f"| `{FIG}/{f}` | {cap} |")
    return "\n".join(out)


man = pd.read_csv(config.DATA_DIR / "manifest.csv")
clog = tab("cleaning_log.csv")
bcv, bte, bpc = tab("baseline_cv.csv"), tab("baseline_test.csv"), tab("baseline_per_class.csv")
ba = tab("tune_before_after.csv")
gains = tab("tune_gain_vs_noise.csv")
hp = json.loads((R / "best_hyperparameters.json").read_text())
meta = json.loads((config.MODELS_DIR / "best_model_meta.json").read_text())
abl = tab("abl_feature_groups.csv")
final = ba[ba.candidate == hp["final_model"]["candidate"]].iloc[0]
EXPS = ["handcrafted", "handcrafted_norm", "resnet", "both"]
ct = pd.crosstab(man.label_name, man.official_split).reindex(CL)
bestexp = bcv.loc[bcv.f1_macro_mean.idxmax()]
feat_rank, drop_list, pairs = tab("feat_anova_ranking.csv"), tab("feat_drop_list.csv"), tab("feat_high_corr_pairs.csv")
outl = tab("feat_outlier_summary.csv")
vs = tab("feat_vessel_sanity.csv")
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
L = []
w = L.append

w("# Final report material\n")
w("Everything below is generated from the saved results of scripts 01-09 (`python scripts/make_report_material.py`). Paths are relative to the project root. "
  "Sections follow the report outline: dataset, EDA, preprocessing, algorithms, experimental setup, evaluation, tuning, results and discussion, conclusion.\n")
w("## 0. Headline numbers\n")
w(md(pd.DataFrame([
    ["Final model", f"{hp['final_model']['candidate']} (soft voting of tuned LogReg, SVM-RBF, RF), feature set `{meta['feature_set']}` ({len(meta['feature_columns'])} features)"],
    ["CV macro-F1 (nested 5-fold, train)", f"{final.cv_f1_mean:.3f} ± {final.cv_f1_std:.3f}"],
    ["Test accuracy / macro-F1 / ROC-AUC (OvR)", f"{final.test_accuracy:.3f} / {final.test_f1_macro:.3f} / {final.test_roc_auc_ovr:.3f}"],
    ["Test recall per class", ", ".join(f"{c} {final[f'test_recall_{c}']:.2f}" for c in CL)],
    ["Test macro-F1 over the 3 disease classes only", f"{tab('abl_normal_excluded.csv').iloc[0].macro_f1_3cls:.3f}"],
    ["Best baseline by CV (any experiment)", f"{bestexp.model} on {bestexp.experiment}: CV macro-F1 {bestexp.f1_macro_mean:.3f}"],
    ["Normal train-vs-test separability (LogReg AUC)", ", ".join(f"{k} {v:.3f}" for k, v in tab("abl_train_vs_test_auc.csv").query('`class` == "Normal" and model == "LogReg"').set_index("feature_set").train_vs_test_auc.items())],
], columns=["item", "value"])))

# ------------------------------------------------------------------ 1 dataset
w("\n## 1. Dataset description\n")
w("- **Source:** FIVES, a fundus image dataset for vessel segmentation, hosted on Figshare (DOI 10.6084/m9.figshare.19688169). Citation: Jin K., et al. Sci Data 9, 475 (2022). "
  "It is NOT a Kaggle dataset.")
w("- **Content:** 800 colour fundus photographs, 2048 x 2048 RGB PNG, with a binary vessel-segmentation mask for each image (`Ground truth/`). The class is the last letter of the filename: "
  "A = AMD, D = Diabetic Retinopathy, G = Glaucoma, N = Normal; 200 images per class. Official split: 600 train (150 per class) / 200 test (50 per class). "
  "A `Quality Assessment.xlsx` and `Thumbs.db` files are also present and ignored.")
w("- **Use of masks:** only for the EDA vessel-density plot and the sanity check of our own vessel features; never as model input.")
w("- **Numeric ids restart in each split** (train/1_A and test/1_A are different images), so images are keyed as `<split>_<id>_<letter>`.\n")
w("### Audit and cleaning\n")
w(downgrade(text("results/audit_summary.md"), 3))
w("\nCleaning log (`results/tables/cleaning_log.csv`):\n")
w(md(clog))
w(f"\nManifest after cleaning (`data/processed/manifest.csv`): {len(man)} images. Class counts:\n")
w(md(ct.reset_index().rename(columns={"label_name": "class"})))
w("\nAll removed images are AMD (6 exact duplicates inside one split plus 1 train image that near-duplicates a test image). Test was never touched except for one exact in-test duplicate.\n")
w(figs([("eda_01_class_distribution.png", "class counts overall and by split"), ("eda_02_sample_grid.png", "4 random images per class with vessel masks"),
        ("eda_03_image_size.png", "pixel dimensions and file sizes")]))

# ------------------------------------------------------------------ 2 EDA
w("\n## 2. Exploratory data analysis\n")
w(text("results/eda_insights.md").replace("# EDA insights", "Key observations (`results/eda_insights.md`):"))
w("\nDescriptive statistics by class: `results/tables/eda_summary.csv`.\n")
w(figs([("eda_04_quality_by_class.png", "blur / brightness / contrast by class (violin + box)"), ("eda_05_rgb_histograms.png", "mean RGB histograms per class (FOV only)"),
        ("eda_06_mean_image.png", "mean image per class"), ("eda_07_vessel_density.png", "vessel density from ground-truth masks, by class"),
        ("eda_08_quality_correlation.png", "Spearman correlation of quality metrics per class")]))

# ------------------------------------------------------------------ 3 preprocessing
w("\n## 3. Preprocessing and feature engineering\n")
w("### Image preprocessing (`src/features.py: preprocess`)\n")
w(f"1. Resize longest side to {F.SIZE} px (area interpolation), pad to square with black (images are already square).\n"
  f"2. Field-of-view (FOV) detection: blurred grayscale > {F.FOV_THRESH}, 31x31 morphological opening (removes the notch), largest component, holes filled. Crop to its bounding box, pad to square, resize to {F.SIZE}.\n"
  f"3. Circular FOV mask (radius {F.CIRCLE_FRAC} x half width) intersected with the detected FOV, eroded by {F.ERODE_PX} px. Everything outside is set to 0, which also removes the corner overlay glyph.\n"
  f"4. CLAHE (clip limit 2.0, 8x8 tiles) on the green channel.\n"
  f"5. **Normalised variant** (`handcrafted_norm`, used by the final model): green channel divided by its masked Gaussian blur (sigma = {F.ILLUM_SIGMA:.0f} px = width/8), then each of R, G, B standardised inside the FOV "
  f"(mean {F.NORM_MEAN:.0f}, std {F.NORM_STD:.0f}, clipped to uint8), then CLAHE on the normalised green. Applied to the colour and texture groups only.\n")
w("### Feature groups (116 handcrafted features)\n")
grp = feat_rank.groupby("group").agg(n_features=("feature", "count"), mean_anova_F=("F", "mean"), best_feature=("feature", "first")).reset_index()
grp["description"] = grp.group.map({
    "color": "mean, std, skew, kurtosis, percentiles 5/25/50/75/95 of R,G,B,H,S,V,L,a,b inside the FOV",
    "tex": f"GLCM (contrast, homogeneity, energy, correlation; distances {list(F.GLCM_DISTANCES)} x 4 angles, averaged, {F.GLCM_LEVELS} levels, green channel) + uniform LBP histogram (P={F.LBP_P}, R={F.LBP_R}, 10 bins)",
    "vessel": f"image-derived: multi-scale Frangi (sigmas {list(F.FRANGI_SIGMAS)}) on green, thresholded at {F.VESSEL_REL_THRESH} x p99, small objects < {F.MIN_VESSEL_PX} px removed; density, thickness mean/std, skeleton length/density, branch/end points, segments, tortuosity proxy, mean response",
    "disc": "bright-region features: smoothed gray >= 0.9 / 0.8 x p99 area ratios, number of components, largest component centroid (x, y), distance to centre, eccentricity, solidity, peak-to-mean"})
w(md(grp[["group", "n_features", "description", "mean_anova_F", "best_feature"]], 1))
w("\n**Deep embeddings:** ResNet50 (ImageNet weights), global-average-pooled 2048-d, run in fp16 on the GPU (batch 32) on the preprocessed (FOV-cropped, masked, no CLAHE) image resized to 224 x 224. "
  "The preprocessed image is used instead of the raw frame because the raw frame carries the corner glyph that marks the image source.\n")
w("### Feature analysis (train split only)\n")
sk = outl
w(f"- Missing values: 0; inf: 0. Constant / near-constant features (>= 95% one value): {len(drop_list)}.\n"
  f"- Feature pairs with |r| > 0.95: {len(pairs)}; a greedy correlation filter at 0.95 keeps 75 of the 116 features (many exact colour-space duplicates, e.g. RGB_R vs HSV_V).\n"
  f"- Skew: {int(sk.abs_skew_gt1.sum())}/{len(sk)} features have |skew| > 1; median per-feature IQR-outlier fraction {sk.outlier_frac.median():.1%}, {int((sk.outlier_frac > 0.05).sum())} features above 5%. "
  "This motivates `RobustScaler` as the default scaler.\n"
  f"- Top-ranked features by ANOVA F are almost all colour (28 of the top 30).\n")
w("Top 12 features by ANOVA F (train):\n")
w(md(feat_rank.head(12)[["feature", "group", "F"]], 1))
w("\nVessel features vs ground-truth mask vessel density (train, sanity check only):\n")
w(md(vs))
w("\n### sklearn preprocessing pipeline (`src/preprocess.py: build_pipeline`)\n")
w("`SimpleImputer(median)` -> `RobustScaler` (or `StandardScaler`) -> `VarianceThreshold(0)` -> `CorrelationFilter(0.95)` -> `SelectKBest(f_classif, k=30)`; "
  "all fitted inside the pipeline on the training part of each CV fold only. No SMOTE: classes are balanced (150/144/150/150 per class in train).\n")
w(figs([("feat_01_missing_constant.png", "missing / constant / near-constant analysis"), ("feat_02_top12_distributions.png", "distributions of the top 12 features by class"),
        ("feat_03_correlation_clustered.png", "clustered correlation heatmap with group strip"), ("feat_04_outliers_skew.png", "IQR outliers and skewness"),
        ("feat_05_pca_tsne.png", "PCA and t-SNE of the handcrafted features"), ("feat_06_vessel_vs_groundtruth.png", "image-derived vessel features vs ground-truth density")]))

# ------------------------------------------------------------------ 4 algorithms
w("\n## 4. Algorithms\n")
w("Six classifiers, each placed after the preprocessing pipeline in one `sklearn.pipeline.Pipeline` (`src/models.py`), default hyperparameters except those needed to run:\n")
rows = []
for n, m in baseline_models().items():
    keep = {k: v for k, v in m.get_params().items() if k in ("max_iter", "n_estimators", "kernel", "probability", "n_neighbors", "tree_method", "device", "random_state", "C", "gamma")}
    rows.append({"model": n, "class": type(m).__name__, "non-default / key settings": str(keep)})
w(md(pd.DataFrame(rows)))
w("\nXGBoost uses `tree_method=\"hist\"` on the GPU (CUDA).\n")
w(f"**Tuning:** RandomizedSearchCV, {hp['n_iter']} draws, 5-fold stratified CV, scoring macro-F1, over the classifier hyperparameters plus `select__k` {{10,15,20,30,40,60,all}}, the scaler {{Robust, Standard}} and the correlation threshold {{0.85,...,0.995}}.\n")
w("**Ensembles** (built from the 3 tuned models): soft `VotingClassifier`, `StackingClassifier` (logistic-regression meta-learner, 5-fold internal CV, predicted probabilities) and `BaggingClassifier` (30 bootstrap replicas) of the best tuned base model.\n")
w("Tuned hyperparameters (`results/best_hyperparameters.json`):\n")
w("```json\n" + json.dumps(hp, indent=2) + "\n```\n")

# ------------------------------------------------------------------ 5 setup
w("## 5. Experimental setup\n")
w(f"- **Split:** official train ({int((man.official_split == 'train').sum())}) / test ({int((man.official_split == 'test').sum())}); the test split is used once, for final reporting; no choice (features, hyperparameters, model) depends on it.\n"
  f"- **Cross-validation:** 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on the train split, identical folds for every model/experiment. Tuned models and ensembles are scored with nested CV (searches re-run inside each outer training fold).\n"
  "- **Experiments (feature sets):** `handcrafted` (116), `handcrafted_norm` (116), `resnet` (2048), `both` (handcrafted + resnet = 2164).\n"
  "- **Metrics:** accuracy, macro precision / recall / F1, ROC-AUC one-vs-rest (macro); per-class precision / recall / F1; confusion matrices. Regression metrics (MAE, MSE, RMSE, R²) do not apply to this classification task.\n"
  "- **Leakage control:** imputer, scaler, variance/correlation filters and SelectKBest are inside the Pipeline; duplicates removed before splitting into folds; train/test near-duplicate removed from train.\n"
  f"- **Seed:** 42 everywhere. **Hardware:** {gpu}, driver CUDA 13.0 (nvidia-smi); **software:** Python {platform.python_version()}, scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}, torch {torch.__version__}, numpy {np.__version__}, pandas {pd.__version__}; Linux (Arch).\n")
w("Fit and predict times of the baselines (full train fit, test predict; seconds):\n")
w(md(bte[["experiment", "model", "fit_time_s", "predict_time_s"]], 3))
w("\n" + figs([("cmp_08_time_vs_f1.png", "time vs macro-F1")]))

# ------------------------------------------------------------------ 6 evaluation
w("\n## 6. Evaluation of the baseline models\n")
w("All 24 combinations, CV (mean ± std over 5 folds) and test side by side (`results/tables/cmp_comparison_table.csv`):\n")
tbl = tab("cmp_comparison_table.csv")
show = pd.DataFrame({"experiment": tbl.experiment, "model": tbl.model,
                     "CV acc": tbl.cv_accuracy_mean.map("{:.3f}".format) + " ± " + tbl.cv_accuracy_std.map("{:.3f}".format),
                     "CV F1": tbl.cv_f1_macro_mean.map("{:.3f}".format) + " ± " + tbl.cv_f1_macro_std.map("{:.3f}".format),
                     "CV AUC": tbl.cv_roc_auc_ovr_mean.map("{:.3f}".format), "test acc": tbl.test_accuracy.map("{:.3f}".format),
                     "test P": tbl.test_precision_macro.map("{:.3f}".format), "test R": tbl.test_recall_macro.map("{:.3f}".format),
                     "test F1": tbl.test_f1_macro.map("{:.3f}".format), "test AUC": tbl.test_roc_auc_ovr.map("{:.3f}".format),
                     "gap F1 (CV-test)": tbl.gap_f1_cv_minus_test.map("{:+.3f}".format)})
w(show.to_markdown(index=False))
w("\nPer-class test recall and CV recall of the baselines (`results/tables/baseline_per_class.csv`), Normal rows:\n")
w(md(bpc[bpc["class"] == "Normal"][["experiment", "model", "test_precision", "test_recall", "test_f1", "cv_recall_mean"]]))
w("\nEffect of per-image normalisation (handcrafted vs handcrafted_norm), `cmp_norm_effect.csv`:\n")
w(md(tab("cmp_norm_effect.csv")))
w("\nMcNemar test, top-2 models by CV F1 (`cmp_mcnemar.csv`):\n")
w(md(tab("cmp_mcnemar.csv"), 4))
w("\nMost confused class pairs (`cmp_confused_pairs.csv`):\n")
w(md(tab("cmp_confused_pairs.csv")))
w("\n" + downgrade(text("results/comparison_summary.md"), 2) + "\n")
w(figs([("cmp_01_comparison_table.png", "comparison table (CV vs test, best highlighted)"), ("cmp_02_test_metrics_bars.png", "test accuracy / precision / recall / F1 by model, per experiment"),
        ("cmp_03_cv_test_gap.png", "CV-minus-test gap"), ("cmp_04_per_class_recall.png", "per-class test recall heatmaps (Normal outlined)"),
        ("cmp_05_confusion_matrices.png", "confusion matrices of the best models"), ("cmp_06_roc_curves.png", "one-vs-rest ROC curves"),
        ("cmp_07_cv_fold_boxplots.png", "fold-wise CV scores"), ("cmp_10_norm_effect.png", "effect of per-image normalisation")]))

# ------------------------------------------------------------------ 7 tuning
w("\n## 7. Hyperparameter tuning and ensembling\n")
w(f"Experiment `{hp['experiment']}`, models tuned: {', '.join(hp['models'])}. CV values for tuned models and ensembles are nested CV.\n")
cols = ["stage", "candidate", "cv_f1_mean", "cv_f1_std", "test_f1_macro", "test_accuracy", "test_roc_auc_ovr"] + [f"test_recall_{c}" for c in CL] + ["cv_recall_Normal"]
w(md(ba[cols]))
w("\nGain over the reference (baseline) against CV noise and the test bootstrap CI (`tune_gain_vs_noise.csv`):\n")
w(md(gains[["candidate", "reference", "cv_gain_f1", "reference_cv_std", "gain_over_ref_std", "folds_improved", "test_gain_f1", "test_gain_ci_low", "test_gain_ci_high", "cv_verdict", "test_verdict"]]))
w("\nTop 15 permutation-importance features of the final model (macro-F1 drop, train CV folds):\n")
w(md(tab("tune_permutation_importance.csv").head(15), 4))
w("\nLearning curve (`tune_learning_curve.csv`):\n")
w(md(tab("tune_learning_curve.csv")))
w("\n" + downgrade(text("results/tuning_summary.md"), 2) + "\n")
w(figs([("tune_01_before_after.png", "before vs after tuning/ensembling, CV vs test, macro-F1 and Normal recall"), ("tune_02_learning_curve.png", "learning curve of the final model"),
        ("tune_03_permutation_importance.png", "permutation importance (train CV folds)")]))

# ------------------------------------------------------------------ 8 results and discussion
w("\n## 8. Results and discussion\n")
w("### Feature-group ablation\n")
w(md(abl))
w("\n### Normal excluded, Normal-vs-rest\n")
w(md(tab("abl_normal_excluded.csv")))
w("\n" + md(tab("abl_normal_vs_rest.csv")))
w("\n### Train-vs-test classifier AUC (0.5 = no shift)\n")
w(md(tab("abl_train_vs_test_auc.csv")))
w("\nBy feature group (Normal only):\n")
w(md(tab("abl_shift_by_group.csv")))
w("\n### Robustness\n")
w(md(tab("abl_robustness.csv")))
w("\n### Most confident errors\n")
w(md(tab("abl_top_errors.csv")))
w("\n" + downgrade(text("results/ablation_summary.md"), 2) + "\n")
w(figs([("abl_01_feature_groups.png", "feature-group ablation"), ("abl_02_source_shift.png", "train-vs-test AUC per class and per feature group"),
        ("abl_03_robustness.png", "macro-F1 and per-class recall under blur / brightness / noise"), ("abl_04_top_errors.png", "12 most confident wrong test predictions")]))

# ------------------------------------------------------------------ 9 conclusion
w("\n## 9. Conclusion and future scope\n")
w(f"- A classical-ML pipeline (FOV preprocessing, per-image normalisation, 116 handcrafted features, tuned LogReg/SVM/RF soft-voting ensemble) reaches CV macro-F1 {final.cv_f1_mean:.3f} and test macro-F1 {final.test_f1_macro:.3f} (accuracy {final.test_accuracy:.3f}) on 4 classes (chance 0.25).")
w(f"- Tuning and ensembling gave gains of at most {gains.cv_gain_f1.max():+.3f} macro-F1 in nested CV, below one CV standard deviation; the model ranking among the top candidates is not statistically meaningful (McNemar p = {tab('cmp_mcnemar.csv').p_exact.iloc[0]:.2f}).")
w("- The three disease classes generalise from train to test (test recall in line with CV); Normal does not, because Normal training and test images come from different sources (train-vs-test AUC about 0.96 for Normal, at most 0.65 for the others). "
  "Reported test performance mixes these two situations and must be read per class.")
w("- Colour features carry most of the signal, and part of it is acquisition style rather than pathology; the model is not robust to blur or noise and over-uses 'blurrier = Normal'.")
w("- **Future scope:** (1) Normal training images from several sources/devices, and a multi-centre test set; (2) fine-tuning a CNN on GPU with strong colour/blur augmentation and the FOV mask; "
  "(3) a trained vessel segmenter (the FIVES masks) in place of the Frangi proxy (r = 0.56 with ground truth) so vessel features can be accurate and used at inference; (4) probability calibration and an abstain option; "
  "(5) explainability (Grad-CAM / SHAP) and clinician review of the confident errors; (6) domain-adaptation or source-invariant features.\n")
w("## Appendix: result files\n")
w("Tables: `results/tables/` (audit_*, eda_summary, feat_*, baseline_*, cmp_*, tune_*, abl_*, cleaning_log). Predictions: `results/predictions/` (baselines `<experiment>__<model>.csv` with out-of-fold and test rows; tuned/ensembles `tune__<candidate>.csv`). "
  "Model: `models/best_model.joblib`, `models/classes.json`, `models/best_model_meta.json`. Features: `data/processed/features_*.csv|npy`, `manifest.csv`.\n")
(R / "final_report_material.md").write_text("\n".join(L))
print(f"Wrote {R / 'final_report_material.md'} ({len(chr(10).join(L)) / 1024:.0f} KB)")
