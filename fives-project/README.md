# Retinal disease classification on FIVES with classical machine learning
**Live demo:** [minor-retinal-disease-classification.streamlit.app](https://minor-retinal-disease-classification.streamlit.app/)

Four-class classification of colour fundus photographs: **Normal, AMD (age-related macular degeneration), Diabetic Retinopathy (DR), Glaucoma**.
Images are turned into numeric features (colour, texture, vessel and bright-region descriptors, plus ResNet50 embeddings) and classified with
scikit-learn / XGBoost models inside leakage-free pipelines, compared fairly, tuned, ensembled and stress-tested. A single raw image can be classified with
`src/predict.py`, and the Streamlit app in `app/app.py` wraps it.

## Dataset

**FIVES** - *A Fundus Image Dataset for AI-based Vessel Segmentation*, hosted on **Figshare** (DOI [10.6084/m9.figshare.19688169](https://doi.org/10.6084/m9.figshare.19688169)).
Citation: Jin K., et al. Sci Data **9**, 475 (2022).
800 RGB fundus images (2048 x 2048 PNG), 200 per class, with binary vessel masks; the class is the last letter of the file name (A, D, G, N).
Official split: 600 train / 200 test. The dataset is **not** from Kaggle and is not redistributed here; set `DATASET_PATH` in `config.py` to the extracted folder
(`<DATASET_PATH>/{train,test}/{Original,Ground truth}/`). The vessel masks are used only to check our own vessel features, never as model input.

## Pipeline

| step | script | what it does |
|---|---|---|
| 01 | `01_audit_dataset.py` | layout, label parsing, integrity, duplicates (md5 + phash confirmed by pixel correlation), quality stats |
| 02 | `02_build_manifest.py` | cleaned manifest: drops exact duplicates and a train/test near-duplicate (test never reduced except one in-test duplicate), logs every removal |
| 03 | `03_eda.py` | 8 EDA figures + insights |
| 04 | `04_extract_features.py` | FOV crop/mask + CLAHE, 116 handcrafted features (plain and per-image-normalised variants), ResNet50 embeddings (GPU, fp16) |
| 05 | `05_feature_analysis.py` | missing/constant, ANOVA ranking, correlation, outliers, PCA/t-SNE, vessel sanity check |
| 06 | `06_train_baselines.py` | 6 models x 4 feature sets, 5-fold CV then one test evaluation |
| 07 | `07_compare_models.py` | tables/figures, McNemar test, written comparison |
| 08 | `08_tune_and_ensemble.py` | randomized search, nested CV, voting/stacking/bagging, final model saved |
| 09 | `09_ablation.py` | feature-group ablation, source-shift analysis, robustness, error analysis |

Shared code: `config.py` (paths, seed, classes), `src/features.py`, `src/preprocess.py` (sklearn pipeline + `load_dataset`), `src/models.py`, `src/predict.py`.
All fitted preprocessing (imputer, scaler, variance/correlation filters, SelectKBest) lives inside one `Pipeline`, so it is fitted on training folds only. No SMOTE: classes are balanced. Seed 42 everywhere.

## Run (Linux, uv)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt        # torch wheels bundle CUDA; an NVIDIA GPU is used for ResNet50 and XGBoost if present
# edit DATASET_PATH in config.py, then:
python scripts/run_all.py                 # scripts 01-09 in order, stops at the first failure (~13 min on a re-run with feature caches; the first run also extracts features and downloads ResNet50 weights); --start N resumes
python scripts/make_report_material.py    # results/final_report_material.md
```

Predict one image (path or RGB array):

```python
from src.predict import predict_image
label, probs = predict_image("path/to/fundus.png")
```

An NVIDIA GPU is required by the ResNet50 step of `04_extract_features.py` (the script asserts CUDA); everything else runs on CPU too. Tested on Arch Linux, RTX 3050 6 GB, driver 580 / CUDA 13.0.

## Run the app

```bash
source .venv/bin/activate
uv pip install -r requirements.txt          # includes the pinned streamlit
streamlit run app/app.py                    # works from any directory, e.g. streamlit run /path/to/fives-project/app/app.py
# headless / health check:
streamlit run app/app.py --server.headless true   # then: curl http://localhost:8501/_stcore/health  -> ok
```

Pages: **Predict** (upload or pick a sample test image), **Model performance**, **Dataset and EDA**, **About**. Every number shown is read from
`models/` and `results/`. The app needs only `app/`, `src/`, `models/`, `results/`, `config.py` and `requirements.txt`; the dataset is optional
(the "sample image" dropdown appears only if `DATASET_PATH` in `config.py` exists). Inference is CPU-only. Theme: `.streamlit/config.toml`.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub, including `app/`, `src/`, `models/` (`best_model.joblib` is about 7 MB, under GitHub's 100 MB limit), `results/`, `config.py`, `requirements.txt` and `.streamlit/`.
2. On [share.streamlit.io](https://share.streamlit.io) choose **New app**, select the repo and branch, and set the main file path to `app/app.py`. Use Python 3.12.
3. Deploy. Streamlit Cloud installs `app/requirements.txt` (the dependency file next to the main file takes precedence over the root `requirements.txt`). It lists only inference dependencies with the same pins as the root file; scikit-learn must stay equal to the version that pickled `models/best_model.joblib`. Later starts reuse the cached environment.

Notes: the root `requirements.txt` (torch, xgboost, jupyter) is for re-running the training scripts, not for the app. `torch` is imported lazily in `src/predict.py` only if the saved model uses ResNet features; the current model does not. `DATASET_PATH` need not exist on the server; the sample-image dropdown is simply hidden.

## Results

Final model: **soft-voting ensemble of tuned Logistic Regression, SVM-RBF and Random Forest** on the per-image-normalised handcrafted features,
chosen by cross-validation only (the test split was used once for reporting).

| model (features) | CV macro-F1 (train, 5-fold) | test macro-F1 | test accuracy |
|---|---|---|---|
| LogReg (handcrafted), best baseline of its set | 0.677 ± 0.029 | 0.646 | 0.648 |
| LogReg (handcrafted_norm) | 0.710 ± 0.041 | 0.542 | 0.563 |
| LogReg (ResNet50) | 0.657 ± 0.060 | 0.557 | 0.558 |
| SVM-RBF (handcrafted + ResNet50) | 0.687 ± 0.032 | 0.548 | 0.558 |
| Tuned SVM-RBF (handcrafted_norm) | 0.696 ± 0.028 | 0.609 | 0.613 |
| Stacking (handcrafted_norm) | 0.716 ± 0.031 | 0.600 | 0.623 |
| **Voting (handcrafted_norm), final** | **0.721 ± 0.044** | **0.582** | **0.603** |

CV for tuned models and ensembles is nested CV. Final model, test per-class recall: Normal 0.24, AMD 0.67, DR 0.70, Glaucoma 0.80; test ROC-AUC (OvR) 0.866.
Test macro-F1 over the three disease classes only: 0.760. Chance is 0.25. Tuning and ensembling improved CV macro-F1 by at most about 0.02, less than one CV standard deviation, so the ranking
among the top candidates is not statistically meaningful. Full numbers: `results/comparison_summary.md`, `results/tuning_summary.md`, `results/ablation_summary.md`, `results/final_report_material.md`.

## Folder structure

```
fives-project/
  config.py                paths, constants, class map, seed
  app/                     app.py (Streamlit app); .streamlit/config.toml holds the theme
  src/                     features.py, preprocess.py, models.py, predict.py
  scripts/                 01_ ... 09_, run_all.py, make_report_material.py
  notebooks/               fives_analysis.ipynb (executed; re-displays saved results)
  data/processed/          manifest.csv, features_*.csv, features_resnet.npy (+ cache/, git-ignored)
  models/                  best_model.joblib, classes.json, best_model_meta.json
  results/figures/         eda_*, feat_*, cmp_*, tune_*, abl_*  (PNG, 200 dpi)
  results/tables/          audit_*, baseline_*, cmp_*, tune_*, abl_*, cleaning_log (CSV)
  results/predictions/     per-model out-of-fold and test predictions + probabilities
  results/*.md|json        audit, EDA, comparison, tuning, ablation summaries; hyperparameters; report material
```

## Limitations

- **Normal train/test source shift.** Normal training images come from a different source than Normal test images: 67-76 of 116 handcrafted features differ significantly
  (Mann-Whitney p < 0.001), Normal train images are about twice as blurry (blur score 4.1 vs 8.4), and a classifier separates train from test Normal images with AUC about 0.96-0.98
  (the other classes: 0.48-0.65). Per-image normalisation did not remove it. Consequently Normal recall is about 0.8 in CV but 0.24 on test for the final model; CV cannot reveal this and test numbers
  must be read per class. Treat Normal predictions as unreliable until Normal images from more than one source are available.
- **Corner-glyph shortcut.** A small overlay glyph in the image corner (outside the retina) is present in 3% of train Normal images but 86% of test Normal images (80-98% for the other classes in both splits),
  so it marks the image source and a model could use it as a label shortcut. It was removed by masking everything outside a circular field-of-view mask before feature extraction,
  and ResNet50 is fed the masked image instead of the raw frame for the same reason.
- **Vessel features are only a proxy.** Our image-derived vessel density agrees only moderately with the ground-truth masks (Pearson r = 0.56; it detects about two thirds of the true vessel pixels) and contributes little to the final model.
- **Colour dominates.** Most predictive features are colour statistics, and part of that signal is acquisition style (brightness/contrast alone predicted the class at about 66% in EDA). The model is not robust to blur or sensor noise
  (test macro-F1 falls from 0.58 to 0.39 with blur sigma 4 and to 0.12 with noise sigma 20).
- Small data: 594 training and 199 test images, so one test-recall value has a standard error of roughly 0.07. One train/test partition and one CV seed.
- The `handcrafted_norm` experiment was motivated by diagnostics that looked at test features; model selection itself used CV only.
- Not a medical device; for coursework only.

