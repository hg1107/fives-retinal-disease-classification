# Final report material

Everything below is generated from the saved results of scripts 01-09 (`python scripts/make_report_material.py`). Paths are relative to the project root. Sections follow the report outline: dataset, EDA, preprocessing, algorithms, experimental setup, evaluation, tuning, results and discussion, conclusion.

## 0. Headline numbers

| item                                           | value                                                                                            |
|:-----------------------------------------------|:-------------------------------------------------------------------------------------------------|
| Final model                                    | voting (soft voting of tuned LogReg, SVM-RBF, RF), feature set `handcrafted_norm` (116 features) |
| CV macro-F1 (nested 5-fold, train)             | 0.721 ± 0.044                                                                                    |
| Test accuracy / macro-F1 / ROC-AUC (OvR)       | 0.603 / 0.582 / 0.866                                                                            |
| Test recall per class                          | Normal 0.24, AMD 0.67, Diabetic Retinopathy 0.70, Glaucoma 0.80                                  |
| Test macro-F1 over the 3 disease classes only  | 0.760                                                                                            |
| Best baseline by CV (any experiment)           | LogisticRegression on handcrafted_norm: CV macro-F1 0.710                                        |
| Normal train-vs-test separability (LogReg AUC) | handcrafted 0.976, handcrafted_norm 0.963, resnet 0.978                                          |

## 1. Dataset description

- **Source:** FIVES, a fundus image dataset for vessel segmentation, hosted on Figshare (DOI 10.6084/m9.figshare.19688169). Citation: Jin K., et al. Sci Data 9, 475 (2022). It is NOT a Kaggle dataset.
- **Content:** 800 colour fundus photographs, 2048 x 2048 RGB PNG, with a binary vessel-segmentation mask for each image (`Ground truth/`). The class is the last letter of the filename: A = AMD, D = Diabetic Retinopathy, G = Glaucoma, N = Normal; 200 images per class. Official split: 600 train (150 per class) / 200 test (50 per class). A `Quality Assessment.xlsx` and `Thumbs.db` files are also present and ignored.
- **Use of masks:** only for the EDA vessel-density plot and the sanity check of our own vessel features; never as model input.
- **Numeric ids restart in each split** (train/1_A and test/1_A are different images), so images are keyed as `<split>_<id>_<letter>`.

### Audit and cleaning

#### Dataset audit summary

Root: `/home/ghotra/Downloads/dataset`  
Originals: 800, masks: 800, other files: 2

##### Findings

- Layout: originals in `test/Original`, `train/Original`; masks (vessel segmentation, not class labels) in `test/Ground truth`, `train/Ground truth`; splits found: ['test', 'train'].
- Label parsing: all 800 original images match `<id>_<A|D|G|N>`.
- Numeric ids restart in each split (200 ids appear in both train and test, e.g. train/1_A and test/1_A are different images). Files must be keyed by split+filename, never id alone.
- Class counts per split:


| label                |   test |   train |   total |
|:---------------------|-------:|--------:|--------:|
| AMD                  |     50 |     150 |     200 |
| Diabetic Retinopathy |     50 |     150 |     200 |
| Glaucoma             |     50 |     150 |     200 |
| Normal               |     50 |     150 |     200 |

Classes are perfectly balanced within each split.
- Integrity: every original and mask opened and decoded without error.
- All originals are 2048x2048 RGB/PNG.
- Masks: all 800 originals have a same-name mask in the same split.
- Duplicates: 6 exact md5 groups; phash gave 2922 candidates (unreliable alone: it collides across classes), of which 9 are confirmed by pixel correlation >= 0.99; 1 confirmed pairs cross train/test.
- Quality (whole image, black background included), mean per class:


| label                |   brightness |   contrast |   blur_lapvar |   saturation |
|:---------------------|-------------:|-----------:|--------------:|-------------:|
| AMD                  |        65.72 |      40.52 |          9.29 |       209.51 |
| Diabetic Retinopathy |        52.94 |      33.58 |          8.14 |       216.57 |
| Glaucoma             |        37    |      26.24 |          7.56 |       195.89 |
| Normal               |        46.92 |      30.28 |          5.15 |       194.69 |

If these differ strongly by class, acquisition-style differences could leak into colour/intensity features (a shortcut to flag in EDA).
- Max/min ratio of class means: brightness=1.78, contrast=1.54, blur_lapvar=1.8, saturation=1.11

##### Issues

- 2 non-image file(s) present: ['Quality Assessment.xlsx', 'Thumbs.db']
- 6 exact-duplicate group(s) by md5 (labels: ['AMD']; 0 span splits): de-duplicate before CV or folds leak
- 3 further near-duplicate pair(s) that are not byte-identical
- 1 confirmed near-duplicate pair(s) across train/test (leakage)

Cleaning log (`results/tables/cleaning_log.csv`):

| split   | file      | reason                       | detail                                                             |
|:--------|:----------|:-----------------------------|:-------------------------------------------------------------------|
| train   | 132_A.png | exact_duplicate              | md5=35dc54e982bf1fe229d2221bbeaeae34; kept train/39_A.png          |
| train   | 97_A.png  | exact_duplicate              | md5=43b47604628b30acdc20761d21b72daf; kept train/53_A.png          |
| train   | 134_A.png | exact_duplicate              | md5=56115f6921063b22a53e6b9f6cf96fd1; kept train/55_A.png          |
| train   | 118_A.png | exact_duplicate              | md5=942ff7e5a25cff7e5adb834660a84177; kept train/34_A.png          |
| train   | 121_A.png | exact_duplicate              | md5=a58a62ab138d0543e3255aea7cf0cf06; kept train/61_A.png          |
| test    | 48_A.png  | exact_duplicate              | md5=bcccf20f77be8c137506c321cc8b9915; kept test/32_A.png           |
| train   | 80_A.png  | near_duplicate_crosses_split | pair with test/21_A.png (phash 6, corr 0.9937); train copy removed |

Manifest after cleaning (`data/processed/manifest.csv`): 793 images. Class counts:

| class                |   test |   train |
|:---------------------|-------:|--------:|
| Normal               |     50 |     150 |
| AMD                  |     49 |     144 |
| Diabetic Retinopathy |     50 |     150 |
| Glaucoma             |     50 |     150 |

All removed images are AMD (6 exact duplicates inside one split plus 1 train image that near-duplicates a test image). Test was never touched except for one exact in-test duplicate.

| figure | what it shows |
|---|---|
| `results/figures/eda_01_class_distribution.png` | class counts overall and by split |
| `results/figures/eda_02_sample_grid.png` | 4 random images per class with vessel masks |
| `results/figures/eda_03_image_size.png` | pixel dimensions and file sizes |

## 2. Exploratory data analysis

Key observations (`results/eda_insights.md`):

- **Class balance:** 793 images; train {'Normal': 150, 'AMD': 144, 'Diabetic Retinopathy': 150, 'Glaucoma': 150}, test {'Normal': 50, 'AMD': 49, 'Diabetic Retinopathy': 50, 'Glaucoma': 50}. Imbalance ratio (max/min) is 1.04 in train and 1.02 in test (only AMD lost images to de-duplication), so accuracy is a fair metric but macro-F1 will be reported too.
- **Uniform geometry:** 1 distinct resolution(s) (2048x2048), all RGB PNG; file size 0.4-3.3 MB. No resizing is needed to unify inputs, only to reduce cost.
- **Brightness and contrast differ strongly by class** (Kruskal-Wallis p<1e-10, e2=0.37 and p<1e-10, e2=0.33). Mean gray: Normal 46.9, AMD 65.6, Diabetic Retinopathy 52.9, Glaucoma 37.0; std gray: Normal 30.3, AMD 40.4, Diabetic Retinopathy 33.6, Glaucoma 26.2. AMD images are ~1.8x brighter than Glaucoma.
- **Sharpness differs by class, driven by the Normal training images** (p<1e-10, e2=0.27). Laplacian variance: Normal 5.15, AMD 9.28, Diabetic Retinopathy 8.14, Glaucoma 7.56. Normal is 4.09 in train but 8.36 in test, while the other classes agree across splits (train/test: AMD 9.28/9.30, Diabetic Retinopathy 8.17/8.05, Glaucoma 7.61/7.42) - the train Normal images (ids 451-600) look like a different acquisition batch.
- **Overlay glyph in the image corner is a source marker:** bright pixels in the top-left 120px corner (outside the FOV) appear in Normal 3%/86%, AMD 98%/96%, Diabetic Retinopathy 80%/82%, Glaucoma 85%/86% of images (train/test). Normal is the outlier: 3% of train but 86% of test Normal images carry it, so a model can learn 'no glyph (and soft focus) = Normal' on train and fail on test. Mask or crop the corner (use a circular FOV mask) before feature extraction, and expect a Normal train/test shift to show up as Normal recall errors.
- **Black border differs by class:** FOV fraction of the frame Normal 0.772, AMD 0.757, Diabetic Retinopathy 0.757, Glaucoma 0.747 (p<1e-10, e2=0.40). Whole-image brightness/contrast/blur therefore mix scene content with border size; mask the FOV before computing features.
- **Shortcut-learning risk is real:** a random forest on only the 4 quality numbers (brightness, contrast, blur, FOV fraction) reaches 66.3% +/- 3.2% 5-fold CV accuracy on train (chance 25%); mean R/G/B alone reaches 50.5% +/- 2.5%. Global intensity features must be normalised (e.g. per-image colour/illumination normalisation) and results interpreted with this baseline in mind.
- **Vessel density** (vessel px / FOV px) by class: Normal 0.115, AMD 0.097, Diabetic Retinopathy 0.090, Glaucoma 0.086; ordered Glaucoma < Diabetic Retinopathy < AMD < Normal (Kruskal-Wallis p<1e-10, e2=0.26). Masks are binary (2 gray levels), but they exist only for dataset images: using density as a feature in the app would need a vessel segmenter.
- **Colour balance differs:** mean channel values inside the FOV (R/G/B) differ across classes (R p<1e-10, e2=0.40; G e2=0.35; B e2=0.14); green share of total intensity: Normal 0.269, AMD 0.286, Diabetic Retinopathy 0.279, Glaucoma 0.264. In the histograms blue is concentrated at low values, green sits mid-low and red is broad and highest; AMD shows a red saturation spike at 255.
- **Metric structure:** brightness and contrast are strongly coupled (Spearman Normal 0.91, AMD 0.98, Diabetic Retinopathy 0.96, Glaucoma 0.90); all off-diagonal |rho| range 0.26-0.98 across classes, so correlated features are expected and regularisation/feature selection inside the CV pipeline is warranted. Train vs test distributions of the quality metrics differ (Mann-Whitney p: blur_score 2.4e-05, brightness 0.86, contrast 0.2).

Descriptive statistics by class: `results/tables/eda_summary.csv`.

| figure | what it shows |
|---|---|
| `results/figures/eda_04_quality_by_class.png` | blur / brightness / contrast by class (violin + box) |
| `results/figures/eda_05_rgb_histograms.png` | mean RGB histograms per class (FOV only) |
| `results/figures/eda_06_mean_image.png` | mean image per class |
| `results/figures/eda_07_vessel_density.png` | vessel density from ground-truth masks, by class |
| `results/figures/eda_08_quality_correlation.png` | Spearman correlation of quality metrics per class |

## 3. Preprocessing and feature engineering

### Image preprocessing (`src/features.py: preprocess`)

1. Resize longest side to 512 px (area interpolation), pad to square with black (images are already square).
2. Field-of-view (FOV) detection: blurred grayscale > 10, 31x31 morphological opening (removes the notch), largest component, holes filled. Crop to its bounding box, pad to square, resize to 512.
3. Circular FOV mask (radius 0.96 x half width) intersected with the detected FOV, eroded by 3 px. Everything outside is set to 0, which also removes the corner overlay glyph.
4. CLAHE (clip limit 2.0, 8x8 tiles) on the green channel.
5. **Normalised variant** (`handcrafted_norm`, used by the final model): green channel divided by its masked Gaussian blur (sigma = 64 px = width/8), then each of R, G, B standardised inside the FOV (mean 128, std 48, clipped to uint8), then CLAHE on the normalised green. Applied to the colour and texture groups only.

### Feature groups (116 handcrafted features)

| group   |   n_features | description                                                                                                                                                                                                                                    |   mean_anova_F | best_feature      |
|:--------|-------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------:|:------------------|
| color   |           81 | mean, std, skew, kurtosis, percentiles 5/25/50/75/95 of R,G,B,H,S,V,L,a,b inside the FOV                                                                                                                                                       |           63.8 | color_LAB_b_mean  |
| disc    |           10 | bright-region features: smoothed gray >= 0.9 / 0.8 x p99 area ratios, number of components, largest component centroid (x, y), distance to centre, eccentricity, solidity, peak-to-mean                                                        |           21.2 | disc_peak_to_mean |
| tex     |           14 | GLCM (contrast, homogeneity, energy, correlation; distances [1, 3, 5] x 4 angles, averaged, 32 levels, green channel) + uniform LBP histogram (P=8, R=1, 10 bins)                                                                              |           57.4 | tex_lbp_5         |
| vessel  |           11 | image-derived: multi-scale Frangi (sigmas [1, 2, 3, 4]) on green, thresholded at 0.1 x p99, small objects < 100 px removed; density, thickness mean/std, skeleton length/density, branch/end points, segments, tortuosity proxy, mean response |           30   | vessel_density    |

**Deep embeddings:** ResNet50 (ImageNet weights), global-average-pooled 2048-d, run in fp16 on the GPU (batch 32) on the preprocessed (FOV-cropped, masked, no CLAHE) image resized to 224 x 224. The preprocessed image is used instead of the raw frame because the raw frame carries the corner glyph that marks the image source.

### Feature analysis (train split only)

- Missing values: 0; inf: 0. Constant / near-constant features (>= 95% one value): 0.
- Feature pairs with |r| > 0.95: 162; a greedy correlation filter at 0.95 keeps 75 of the 116 features (many exact colour-space duplicates, e.g. RGB_R vs HSV_V).
- Skew: 48/116 features have |skew| > 1; median per-feature IQR-outlier fraction 3.5%, 28 features above 5%. This motivates `RobustScaler` as the default scaler.
- Top-ranked features by ANOVA F are almost all colour (28 of the top 30).

Top 12 features by ANOVA F (train):

| feature          | group   |     F |
|:-----------------|:--------|------:|
| color_LAB_b_mean | color   | 137.4 |
| color_LAB_b_p50  | color   | 133.4 |
| color_LAB_b_p75  | color   | 133.2 |
| color_LAB_b_p25  | color   | 124.9 |
| color_LAB_b_p95  | color   | 123.8 |
| color_RGB_R_mean | color   | 122   |
| color_HSV_V_mean | color   | 122   |
| color_RGB_R_p75  | color   | 121   |
| color_HSV_V_p75  | color   | 121   |
| color_RGB_R_p50  | color   | 119.7 |
| color_HSV_V_p50  | color   | 119.7 |
| color_LAB_L_mean | color   | 118.7 |

Vessel features vs ground-truth mask vessel density (train, sanity check only):

| feature                 |   pearson_r |   spearman_rho |
|:------------------------|------------:|---------------:|
| vessel_density          |       0.562 |          0.566 |
| vessel_skeleton_density |       0.41  |          0.461 |
| vessel_frangi_mean      |       0.379 |          0.392 |

### sklearn preprocessing pipeline (`src/preprocess.py: build_pipeline`)

`SimpleImputer(median)` -> `RobustScaler` (or `StandardScaler`) -> `VarianceThreshold(0)` -> `CorrelationFilter(0.95)` -> `SelectKBest(f_classif, k=30)`; all fitted inside the pipeline on the training part of each CV fold only. No SMOTE: classes are balanced (150/144/150/150 per class in train).

| figure | what it shows |
|---|---|
| `results/figures/feat_01_missing_constant.png` | missing / constant / near-constant analysis |
| `results/figures/feat_02_top12_distributions.png` | distributions of the top 12 features by class |
| `results/figures/feat_03_correlation_clustered.png` | clustered correlation heatmap with group strip |
| `results/figures/feat_04_outliers_skew.png` | IQR outliers and skewness |
| `results/figures/feat_05_pca_tsne.png` | PCA and t-SNE of the handcrafted features |
| `results/figures/feat_06_vessel_vs_groundtruth.png` | image-derived vessel features vs ground-truth density |

## 4. Algorithms

Six classifiers, each placed after the preprocessing pipeline in one `sklearn.pipeline.Pipeline` (`src/models.py`), default hyperparameters except those needed to run:

| model              | class                  | non-default / key settings                                                                             |
|:-------------------|:-----------------------|:-------------------------------------------------------------------------------------------------------|
| LogisticRegression | LogisticRegression     | {'C': 1.0, 'max_iter': 5000, 'random_state': 42}                                                       |
| RandomForest       | RandomForestClassifier | {'n_estimators': 100, 'random_state': 42}                                                              |
| SVM_RBF            | SVC                    | {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': 42} |
| KNN                | KNeighborsClassifier   | {'n_neighbors': 5}                                                                                     |
| XGBoost            | XGBClassifier          | {'device': 'cuda', 'gamma': None, 'n_estimators': None, 'random_state': 42, 'tree_method': 'hist'}     |
| GaussianNB         | GaussianNB             | {}                                                                                                     |

XGBoost uses `tree_method="hist"` on the GPU (CUDA).

**Tuning:** RandomizedSearchCV, 40 draws, 5-fold stratified CV, scoring macro-F1, over the classifier hyperparameters plus `select__k` {10,15,20,30,40,60,all}, the scaler {Robust, Standard} and the correlation threshold {0.85,...,0.995}.

**Ensembles** (built from the 3 tuned models): soft `VotingClassifier`, `StackingClassifier` (logistic-regression meta-learner, 5-fold internal CV, predicted probabilities) and `BaggingClassifier` (30 bootstrap replicas) of the best tuned base model.

Tuned hyperparameters (`results/best_hyperparameters.json`):

```json
{
  "experiment": "handcrafted_norm",
  "n_iter": 40,
  "bagging_base_model": "SVM_RBF",
  "models": {
    "LogisticRegression": {
      "best_inner_cv_f1": 0.7147248717396154,
      "params": {
        "clf__C": 0.6403036652671171,
        "clf__class_weight": null,
        "corr__threshold": 0.95,
        "scale": "RobustScaler",
        "select__k": "all"
      }
    },
    "SVM_RBF": {
      "best_inner_cv_f1": 0.7241890560671311,
      "params": {
        "clf__C": 423.9491473869762,
        "clf__class_weight": null,
        "clf__gamma": 0.0008160948743089922,
        "corr__threshold": 0.85,
        "scale": "StandardScaler",
        "select__k": 40
      }
    },
    "RandomForest": {
      "best_inner_cv_f1": 0.6892402493238419,
      "params": {
        "select__k": 60,
        "scale": "StandardScaler",
        "corr__threshold": 0.98,
        "clf__n_estimators": 500,
        "clf__min_samples_split": 4,
        "clf__min_samples_leaf": 2,
        "clf__max_features": 0.5,
        "clf__max_depth": null
      }
    }
  },
  "final_model": {
    "candidate": "voting",
    "cv_f1": 0.7208464388875477,
    "cv_f1_std": 0.04440740653777269,
    "cv_estimate": "nested 5-fold CV"
  }
}
```

## 5. Experimental setup

- **Split:** official train (594) / test (199); the test split is used once, for final reporting; no choice (features, hyperparameters, model) depends on it.
- **Cross-validation:** 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on the train split, identical folds for every model/experiment. Tuned models and ensembles are scored with nested CV (searches re-run inside each outer training fold).
- **Experiments (feature sets):** `handcrafted` (116), `handcrafted_norm` (116), `resnet` (2048), `both` (handcrafted + resnet = 2164).
- **Metrics:** accuracy, macro precision / recall / F1, ROC-AUC one-vs-rest (macro); per-class precision / recall / F1; confusion matrices. Regression metrics (MAE, MSE, RMSE, R²) do not apply to this classification task.
- **Leakage control:** imputer, scaler, variance/correlation filters and SelectKBest are inside the Pipeline; duplicates removed before splitting into folds; train/test near-duplicate removed from train.
- **Seed:** 42 everywhere. **Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU, driver CUDA 13.0 (nvidia-smi); **software:** Python 3.12.14, scikit-learn 1.9.1, xgboost 3.4.1, torch 2.14.1+cu130, numpy 2.5.4, pandas 3.1.0; Linux (Arch).

Fit and predict times of the baselines (full train fit, test predict; seconds):

| experiment       | model              |   fit_time_s |   predict_time_s |
|:-----------------|:-------------------|-------------:|-----------------:|
| handcrafted      | LogisticRegression |        0.018 |            0.001 |
| handcrafted      | RandomForest       |        0.103 |            0.024 |
| handcrafted      | SVM_RBF            |        0.045 |            0.005 |
| handcrafted      | KNN                |        0.011 |            0.002 |
| handcrafted      | XGBoost            |        0.313 |            0.007 |
| handcrafted      | GaussianNB         |        0.011 |            0.001 |
| handcrafted_norm | LogisticRegression |        0.019 |            0.002 |
| handcrafted_norm | RandomForest       |        0.105 |            0.024 |
| handcrafted_norm | SVM_RBF            |        0.046 |            0.005 |
| handcrafted_norm | KNN                |        0.012 |            0.003 |
| handcrafted_norm | XGBoost            |        0.322 |            0.006 |
| handcrafted_norm | GaussianNB         |        0.012 |            0.001 |
| resnet           | LogisticRegression |        0.474 |            0.008 |
| resnet           | RandomForest       |        0.55  |            0.018 |
| resnet           | SVM_RBF            |        0.488 |            0.01  |
| resnet           | KNN                |        0.454 |            0.008 |
| resnet           | XGBoost            |        0.751 |            0.012 |
| resnet           | GaussianNB         |        0.453 |            0.007 |
| both             | LogisticRegression |        0.504 |            0.013 |
| both             | RandomForest       |        0.562 |            0.03  |
| both             | SVM_RBF            |        0.464 |            0.011 |
| both             | KNN                |        0.454 |            0.009 |
| both             | XGBoost            |        0.76  |            0.014 |
| both             | GaussianNB         |        0.463 |            0.008 |

| figure | what it shows |
|---|---|
| `results/figures/cmp_08_time_vs_f1.png` | time vs macro-F1 |

## 6. Evaluation of the baseline models

All 24 combinations, CV (mean ± std over 5 folds) and test side by side (`results/tables/cmp_comparison_table.csv`):

| experiment       | model              | CV acc        | CV F1         |   CV AUC |   test acc |   test P |   test R |   test F1 |   test AUC |   gap F1 (CV-test) |
|:-----------------|:-------------------|:--------------|:--------------|---------:|-----------:|---------:|---------:|----------:|-----------:|-------------------:|
| handcrafted      | LogisticRegression | 0.678 ± 0.029 | 0.677 ± 0.029 |    0.886 |      0.648 |    0.657 |    0.649 |     0.646 |      0.879 |              0.031 |
| handcrafted      | RandomForest       | 0.672 ± 0.035 | 0.670 ± 0.035 |    0.884 |      0.598 |    0.599 |    0.598 |     0.583 |      0.844 |              0.088 |
| handcrafted      | SVM_RBF            | 0.658 ± 0.023 | 0.653 ± 0.025 |    0.883 |      0.608 |    0.619 |    0.609 |     0.592 |      0.846 |              0.061 |
| handcrafted      | KNN                | 0.618 ± 0.020 | 0.610 ± 0.020 |    0.847 |      0.543 |    0.539 |    0.543 |     0.538 |      0.768 |              0.072 |
| handcrafted      | XGBoost            | 0.675 ± 0.032 | 0.673 ± 0.030 |    0.886 |      0.563 |    0.549 |    0.564 |     0.538 |      0.84  |              0.135 |
| handcrafted      | GaussianNB         | 0.588 ± 0.020 | 0.577 ± 0.021 |    0.844 |      0.447 |    0.434 |    0.449 |     0.429 |      0.732 |              0.148 |
| handcrafted_norm | LogisticRegression | 0.710 ± 0.042 | 0.710 ± 0.041 |    0.899 |      0.563 |    0.549 |    0.563 |     0.542 |      0.827 |              0.168 |
| handcrafted_norm | RandomForest       | 0.672 ± 0.051 | 0.667 ± 0.054 |    0.883 |      0.583 |    0.583 |    0.583 |     0.577 |      0.844 |              0.09  |
| handcrafted_norm | SVM_RBF            | 0.678 ± 0.039 | 0.678 ± 0.036 |    0.893 |      0.598 |    0.594 |    0.598 |     0.59  |      0.845 |              0.088 |
| handcrafted_norm | KNN                | 0.633 ± 0.034 | 0.630 ± 0.032 |    0.851 |      0.563 |    0.562 |    0.563 |     0.562 |      0.799 |              0.068 |
| handcrafted_norm | XGBoost            | 0.658 ± 0.027 | 0.656 ± 0.028 |    0.873 |      0.633 |    0.634 |    0.634 |     0.628 |      0.858 |              0.028 |
| handcrafted_norm | GaussianNB         | 0.591 ± 0.039 | 0.589 ± 0.037 |    0.836 |      0.472 |    0.456 |    0.473 |     0.459 |      0.759 |              0.13  |
| resnet           | LogisticRegression | 0.660 ± 0.061 | 0.657 ± 0.060 |    0.881 |      0.558 |    0.566 |    0.559 |     0.557 |      0.842 |              0.1   |
| resnet           | RandomForest       | 0.638 ± 0.033 | 0.635 ± 0.033 |    0.86  |      0.518 |    0.522 |    0.518 |     0.516 |      0.815 |              0.119 |
| resnet           | SVM_RBF            | 0.657 ± 0.026 | 0.652 ± 0.025 |    0.875 |      0.538 |    0.554 |    0.538 |     0.542 |      0.818 |              0.111 |
| resnet           | KNN                | 0.589 ± 0.035 | 0.580 ± 0.034 |    0.823 |      0.533 |    0.542 |    0.534 |     0.533 |      0.772 |              0.047 |
| resnet           | XGBoost            | 0.635 ± 0.032 | 0.632 ± 0.029 |    0.86  |      0.558 |    0.561 |    0.559 |     0.555 |      0.806 |              0.077 |
| resnet           | GaussianNB         | 0.618 ± 0.031 | 0.608 ± 0.031 |    0.837 |      0.422 |    0.429 |    0.423 |     0.424 |      0.738 |              0.184 |
| both             | LogisticRegression | 0.673 ± 0.022 | 0.670 ± 0.025 |    0.895 |      0.563 |    0.558 |    0.563 |     0.555 |      0.858 |              0.115 |
| both             | RandomForest       | 0.685 ± 0.043 | 0.683 ± 0.044 |    0.891 |      0.538 |    0.541 |    0.538 |     0.529 |      0.841 |              0.154 |
| both             | SVM_RBF            | 0.690 ± 0.031 | 0.687 ± 0.032 |    0.902 |      0.558 |    0.553 |    0.559 |     0.548 |      0.847 |              0.139 |
| both             | KNN                | 0.685 ± 0.036 | 0.681 ± 0.037 |    0.869 |      0.533 |    0.529 |    0.533 |     0.526 |      0.756 |              0.155 |
| both             | XGBoost            | 0.677 ± 0.026 | 0.676 ± 0.026 |    0.885 |      0.563 |    0.557 |    0.563 |     0.553 |      0.843 |              0.123 |
| both             | GaussianNB         | 0.643 ± 0.033 | 0.637 ± 0.036 |    0.872 |      0.447 |    0.434 |    0.448 |     0.433 |      0.765 |              0.204 |

Per-class test recall and CV recall of the baselines (`results/tables/baseline_per_class.csv`), Normal rows:

| experiment       | model              |   test_precision |   test_recall |   test_f1 |   cv_recall_mean |
|:-----------------|:-------------------|-----------------:|--------------:|----------:|-----------------:|
| handcrafted      | LogisticRegression |            0.65  |          0.52 |     0.578 |            0.74  |
| handcrafted      | RandomForest       |            0.6   |          0.3  |     0.4   |            0.753 |
| handcrafted      | SVM_RBF            |            0.609 |          0.28 |     0.384 |            0.78  |
| handcrafted      | KNN                |            0.489 |          0.44 |     0.463 |            0.807 |
| handcrafted      | XGBoost            |            0.476 |          0.2  |     0.282 |            0.787 |
| handcrafted      | GaussianNB         |            0.205 |          0.16 |     0.18  |            0.767 |
| handcrafted_norm | LogisticRegression |            0.417 |          0.2  |     0.27  |            0.8   |
| handcrafted_norm | RandomForest       |            0.514 |          0.36 |     0.424 |            0.787 |
| handcrafted_norm | SVM_RBF            |            0.529 |          0.36 |     0.429 |            0.8   |
| handcrafted_norm | KNN                |            0.489 |          0.46 |     0.474 |            0.86  |
| handcrafted_norm | XGBoost            |            0.6   |          0.42 |     0.494 |            0.74  |
| handcrafted_norm | GaussianNB         |            0.323 |          0.2  |     0.247 |            0.713 |
| resnet           | LogisticRegression |            0.486 |          0.36 |     0.414 |            0.753 |
| resnet           | RandomForest       |            0.421 |          0.32 |     0.364 |            0.753 |
| resnet           | SVM_RBF            |            0.455 |          0.4  |     0.426 |            0.833 |
| resnet           | KNN                |            0.406 |          0.52 |     0.456 |            0.847 |
| resnet           | XGBoost            |            0.463 |          0.38 |     0.418 |            0.727 |
| resnet           | GaussianNB         |            0.378 |          0.34 |     0.358 |            0.86  |
| both             | LogisticRegression |            0.441 |          0.3  |     0.357 |            0.773 |
| both             | RandomForest       |            0.433 |          0.26 |     0.325 |            0.78  |
| both             | SVM_RBF            |            0.375 |          0.24 |     0.293 |            0.813 |
| both             | KNN                |            0.395 |          0.3  |     0.341 |            0.86  |
| both             | XGBoost            |            0.484 |          0.3  |     0.37  |            0.773 |
| both             | GaussianNB         |            0.182 |          0.12 |     0.145 |            0.813 |

Effect of per-image normalisation (handcrafted vs handcrafted_norm), `cmp_norm_effect.csv`:

| model   |   cv_f1_base |   test_f1_base |   normal_recall_test_base |   cv_f1_norm |   test_f1_norm |   normal_recall_test_norm |   d_cv_f1 |   d_test_f1 |   d_normal_recall_test |
|:--------|-------------:|---------------:|--------------------------:|-------------:|---------------:|--------------------------:|----------:|------------:|-----------------------:|
| LogReg  |        0.677 |          0.646 |                     0.52  |        0.71  |          0.542 |                     0.2   |     0.033 |      -0.104 |                 -0.32  |
| RF      |        0.67  |          0.583 |                     0.3   |        0.667 |          0.577 |                     0.36  |    -0.003 |      -0.006 |                  0.06  |
| SVM     |        0.653 |          0.592 |                     0.28  |        0.678 |          0.59  |                     0.36  |     0.025 |      -0.002 |                  0.08  |
| KNN     |        0.61  |          0.538 |                     0.44  |        0.63  |          0.562 |                     0.46  |     0.02  |       0.024 |                  0.02  |
| XGB     |        0.673 |          0.538 |                     0.2   |        0.656 |          0.628 |                     0.42  |    -0.017 |       0.09  |                  0.22  |
| GNB     |        0.577 |          0.429 |                     0.16  |        0.589 |          0.459 |                     0.2   |     0.012 |       0.029 |                  0.04  |
| MEAN    |        0.643 |          0.554 |                     0.317 |        0.655 |          0.56  |                     0.333 |     0.011 |       0.005 |                  0.017 |

McNemar test, top-2 models by CV F1 (`cmp_mcnemar.csv`):

| model_A                               | model_B        |   both_correct |   only_A_correct |   only_B_correct |   both_wrong |   chi2_cc |   p_chi2 |   p_exact |   acc_A |   acc_B |
|:--------------------------------------|:---------------|---------------:|-----------------:|-----------------:|-------------:|----------:|---------:|----------:|--------:|--------:|
| handcrafted_norm | LogisticRegression | both | SVM_RBF |             86 |               26 |               25 |           62 |         0 |        1 |         1 |  0.5628 |  0.5578 |

Most confused class pairs (`cmp_confused_pairs.csv`):

| true                 | predicted            |   mean_rate_all_24_models |   rate_selected_model |   count_selected_model |
|:---------------------|:---------------------|--------------------------:|----------------------:|-----------------------:|
| Normal               | Glaucoma             |                     0.487 |                 0.38  |                     19 |
| Diabetic Retinopathy | AMD                  |                     0.232 |                 0.16  |                      8 |
| Glaucoma             | Normal               |                     0.187 |                 0.14  |                      7 |
| AMD                  | Diabetic Retinopathy |                     0.185 |                 0.245 |                     12 |
| Glaucoma             | Diabetic Retinopathy |                     0.133 |                 0.16  |                      8 |
| Diabetic Retinopathy | Normal               |                     0.128 |                 0.08  |                      4 |
| Normal               | Diabetic Retinopathy |                     0.101 |                 0.3   |                     15 |
| Normal               | AMD                  |                     0.09  |                 0.12  |                      6 |
| AMD                  | Normal               |                     0.086 |                 0.061 |                      3 |
| Diabetic Retinopathy | Glaucoma             |                     0.083 |                 0.04  |                      2 |
| Glaucoma             | AMD                  |                     0.053 |                 0.04  |                      2 |
| AMD                  | Glaucoma             |                     0.042 |                 0.02  |                      1 |

### Model comparison summary

All numbers come from `results/tables/baseline_*.csv` (script 06). CV = 5-fold stratified CV on the train split (594 images);
test = the official test split (199 images), used once. Identical preprocessing (median impute, RobustScaler, variance filter,
correlation filter 0.95, SelectKBest k=30) and identical folds for all 24 model x experiment combinations. Hyperparameters are still at defaults.

#### Selected model (by CV macro-F1, not test)

**LogisticRegression** on **handcrafted_norm** features: CV macro-F1 0.710 ± 0.041, CV accuracy 0.710 ± 0.042.

Test result (single evaluation): accuracy 0.563, macro-F1 0.542, macro precision 0.549,
macro recall 0.563, ROC-AUC (OvR) 0.827. Gap CV - test macro-F1: +0.168.

| class | test precision | test recall | test F1 | CV recall |
|---|---|---|---|---|
| Normal | 0.42 | 0.20 | 0.27 | 0.80 |
| AMD | 0.67 | 0.67 | 0.67 | 0.66 |
| Diabetic Retinopathy | 0.51 | 0.72 | 0.60 | 0.60 |
| Glaucoma | 0.60 | 0.66 | 0.63 | 0.78 |

The CV lead of this model over the runner-up (SVM_RBF, both, CV F1 0.687) is 0.022,
smaller than one CV standard deviation (0.041), so the ranking among the top models is not statistically meaningful.
McNemar on test predictions of the top-2 by CV F1 (handcrafted_norm | LogisticRegression vs both | SVM_RBF): only-A-correct 26, only-B-correct 25,
exact p = 1.000, chi-square (continuity-corrected) p = 1.000; no significant difference at 0.05.

Best model per experiment by CV F1: handcrafted: LogisticRegression (CV 0.677, test 0.646); handcrafted_norm: LogisticRegression (CV 0.710, test 0.542); resnet: LogisticRegression (CV 0.657, test 0.557); both: SVM_RBF (CV 0.687, test 0.548).

#### Most confused class pairs (test, mean over all 24 models; selected model in brackets)

- true Normal predicted as Glaucoma: 48.7% of that class [38.0%, 19 images]
- true Diabetic Retinopathy predicted as AMD: 23.2% of that class [16.0%, 8 images]
- true Glaucoma predicted as Normal: 18.7% of that class [14.0%, 7 images]
- true AMD predicted as Diabetic Retinopathy: 18.5% of that class [24.5%, 12 images]

Where true Normal test images go on average: Normal 32%, AMD 9%, Diabetic Retinopathy 10%, Glaucoma 49%.

#### Normal-class source shift

Normal images in the train split come from a different source than the Normal images in the test split (EDA: a corner overlay glyph is present in 3% of
train Normal but 86% of test Normal images; the other classes show it at 80-98% in both splits).
- Mean blur score of Normal images: 4.09 in train vs 8.36 in test.
- Features whose Normal distribution differs between train and test (Mann-Whitney p < 0.001): 76/116 (handcrafted), 67/116 (handcrafted_norm).
  For AMD / DR / Glaucoma the counts are 0 / 0 / 2 (handcrafted) and 0 / 0 / 2 (normalised).
- Consequence: mean Normal recall over all 24 models is 0.79 in CV but 0.32 on test; the other three classes average 0.61 in CV vs 0.62 on test.
  By experiment (CV / test): handcrafted 0.77/0.32; handcrafted_norm 0.78/0.33; resnet 0.80/0.39; both 0.80/0.25.
  Models trained on the train Normal source learn "that look" as Normal; test Normal images look different and are mostly assigned to other classes.
- Per-image normalisation (illumination correction + channel standardisation) did not remove the shift (see `cmp_norm_effect.csv`): mean change over the six models in CV F1 +0.011, test F1 +0.005, Normal test recall +0.017.
- CV folds are drawn from the train split only, so every CV Normal image comes from the same source as the Normal images the model trained on; CV cannot see this shift; the CV-to-test gap (macro-F1) ranges from +0.028 to +0.204 across the 24 combinations.

#### Limitations

- Test set size is 199 images; the standard error of the selected model's test accuracy is about ±0.035. Differences of a few points between models are within noise.
- One train/test partition and one CV seed; no repeated CV. Fold-wise CV F1 std is about 0.033.
- The shift is a property of the data source, not something the models can learn from this training set: reported test scores are therefore a pessimistic-but-honest estimate for images from the test source, and CV scores are optimistic for it. Which one matters for deployment depends on which source future images resemble.
- Hyperparameters, feature count (k=30) and the ResNet/both feature sets were not tuned here, so ResNet features are probably under-used.
- Earlier diagnostics (the Normal shift test, a quick random-forest probe) looked at test data and motivated creating the `handcrafted_norm` variant; no model or hyperparameter was selected from test results, and the selected model above is chosen on CV only, but the existence of that experiment is not test-blind.
- Vessel features are only moderately correlated with ground-truth vessel density (Pearson 0.56), colour features dominate feature rankings, and global intensity alone predicted class at ~66% in EDA, so part of the performance may be acquisition-style shortcuts rather than pathology.

#### Note on regression metrics

MAE, MSE, RMSE and R² are regression metrics and do not apply to this 4-class classification task; accuracy, macro precision/recall/F1, per-class recall, confusion matrices and one-vs-rest ROC-AUC are used instead.

| figure | what it shows |
|---|---|
| `results/figures/cmp_01_comparison_table.png` | comparison table (CV vs test, best highlighted) |
| `results/figures/cmp_02_test_metrics_bars.png` | test accuracy / precision / recall / F1 by model, per experiment |
| `results/figures/cmp_03_cv_test_gap.png` | CV-minus-test gap |
| `results/figures/cmp_04_per_class_recall.png` | per-class test recall heatmaps (Normal outlined) |
| `results/figures/cmp_05_confusion_matrices.png` | confusion matrices of the best models |
| `results/figures/cmp_06_roc_curves.png` | one-vs-rest ROC curves |
| `results/figures/cmp_07_cv_fold_boxplots.png` | fold-wise CV scores |
| `results/figures/cmp_10_norm_effect.png` | effect of per-image normalisation |

## 7. Hyperparameter tuning and ensembling

Experiment `handcrafted_norm`, models tuned: LogisticRegression, SVM_RBF, RandomForest. CV values for tuned models and ensembles are nested CV.

| stage    | candidate                   |   cv_f1_mean |   cv_f1_std |   test_f1_macro |   test_accuracy |   test_roc_auc_ovr |   test_recall_Normal |   test_recall_AMD |   test_recall_Diabetic Retinopathy |   test_recall_Glaucoma |   cv_recall_Normal |
|:---------|:----------------------------|-------------:|------------:|----------------:|----------------:|-------------------:|---------------------:|------------------:|-----------------------------------:|-----------------------:|-------------------:|
| baseline | baseline_LogisticRegression |        0.71  |       0.041 |           0.542 |           0.563 |              0.827 |                 0.2  |             0.673 |                               0.72 |                   0.66 |              0.8   |
| baseline | baseline_RandomForest       |        0.667 |       0.054 |           0.577 |           0.583 |              0.844 |                 0.36 |             0.633 |                               0.68 |                   0.66 |              0.787 |
| baseline | baseline_SVM_RBF            |        0.678 |       0.036 |           0.59  |           0.598 |              0.845 |                 0.36 |             0.633 |                               0.7  |                   0.7  |              0.8   |
| tuned    | tuned_LogisticRegression    |        0.701 |       0.036 |           0.546 |           0.568 |              0.84  |                 0.22 |             0.633 |                               0.62 |                   0.8  |              0.78  |
| tuned    | tuned_RandomForest          |        0.689 |       0.015 |           0.56  |           0.578 |              0.851 |                 0.24 |             0.714 |                               0.68 |                   0.68 |              0.8   |
| tuned    | tuned_SVM_RBF               |        0.696 |       0.028 |           0.609 |           0.613 |              0.86  |                 0.42 |             0.673 |                               0.7  |                   0.66 |              0.807 |
| ensemble | bagging                     |        0.706 |       0.028 |           0.578 |           0.593 |              0.861 |                 0.26 |             0.653 |                               0.7  |                   0.76 |              0.793 |
| ensemble | stacking                    |        0.716 |       0.031 |           0.6   |           0.623 |              0.863 |                 0.24 |             0.714 |                               0.76 |                   0.78 |              0.753 |
| ensemble | voting                      |        0.721 |       0.044 |           0.582 |           0.603 |              0.866 |                 0.24 |             0.673 |                               0.7  |                   0.8  |              0.8   |

Gain over the reference (baseline) against CV noise and the test bootstrap CI (`tune_gain_vs_noise.csv`):

| candidate                | reference                   |   cv_gain_f1 |   reference_cv_std |   gain_over_ref_std |   folds_improved |   test_gain_f1 |   test_gain_ci_low |   test_gain_ci_high | cv_verdict           | test_verdict                       |
|:-------------------------|:----------------------------|-------------:|-------------------:|--------------------:|-----------------:|---------------:|-------------------:|--------------------:|:---------------------|:-----------------------------------|
| tuned_LogisticRegression | baseline_LogisticRegression |       -0.009 |              0.041 |              -0.208 |                3 |          0.005 |             -0.053 |               0.063 | no CV gain           | positive on test but CI includes 0 |
| tuned_SVM_RBF            | baseline_SVM_RBF            |        0.018 |              0.036 |               0.508 |                3 |          0.019 |             -0.05  |               0.089 | gain within CV noise | positive on test but CI includes 0 |
| tuned_RandomForest       | baseline_RandomForest       |        0.022 |              0.054 |               0.406 |                3 |         -0.017 |             -0.073 |               0.04  | gain within CV noise | does not survive on test           |
| voting                   | baseline_LogisticRegression |        0.011 |              0.041 |               0.274 |                3 |          0.04  |             -0.017 |               0.097 | gain within CV noise | positive on test but CI includes 0 |
| stacking                 | baseline_LogisticRegression |        0.007 |              0.041 |               0.162 |                2 |          0.058 |              0.001 |               0.115 | gain within CV noise | survives on test (CI > 0)          |
| bagging                  | baseline_LogisticRegression |       -0.004 |              0.041 |              -0.087 |                2 |          0.036 |             -0.018 |               0.091 | no CV gain           | positive on test but CI includes 0 |

Top 15 permutation-importance features of the final model (macro-F1 drop, train CV folds):

| feature              |   importance_mean |   importance_std_across_folds |
|:---------------------|------------------:|------------------------------:|
| tex_lbp_0            |            0.0614 |                        0.024  |
| color_LAB_b_skew     |            0.0543 |                        0.0266 |
| tex_lbp_6            |            0.0409 |                        0.026  |
| color_LAB_b_kurt     |            0.0334 |                        0.0176 |
| disc_peak_to_mean    |            0.0289 |                        0.0218 |
| tex_glcm_correlation |            0.0238 |                        0.0116 |
| color_HSV_H_p50      |            0.0229 |                        0.0185 |
| tex_glcm_energy      |            0.0183 |                        0.0114 |
| color_RGB_G_mean     |            0.0145 |                        0.0095 |
| color_RGB_R_kurt     |            0.0142 |                        0.015  |
| color_HSV_V_skew     |            0.0137 |                        0.0069 |
| color_LAB_b_p75      |            0.0134 |                        0.0172 |
| color_LAB_b_mean     |            0.0133 |                        0.0179 |
| vessel_density       |            0.013  |                        0.0039 |
| color_HSV_H_kurt     |            0.0129 |                        0.0111 |

Learning curve (`tune_learning_curve.csv`):

|   train_size |   train_f1_mean |   train_f1_std |   cv_f1_mean |   cv_f1_std |
|-------------:|----------------:|---------------:|-------------:|------------:|
|           95 |           0.998 |          0.004 |        0.651 |       0.058 |
|          171 |           0.981 |          0.004 |        0.681 |       0.038 |
|          247 |           0.972 |          0.008 |        0.694 |       0.024 |
|          322 |           0.961 |          0.011 |        0.713 |       0.016 |
|          399 |           0.957 |          0.007 |        0.715 |       0.025 |
|          475 |           0.95  |          0.008 |        0.737 |       0.031 |

### Tuning and ensembling summary

Experiment: **handcrafted_norm** (best baseline CV macro-F1). Tuned models (top 3 by baseline CV F1 in that experiment): LogisticRegression, SVM_RBF, RandomForest.
Each was tuned with RandomizedSearchCV (40 draws, 5-fold stratified CV, macro-F1) over its hyperparameters, `select__k`, the scaler and the
correlation-filter threshold. Ensembles (soft voting, stacking with a logistic-regression meta-learner, bagging of SVM_RBF) were built from the tuned models.
CV for every tuned model and ensemble is **nested CV** (searches are re-run inside each outer training fold), so it is not inflated by the search;
baseline CV numbers are the plain 5-fold CV from script 06 on the same folds. Test was used once at the end.

#### Did tuning help?

- tuned_LogisticRegression: CV F1 -0.009 (-0.21 reference CV std; 3/5 folds improved), test F1 +0.005 (95% bootstrap CI -0.053 to +0.063) -> no CV gain; positive on test but CI includes 0
- tuned_SVM_RBF: CV F1 +0.018 (+0.51 reference CV std; 3/5 folds improved), test F1 +0.019 (95% bootstrap CI -0.050 to +0.089) -> gain within CV noise; positive on test but CI includes 0
- tuned_RandomForest: CV F1 +0.022 (+0.41 reference CV std; 3/5 folds improved), test F1 -0.017 (95% bootstrap CI -0.073 to +0.040) -> gain within CV noise; does not survive on test

Mean CV gain of tuning over the three baselines: +0.011 macro-F1 (mean reference CV std 0.044); mean test gain +0.002.

#### Did ensembling help?

Reference for ensembles = best baseline by CV (LogisticRegression, CV F1 0.710 ± 0.041, test F1 0.542).

- voting: CV F1 +0.011 (+0.27 reference CV std; 3/5 folds improved), test F1 +0.040 (95% bootstrap CI -0.017 to +0.097) -> gain within CV noise; positive on test but CI includes 0
- stacking: CV F1 +0.007 (+0.16 reference CV std; 2/5 folds improved), test F1 +0.058 (95% bootstrap CI +0.001 to +0.115) -> gain within CV noise; survives on test (CI > 0)
- bagging: CV F1 -0.004 (-0.09 reference CV std; 2/5 folds improved), test F1 +0.036 (95% bootstrap CI -0.018 to +0.091) -> no CV gain; positive on test but CI includes 0

#### Verdict on noise

Largest CV gain: tuned_RandomForest (+0.022, +0.41 reference CV std). No gain exceeds one CV standard deviation of the baseline, so every improvement is within CV noise. 1 of 6 candidates show a test gain whose bootstrap CI excludes 0. That result is borderline (lower CI bound close to 0) and is one of six comparisons at 95%, where about 0.3 false positives are expected by chance; it is not backed by a CV gain beyond noise.

#### Final model (selected by CV only)

**voting** with 0.721 ± 0.044 CV macro-F1 (nested 5-fold CV). Test (single evaluation): accuracy 0.603,
macro-F1 0.582, ROC-AUC 0.866; per-class test recall: Normal 0.24, AMD 0.67, Diabetic Retinopathy 0.70, Glaucoma 0.80.
Saved: `models/best_model.joblib` (fitted estimator; every member includes the full preprocessing Pipeline), `models/classes.json`,
`models/best_model_meta.json` (feature set `handcrafted_norm`, feature columns and extractor `src.features.extract_handcrafted_norm`), `results/best_hyperparameters.json`.
Top permutation-importance features (train CV folds): tex_lbp_0 (0.061), color_LAB_b_skew (0.054), tex_lbp_6 (0.041), color_LAB_b_kurt (0.033), disc_peak_to_mean (0.029).

#### Caveats

- **Normal source shift.** Normal training images come from a different source than Normal test images (see `comparison_summary.md`). CV cannot see this, so any CV gain
  may not transfer to test: CV Normal recall of the final model is 0.80 versus 0.24 on test. Selection by CV therefore cannot fix, and may even
  reward, fitting the train-source Normal look.
- The baseline CV score used as reference for LogisticRegression/ensembles is itself the maximum of 24 baseline CV scores, so it is slightly optimistic (winner's curse); the nested estimates of tuned models are not.
- Of the 10 most important features (permutation importance on train CV folds), 6 differ significantly between train and test Normal images (p < 0.001): tex_lbp_0, color_LAB_b_skew, color_LAB_b_kurt, disc_peak_to_mean, tex_glcm_correlation, color_HSV_H_p50. Importance computed on train therefore partly reflects the train-source Normal look.
- Tuned models are chosen from 40 draws on 594 images; differences between top configurations are small relative to fold-to-fold variation (inner CV std about 0.031).
- The test set has 199 images; a one-point change in macro-F1 is a few images. Bootstrap CIs above are wide for that reason.
- The CV selection of the final model compares candidates whose CV estimates differ by less than one standard deviation, so the choice among them is close to arbitrary.
- Bagging a stable linear model or SVM rarely helps: it reduces variance, which these models have little of.

| figure | what it shows |
|---|---|
| `results/figures/tune_01_before_after.png` | before vs after tuning/ensembling, CV vs test, macro-F1 and Normal recall |
| `results/figures/tune_02_learning_curve.png` | learning curve of the final model |
| `results/figures/tune_03_permutation_importance.png` | permutation importance (train CV folds) |

## 8. Results and discussion

### Feature-group ablation

| config                 |   n_features |   cv_f1_mean |   cv_f1_std |   test_f1 |   test_acc |
|:-----------------------|-------------:|-------------:|------------:|----------:|-----------:|
| all (handcrafted_norm) |          116 |        0.739 |       0.026 |     0.582 |      0.603 |
| only color             |           81 |        0.671 |       0.057 |     0.557 |      0.563 |
| only tex               |           14 |        0.623 |       0.04  |     0.483 |      0.508 |
| only vessel            |           11 |        0.537 |       0.038 |     0.414 |      0.422 |
| only disc              |           10 |        0.531 |       0.035 |     0.464 |      0.467 |
| without color          |           35 |        0.677 |       0.034 |     0.519 |      0.543 |
| without tex            |          102 |        0.695 |       0.027 |     0.58  |      0.588 |
| without vessel         |          105 |        0.73  |       0.034 |     0.582 |      0.593 |
| without disc           |          106 |        0.742 |       0.038 |     0.567 |      0.593 |
| ResNet50 only          |         2048 |        0.708 |       0.036 |     0.582 |      0.593 |

### Normal excluded, Normal-vs-rest

| model                                           |   macro_f1_3cls |   recall_AMD |   recall_Diabetic Retinopathy |   recall_Glaucoma |   cv_recall_AMD |   cv_recall_Diabetic Retinopathy |   cv_recall_Glaucoma |   cv_macro_f1_3cls |
|:------------------------------------------------|----------------:|-------------:|------------------------------:|------------------:|----------------:|---------------------------------:|---------------------:|-------------------:|
| final 4-class, scored on non-Normal test images |           0.76  |        0.673 |                          0.7  |              0.8  |           0.694 |                            0.653 |                0.793 |                    |
| 3-class (Normal removed from train and test)    |           0.772 |        0.694 |                          0.74 |              0.88 |           0.729 |                            0.713 |                0.82  |              0.753 |

| split                   |   auc_normal_vs_rest |   precision |   recall |    f1 |   n_normal |   n_pred_normal |
|:------------------------|---------------------:|------------:|---------:|------:|-----------:|----------------:|
| CV (train, out-of-fold) |                0.939 |       0.737 |     0.82 | 0.776 |        150 |             167 |
| test                    |                0.792 |       0.48  |     0.24 | 0.32  |         50 |              25 |

### Train-vs-test classifier AUC (0.5 = no shift)

| class                | feature_set      | model        |   train_vs_test_auc |   auc_std |   n_train |   n_test |
|:---------------------|:-----------------|:-------------|--------------------:|----------:|----------:|---------:|
| Normal               | handcrafted      | LogReg       |               0.976 |     0.013 |       150 |       50 |
| Normal               | handcrafted      | RandomForest |               0.979 |     0.016 |       150 |       50 |
| Normal               | handcrafted_norm | LogReg       |               0.963 |     0.022 |       150 |       50 |
| Normal               | handcrafted_norm | RandomForest |               0.965 |     0.018 |       150 |       50 |
| Normal               | resnet           | LogReg       |               0.978 |     0.021 |       150 |       50 |
| Normal               | resnet           | RandomForest |               0.969 |     0.023 |       150 |       50 |
| AMD                  | handcrafted      | LogReg       |               0.478 |     0.132 |       144 |       49 |
| AMD                  | handcrafted      | RandomForest |               0.495 |     0.117 |       144 |       49 |
| AMD                  | handcrafted_norm | LogReg       |               0.503 |     0.106 |       144 |       49 |
| AMD                  | handcrafted_norm | RandomForest |               0.482 |     0.102 |       144 |       49 |
| AMD                  | resnet           | LogReg       |               0.519 |     0.13  |       144 |       49 |
| AMD                  | resnet           | RandomForest |               0.586 |     0.099 |       144 |       49 |
| Diabetic Retinopathy | handcrafted      | LogReg       |               0.495 |     0.088 |       150 |       50 |
| Diabetic Retinopathy | handcrafted      | RandomForest |               0.431 |     0.094 |       150 |       50 |
| Diabetic Retinopathy | handcrafted_norm | LogReg       |               0.547 |     0.085 |       150 |       50 |
| Diabetic Retinopathy | handcrafted_norm | RandomForest |               0.556 |     0.102 |       150 |       50 |
| Diabetic Retinopathy | resnet           | LogReg       |               0.529 |     0.086 |       150 |       50 |
| Diabetic Retinopathy | resnet           | RandomForest |               0.543 |     0.066 |       150 |       50 |
| Glaucoma             | handcrafted      | LogReg       |               0.623 |     0.1   |       150 |       50 |
| Glaucoma             | handcrafted      | RandomForest |               0.58  |     0.084 |       150 |       50 |
| Glaucoma             | handcrafted_norm | LogReg       |               0.652 |     0.096 |       150 |       50 |
| Glaucoma             | handcrafted_norm | RandomForest |               0.614 |     0.095 |       150 |       50 |
| Glaucoma             | resnet           | LogReg       |               0.508 |     0.102 |       150 |       50 |
| Glaucoma             | resnet           | RandomForest |               0.54  |     0.079 |       150 |       50 |

By feature group (Normal only):

| feature_set      | group   |   n_features | model        |   train_vs_test_auc |   auc_std |
|:-----------------|:--------|-------------:|:-------------|--------------------:|----------:|
| handcrafted      | color   |           81 | LogReg       |               0.975 |     0.017 |
| handcrafted      | color   |           81 | RandomForest |               0.98  |     0.015 |
| handcrafted      | tex     |           14 | LogReg       |               0.907 |     0.047 |
| handcrafted      | tex     |           14 | RandomForest |               0.899 |     0.061 |
| handcrafted      | vessel  |           11 | LogReg       |               0.87  |     0.054 |
| handcrafted      | vessel  |           11 | RandomForest |               0.831 |     0.082 |
| handcrafted      | disc    |           10 | LogReg       |               0.855 |     0.049 |
| handcrafted      | disc    |           10 | RandomForest |               0.806 |     0.074 |
| handcrafted_norm | color   |           81 | LogReg       |               0.948 |     0.025 |
| handcrafted_norm | color   |           81 | RandomForest |               0.947 |     0.032 |
| handcrafted_norm | tex     |           14 | LogReg       |               0.9   |     0.055 |
| handcrafted_norm | tex     |           14 | RandomForest |               0.913 |     0.061 |
| handcrafted_norm | vessel  |           11 | LogReg       |               0.87  |     0.054 |
| handcrafted_norm | vessel  |           11 | RandomForest |               0.831 |     0.082 |
| handcrafted_norm | disc    |           10 | LogReg       |               0.855 |     0.049 |
| handcrafted_norm | disc    |           10 | RandomForest |               0.806 |     0.074 |

### Robustness

| perturbation    |   macro_f1 |   accuracy |   recall_Normal |   recall_AMD |   recall_Diabetic Retinopathy |   recall_Glaucoma |   share_predicted_Normal |   largest_predicted_class_share |   f1_drop_vs_clean |
|:----------------|-----------:|-----------:|----------------:|-------------:|------------------------------:|------------------:|-------------------------:|--------------------------------:|-------------------:|
| clean           |      0.582 |      0.603 |            0.24 |        0.673 |                          0.7  |              0.8  |                    0.126 |                           0.317 |              0     |
| blur sigma=2    |      0.544 |      0.543 |            0.42 |        0.531 |                          0.78 |              0.44 |                    0.196 |                           0.467 |              0.038 |
| blur sigma=4    |      0.387 |      0.407 |            0.54 |        0.163 |                          0.66 |              0.26 |                    0.382 |                           0.497 |              0.195 |
| brightness x0.7 |      0.582 |      0.583 |            0.38 |        0.612 |                          0.66 |              0.68 |                    0.231 |                           0.302 |             -0     |
| brightness x1.3 |      0.537 |      0.593 |            0.08 |        0.837 |                          0.62 |              0.84 |                    0.05  |                           0.357 |              0.045 |
| noise sigma=10  |      0.315 |      0.422 |            0.8  |        0.082 |                          0    |              0.8  |                    0.382 |                           0.598 |              0.267 |
| noise sigma=20  |      0.12  |      0.261 |            0.04 |        0     |                          0    |              1    |                    0.01  |                           0.99  |              0.461 |

### Most confident errors

| image_id   | true                 | predicted            |   confidence |   p_true_class | involves_Normal   |
|:-----------|:---------------------|:---------------------|-------------:|---------------:|:------------------|
| test_163_N | Normal               | Glaucoma             |        0.953 |          0.029 | True              |
| test_155_N | Normal               | Glaucoma             |        0.916 |          0.041 | True              |
| test_164_N | Normal               | Glaucoma             |        0.872 |          0.093 | True              |
| test_118_G | Glaucoma             | Normal               |        0.854 |          0.075 | True              |
| test_60_D  | Diabetic Retinopathy | AMD                  |        0.799 |          0.177 | False             |
| test_146_G | Glaucoma             | Normal               |        0.795 |          0.09  | True              |
| test_111_G | Glaucoma             | Diabetic Retinopathy |        0.767 |          0.131 | False             |
| test_183_N | Normal               | Glaucoma             |        0.766 |          0.151 | True              |
| test_142_G | Glaucoma             | Normal               |        0.755 |          0.197 | True              |
| test_36_A  | AMD                  | Diabetic Retinopathy |        0.73  |          0.213 | False             |
| test_59_D  | Diabetic Retinopathy | AMD                  |        0.722 |          0.242 | False             |
| test_12_A  | AMD                  | Diabetic Retinopathy |        0.709 |          0.243 | False             |

### Ablation, source-shift and robustness summary

Final model: **voting** (soft-voting ensemble of tuned LogReg / SVM / RF pipelines) on **handcrafted_norm** features; CV macro-F1 0.721 (nested), test macro-F1 0.582.
Ablation refits clones of the final model's hyperparameters. Those were tuned on all features and on this same train data, so the ablation CV scores are not nested (the full-feature CV here, 0.739, is about 0.018 above the nested 0.721); compare configurations with each other, not with the nested figure, and subsets are slightly disadvantaged. Test numbers are reported, never used for selection.

#### 1. Feature-group ablation (CV / test macro-F1)

| config                 |   n_features | CV            |   test |
|:-----------------------|-------------:|:--------------|-------:|
| all (handcrafted_norm) |          116 | 0.739 ± 0.026 |  0.582 |
| only color             |           81 | 0.671 ± 0.057 |  0.557 |
| only tex               |           14 | 0.623 ± 0.040 |  0.483 |
| only vessel            |           11 | 0.537 ± 0.038 |  0.414 |
| only disc              |           10 | 0.531 ± 0.035 |  0.464 |
| without color          |           35 | 0.677 ± 0.034 |  0.519 |
| without tex            |          102 | 0.695 ± 0.027 |  0.58  |
| without vessel         |          105 | 0.730 ± 0.034 |  0.582 |
| without disc           |          106 | 0.742 ± 0.038 |  0.567 |
| ResNet50 only          |         2048 | 0.708 ± 0.036 |  0.582 |

- Strongest single group by CV: **only color** (CV 0.671); removing a group hurts most for **without color** (CV 0.677 vs 0.739 with everything).
- Differences between most configurations are within the CV standard deviation (about 0.036); only large drops should be read as real.
- ResNet50 alone: CV 0.708, test 0.582.

#### 2. Source shift of the Normal class

**Normal excluded.** Scoring the final 4-class model on the 149 non-Normal test images (predicting Normal counts as an error) gives macro-F1 over the three classes of 0.760
(recall AMD 0.67, DR 0.70, Glaucoma 0.80; CV recall 0.69 / 0.65 / 0.79).
A 3-class model trained without any Normal images reaches CV macro-F1 0.753 and test macro-F1 0.772 (test recall 0.69 / 0.74 / 0.88),
so the three disease classes transfer from train to test much better than Normal does.

**Normal-vs-rest** (final model, P(Normal) as score): CV (out-of-fold) AUC 0.939, precision 0.74, recall 0.82, F1 0.78; test AUC 0.792, precision 0.48, recall 0.24, F1 0.32
(the model predicts Normal for 25 test images; 50 are truly Normal). True Normal test images are predicted as: Glaucoma 17, Normal 12, Diabetic Retinopathy 11, AMD 10.

**Train-vs-test classifier (images of a single class; AUC 0.5 = train and test are indistinguishable).** For Normal, a LogReg separates train from test with AUC
0.976 (handcrafted), 0.963 (handcrafted_norm), 0.978 (ResNet50); RandomForest 0.979 / 0.965 / 0.969.
The highest AUC among the other three classes (controls) is 0.623 / 0.652 / 0.529. Full table: `abl_train_vs_test_auc.csv`.

**Which feature groups carry the shift** (Normal, LogReg AUC): handcrafted_norm: color 0.948, tex 0.900, vessel 0.870, disc 0.855; handcrafted: color 0.975, tex 0.907, vessel 0.870, disc 0.855.
Most shifted single features (handcrafted_norm, single-feature AUC of train vs test; below 0.5 means lower in train): tex_lbp_9 (0.92), tex_lbp_3 (0.09), color_HSV_V_p25 (0.91), tex_lbp_1 (0.91), tex_lbp_5 (0.11).

#### 3. Robustness (test images degraded, features re-extracted, final model unchanged)

| perturbation    |   macro_f1 |   accuracy |   recall_Normal |   share_predicted_Normal |   f1_drop_vs_clean |
|:----------------|-----------:|-----------:|----------------:|-------------------------:|-------------------:|
| clean           |      0.582 |      0.603 |            0.24 |                    0.126 |              0     |
| blur sigma=2    |      0.544 |      0.543 |            0.42 |                    0.196 |              0.038 |
| blur sigma=4    |      0.387 |      0.407 |            0.54 |                    0.382 |              0.195 |
| brightness x0.7 |      0.582 |      0.583 |            0.38 |                    0.231 |             -0     |
| brightness x1.3 |      0.537 |      0.593 |            0.08 |                    0.05  |              0.045 |
| noise sigma=10  |      0.315 |      0.422 |            0.8  |                    0.382 |              0.267 |
| noise sigma=20  |      0.12  |      0.261 |            0.04 |                    0.01  |              0.461 |

Largest drop: **noise sigma=20** (+0.461 macro-F1). Perturbation definitions: Gaussian blur at 512 px scale; multiplicative brightness (so the black border stays black);
Gaussian noise on the retina only. The model was never trained with such perturbations. Blur raises Normal recall from 0.24 to 0.42 (sigma 2) and 0.54 (sigma 4) while lowering macro-F1, consistent with the model having learned 'blurrier image = Normal' from the blurrier train Normal images (blur score 4.1 vs 8.4 in test). Noise destroys the texture features: at sigma=20 macro-F1 is 0.12 (chance is 0.25) and 99% of predictions fall into one class; the model is not robust to sensor noise and should not be trusted on noisy or heavily compressed images.

#### 4. Error analysis

79 of 199 test images are misclassified; 38 of those errors are true-Normal images. Of the 12 most confident errors, 7 involve Normal
(true Normal: 4, predicted Normal: 3). See `abl_04_top_errors.png` and `abl_top_errors.csv`.

#### What this means for trusting the test numbers

- The test macro-F1 (0.582) is an average of two different things: the three disease classes, whose train and test images are far more alike (control AUC at most 0.65; Glaucoma shows a mild shift, AMD and DR almost none) and whose test performance is close to CV, and Normal,
  whose train and test images are strongly separable (AUC 0.96) and whose test recall (0.24) is far below CV recall (0.82). Test numbers for the disease classes can be trusted at the level of the test-set noise (n=50 per class, about ±0.07 on a recall).
- The Normal test figures measure how well a model trained on the train Normal source transfers to a different source; they are not an estimate of how well the model detects Normal fundi in general, and CV cannot reveal them.
  Both CV and test numbers are therefore biased for Normal, in opposite directions: CV is optimistic for images from the test source, while the test (one source, 50 images) may be pessimistic or optimistic for any other source.
- The deployment risk is concrete: 38 of 50 test Normal images (76%) are labelled as a disease (most often Glaucoma), with mean confidence 0.53 on those wrong predictions. Treat Normal predictions from this model as unreliable until Normal training images from more than one source are available.
- Honest headline numbers to report: CV macro-F1 0.721 (train source), test macro-F1 0.582, test macro-F1 over the three disease classes only 0.760, Normal test recall 0.24.

| figure | what it shows |
|---|---|
| `results/figures/abl_01_feature_groups.png` | feature-group ablation |
| `results/figures/abl_02_source_shift.png` | train-vs-test AUC per class and per feature group |
| `results/figures/abl_03_robustness.png` | macro-F1 and per-class recall under blur / brightness / noise |
| `results/figures/abl_04_top_errors.png` | 12 most confident wrong test predictions |

## 9. Conclusion and future scope

- A classical-ML pipeline (FOV preprocessing, per-image normalisation, 116 handcrafted features, tuned LogReg/SVM/RF soft-voting ensemble) reaches CV macro-F1 0.721 and test macro-F1 0.582 (accuracy 0.603) on 4 classes (chance 0.25).
- Tuning and ensembling gave gains of at most +0.022 macro-F1 in nested CV, below one CV standard deviation; the model ranking among the top candidates is not statistically meaningful (McNemar p = 1.00).
- The three disease classes generalise from train to test (test recall in line with CV); Normal does not, because Normal training and test images come from different sources (train-vs-test AUC about 0.96 for Normal, at most 0.65 for the others). Reported test performance mixes these two situations and must be read per class.
- Colour features carry most of the signal, and part of it is acquisition style rather than pathology; the model is not robust to blur or noise and over-uses 'blurrier = Normal'.
- **Future scope:** (1) Normal training images from several sources/devices, and a multi-centre test set; (2) fine-tuning a CNN on GPU with strong colour/blur augmentation and the FOV mask; (3) a trained vessel segmenter (the FIVES masks) in place of the Frangi proxy (r = 0.56 with ground truth) so vessel features can be accurate and used at inference; (4) probability calibration and an abstain option; (5) explainability (Grad-CAM / SHAP) and clinician review of the confident errors; (6) domain-adaptation or source-invariant features.

## Appendix: result files

Tables: `results/tables/` (audit_*, eda_summary, feat_*, baseline_*, cmp_*, tune_*, abl_*, cleaning_log). Predictions: `results/predictions/` (baselines `<experiment>__<model>.csv` with out-of-fold and test rows; tuned/ensembles `tune__<candidate>.csv`). Model: `models/best_model.joblib`, `models/classes.json`, `models/best_model_meta.json`. Features: `data/processed/features_*.csv|npy`, `manifest.csv`.
