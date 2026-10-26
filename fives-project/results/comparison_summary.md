# Model comparison summary

All numbers come from `results/tables/baseline_*.csv` (script 06). CV = 5-fold stratified CV on the train split (594 images);
test = the official test split (199 images), used once. Identical preprocessing (median impute, RobustScaler, variance filter,
correlation filter 0.95, SelectKBest k=30) and identical folds for all 24 model x experiment combinations. Hyperparameters are still at defaults.

## Selected model (by CV macro-F1, not test)

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

## Most confused class pairs (test, mean over all 24 models; selected model in brackets)

- true Normal predicted as Glaucoma: 48.7% of that class [38.0%, 19 images]
- true Diabetic Retinopathy predicted as AMD: 23.2% of that class [16.0%, 8 images]
- true Glaucoma predicted as Normal: 18.7% of that class [14.0%, 7 images]
- true AMD predicted as Diabetic Retinopathy: 18.5% of that class [24.5%, 12 images]

Where true Normal test images go on average: Normal 32%, AMD 9%, Diabetic Retinopathy 10%, Glaucoma 49%.

## Normal-class source shift

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

## Limitations

- Test set size is 199 images; the standard error of the selected model's test accuracy is about ±0.035. Differences of a few points between models are within noise.
- One train/test partition and one CV seed; no repeated CV. Fold-wise CV F1 std is about 0.033.
- The shift is a property of the data source, not something the models can learn from this training set: reported test scores are therefore a pessimistic-but-honest estimate for images from the test source, and CV scores are optimistic for it. Which one matters for deployment depends on which source future images resemble.
- Hyperparameters, feature count (k=30) and the ResNet/both feature sets were not tuned here, so ResNet features are probably under-used.
- Earlier diagnostics (the Normal shift test, a quick random-forest probe) looked at test data and motivated creating the `handcrafted_norm` variant; no model or hyperparameter was selected from test results, and the selected model above is chosen on CV only, but the existence of that experiment is not test-blind.
- Vessel features are only moderately correlated with ground-truth vessel density (Pearson 0.56), colour features dominate feature rankings, and global intensity alone predicted class at ~66% in EDA, so part of the performance may be acquisition-style shortcuts rather than pathology.

## Note on regression metrics

MAE, MSE, RMSE and R² are regression metrics and do not apply to this 4-class classification task; accuracy, macro precision/recall/F1, per-class recall, confusion matrices and one-vs-rest ROC-AUC are used instead.
