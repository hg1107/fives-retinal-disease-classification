# Tuning and ensembling summary

Experiment: **handcrafted_norm** (best baseline CV macro-F1). Tuned models (top 3 by baseline CV F1 in that experiment): LogisticRegression, SVM_RBF, RandomForest.
Each was tuned with RandomizedSearchCV (40 draws, 5-fold stratified CV, macro-F1) over its hyperparameters, `select__k`, the scaler and the
correlation-filter threshold. Ensembles (soft voting, stacking with a logistic-regression meta-learner, bagging of SVM_RBF) were built from the tuned models.
CV for every tuned model and ensemble is **nested CV** (searches are re-run inside each outer training fold), so it is not inflated by the search;
baseline CV numbers are the plain 5-fold CV from script 06 on the same folds. Test was used once at the end.

## Did tuning help?

- tuned_LogisticRegression: CV F1 -0.009 (-0.21 reference CV std; 3/5 folds improved), test F1 +0.005 (95% bootstrap CI -0.053 to +0.063) -> no CV gain; positive on test but CI includes 0
- tuned_SVM_RBF: CV F1 +0.018 (+0.51 reference CV std; 3/5 folds improved), test F1 +0.019 (95% bootstrap CI -0.050 to +0.089) -> gain within CV noise; positive on test but CI includes 0
- tuned_RandomForest: CV F1 +0.022 (+0.41 reference CV std; 3/5 folds improved), test F1 -0.017 (95% bootstrap CI -0.073 to +0.040) -> gain within CV noise; does not survive on test

Mean CV gain of tuning over the three baselines: +0.011 macro-F1 (mean reference CV std 0.044); mean test gain +0.002.

## Did ensembling help?

Reference for ensembles = best baseline by CV (LogisticRegression, CV F1 0.710 ± 0.041, test F1 0.542).

- voting: CV F1 +0.011 (+0.27 reference CV std; 3/5 folds improved), test F1 +0.040 (95% bootstrap CI -0.017 to +0.097) -> gain within CV noise; positive on test but CI includes 0
- stacking: CV F1 +0.007 (+0.16 reference CV std; 2/5 folds improved), test F1 +0.058 (95% bootstrap CI +0.001 to +0.115) -> gain within CV noise; survives on test (CI > 0)
- bagging: CV F1 -0.004 (-0.09 reference CV std; 2/5 folds improved), test F1 +0.036 (95% bootstrap CI -0.018 to +0.091) -> no CV gain; positive on test but CI includes 0

## Verdict on noise

Largest CV gain: tuned_RandomForest (+0.022, +0.41 reference CV std). No gain exceeds one CV standard deviation of the baseline, so every improvement is within CV noise. 1 of 6 candidates show a test gain whose bootstrap CI excludes 0. That result is borderline (lower CI bound close to 0) and is one of six comparisons at 95%, where about 0.3 false positives are expected by chance; it is not backed by a CV gain beyond noise.

## Final model (selected by CV only)

**voting** with 0.721 ± 0.044 CV macro-F1 (nested 5-fold CV). Test (single evaluation): accuracy 0.603,
macro-F1 0.582, ROC-AUC 0.866; per-class test recall: Normal 0.24, AMD 0.67, Diabetic Retinopathy 0.70, Glaucoma 0.80.
Saved: `models/best_model.joblib` (fitted estimator; every member includes the full preprocessing Pipeline), `models/classes.json`,
`models/best_model_meta.json` (feature set `handcrafted_norm`, feature columns and extractor `src.features.extract_handcrafted_norm`), `results/best_hyperparameters.json`.
Top permutation-importance features (train CV folds): tex_lbp_0 (0.061), color_LAB_b_skew (0.054), tex_lbp_6 (0.041), color_LAB_b_kurt (0.033), disc_peak_to_mean (0.029).

## Caveats

- **Normal source shift.** Normal training images come from a different source than Normal test images (see `comparison_summary.md`). CV cannot see this, so any CV gain
  may not transfer to test: CV Normal recall of the final model is 0.80 versus 0.24 on test. Selection by CV therefore cannot fix, and may even
  reward, fitting the train-source Normal look.
- The baseline CV score used as reference for LogisticRegression/ensembles is itself the maximum of 24 baseline CV scores, so it is slightly optimistic (winner's curse); the nested estimates of tuned models are not.
- Of the 10 most important features (permutation importance on train CV folds), 6 differ significantly between train and test Normal images (p < 0.001): tex_lbp_0, color_LAB_b_skew, color_LAB_b_kurt, disc_peak_to_mean, tex_glcm_correlation, color_HSV_H_p50. Importance computed on train therefore partly reflects the train-source Normal look.
- Tuned models are chosen from 40 draws on 594 images; differences between top configurations are small relative to fold-to-fold variation (inner CV std about 0.031).
- The test set has 199 images; a one-point change in macro-F1 is a few images. Bootstrap CIs above are wide for that reason.
- The CV selection of the final model compares candidates whose CV estimates differ by less than one standard deviation, so the choice among them is close to arbitrary.
- Bagging a stable linear model or SVM rarely helps: it reduces variance, which these models have little of.
