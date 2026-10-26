# Ablation, source-shift and robustness summary

Final model: **voting** (soft-voting ensemble of tuned LogReg / SVM / RF pipelines) on **handcrafted_norm** features; CV macro-F1 0.721 (nested), test macro-F1 0.582.
Ablation refits clones of the final model's hyperparameters. Those were tuned on all features and on this same train data, so the ablation CV scores are not nested (the full-feature CV here, 0.739, is about 0.018 above the nested 0.721); compare configurations with each other, not with the nested figure, and subsets are slightly disadvantaged. Test numbers are reported, never used for selection.

## 1. Feature-group ablation (CV / test macro-F1)

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

## 2. Source shift of the Normal class

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

## 3. Robustness (test images degraded, features re-extracted, final model unchanged)

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

## 4. Error analysis

79 of 199 test images are misclassified; 38 of those errors are true-Normal images. Of the 12 most confident errors, 7 involve Normal
(true Normal: 4, predicted Normal: 3). See `abl_04_top_errors.png` and `abl_top_errors.csv`.

## What this means for trusting the test numbers

- The test macro-F1 (0.582) is an average of two different things: the three disease classes, whose train and test images are far more alike (control AUC at most 0.65; Glaucoma shows a mild shift, AMD and DR almost none) and whose test performance is close to CV, and Normal,
  whose train and test images are strongly separable (AUC 0.96) and whose test recall (0.24) is far below CV recall (0.82). Test numbers for the disease classes can be trusted at the level of the test-set noise (n=50 per class, about ±0.07 on a recall).
- The Normal test figures measure how well a model trained on the train Normal source transfers to a different source; they are not an estimate of how well the model detects Normal fundi in general, and CV cannot reveal them.
  Both CV and test numbers are therefore biased for Normal, in opposite directions: CV is optimistic for images from the test source, while the test (one source, 50 images) may be pessimistic or optimistic for any other source.
- The deployment risk is concrete: 38 of 50 test Normal images (76%) are labelled as a disease (most often Glaucoma), with mean confidence 0.53 on those wrong predictions. Treat Normal predictions from this model as unreliable until Normal training images from more than one source are available.
- Honest headline numbers to report: CV macro-F1 0.721 (train source), test macro-F1 0.582, test macro-F1 over the three disease classes only 0.760, Normal test recall 0.24.
