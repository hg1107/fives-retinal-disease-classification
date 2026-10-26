# Dataset audit summary

Root: `/home/ghotra/Downloads/dataset`  
Originals: 800, masks: 800, other files: 2

## Findings

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

## Issues

- 2 non-image file(s) present: ['Quality Assessment.xlsx', 'Thumbs.db']
- 6 exact-duplicate group(s) by md5 (labels: ['AMD']; 0 span splits): de-duplicate before CV or folds leak
- 3 further near-duplicate pair(s) that are not byte-identical
- 1 confirmed near-duplicate pair(s) across train/test (leakage)
