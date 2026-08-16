# Exploratory Classical Baseline

Status: complete exploratory run  
Date: 2026-08-09  
Dataset: bundled `Data.zip`, `personBase_invert`, unverified row grouping, exact-copy deduplication

## Protocol

The run used 123 unique normalized iris images: 66 reported controls and 57 reported-diabetic
samples. It evaluated five models, five feature sets, five photometric conditions, three seeds, and
five folds. This produced:

| Artifact | Rows | Audit result |
|---|---:|---|
| Fold-level metrics | 1,875 | Complete |
| Sample predictions | 46,125 | Complete |
| Fold assignments | 369 | Complete |
| Missing or duplicate assignments | 0 | Pass |
| Synthetic row groups assigned across folds | 0 | Pass |
| Missing or duplicate predictions | 0 | Pass |

All models trained on original features. Photometric variants were applied only to test-fold
features. The generated run configuration records source hashes, package versions, parameters, and
dataset limitations. Zero row groups crossed folds, but source subject IDs are unavailable, so this
does not prove person-based isolation. The run records separate explicit overrides for unverified
identity and unverified clinical labels.

## Original Image Results

The ten highest mean accuracies under the unmodified test images were:

| Rank | Model | Feature set | Accuracy | F1 | Sensitivity | Specificity |
|---:|---|---|---:|---:|---:|---:|
| 1 | AdaBoost | Classic | 0.840 | 0.821 | 0.795 | 0.879 |
| 2 | Random Forest | All | 0.829 | 0.807 | 0.773 | 0.879 |
| 3 | AdaBoost | All | 0.824 | 0.809 | 0.808 | 0.839 |
| 4 | Random Forest | Classic | 0.819 | 0.797 | 0.784 | 0.849 |
| 5 | Random Forest | Color | 0.807 | 0.779 | 0.743 | 0.864 |
| 6 | MLP | Color | 0.802 | 0.786 | 0.797 | 0.808 |
| 7 | MLP | All | 0.794 | 0.774 | 0.773 | 0.814 |
| 8 | Logistic Regression | Morphology | 0.789 | 0.781 | 0.813 | 0.769 |
| 9 | Logistic Regression | Color | 0.784 | 0.764 | 0.774 | 0.795 |
| 10 | SVM | All | 0.781 | 0.754 | 0.738 | 0.819 |

The best original result used classical descriptors alone. Combining every feature family did not
improve the best score. This supports ablation-based interpretation rather than assuming that a larger
feature vector is better.

## Photometric Robustness

The largest accuracy loss was 0.327 for Random Forest with color features under reduced contrast:
0.807 on original images and 0.481 after perturbation. SVM color features lost 0.292, and logistic
regression color features lost 0.276 under the same condition.

This sensitivity is evidence that color-based conclusions depend strongly on acquisition conditions.
It should be treated as a central limitation, not as a secondary implementation detail.

## Deep Learning Execution Check

ResNet-18 transfer learning with ImageNet weights completed on the real dataset for two folds and one
epoch. EfficientNet-B0 and ViT-B/16 construction passed automated tests with the installed TorchVision
API. The machine has a CPU-only PyTorch build and no CUDA device.

The one-epoch ResNet run is an execution check. Its metrics are not a scientific result. The declared
deep protocol requires all three architectures, five folds, three seeds, fixed full epoch count, and
all robustness conditions on suitable compute.

## Interpretation

These results supersede direct use of the preliminary 0.92 accuracy from the legacy script because
the legacy archive contained 73 exact duplicate rows and the old evaluation did not enforce grouped
folds through the classifier API. The current baseline is a provisional reported-diabetic-versus-
control analysis, not a verified DM2 result. Source subject IDs, clinical label verification,
acquisition metadata, and an external cohort are unavailable.