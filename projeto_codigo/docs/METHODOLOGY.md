# Methodology and Requirement Traceability

Status: implemented research protocol  
Scope: exploratory DM2 association research from conventional iris photographs

## Traceability

| Project requirement | Implementation | Evidence produced |
|---|---|---|
| Iris and pupil segmentation | `iris_dm2.preprocessing.segment_iris` | Geometry matrix and exclusions report |
| Rubber-sheet normalization | `iris_dm2.preprocessing.rubber_sheet_normalize` | RGB arrays with 201 radial by 720 angular samples |
| Intensity, LBP, and Haralick | `classic` feature group | Named columns in `features_<variant>.npz` |
| HSV/LAB moments and regional histograms | `color` feature group | Global and four-sector named columns |
| Vascularization feasibility | `vascular` feature group | Frangi vesselness, skeleton, branch, and red/green proxy columns |
| Normalized iris morphology | `morphology` feature group | Gradient, collarette, and crypt-blob proxy columns |
| Pupil/iris morphology | `geometry` feature group | Diameter ratio, circularity, radial deviation, spectral energy, and lobes |
| Person-based validation | `StratifiedGroupKFold` in `iris_dm2.evaluation` | Real person IDs are required from the raw manifest; runtime overlap assertion |
| Classical models | `iris_dm2.evaluation.build_model` | Fold metrics and sample predictions for five classifiers |
| ResNet, EfficientNet, and ViT | `iris_dm2.deep.build_deep_model` | Deep fold metrics, predictions, and run configuration |
| Accuracy, sensitivity, specificity, precision, and F1 | `classification_metrics` | Mean and standard deviation in summary CSV files |
| Seed stability | Repeated grouped folds | Seed on every metric, prediction, and assignment row |
| Photometric robustness | Five deterministic variants | Variant on every metric and prediction row |
| Recorded execution context | Hashes, versions, parameters, and exact folds | `metadata.json` and run configuration JSON files |
| Segmentation quality | Containment, annulus width, image bounds, and quality threshold | Rejected samples in `exclusions.json` |
| Per-sample provenance | Source path, SHA-256, diagnosis source, and manifest hash | Prepared `manifest.csv` and `metadata.json` |

## Dataset Protocol

The preferred input is a raw-image manifest with explicit `person_id`, clinical binary label, and eye
laterality. Each row also supplies `diagnosis_source`; the preparation output preserves that value,
`diagnosis_verified`, the resolved source path, an image SHA-256 hash, and the source-manifest hash.
Both eyes from one person receive the same group and therefore cannot cross the train/test boundary.
Classical and deep evaluation independently gate subject identity and clinical-label status. Any
provisional override is explicit in the command and recorded in the run configuration.

The bundled legacy archive lacks identifiers and raw photographs. Its `personBase` layouts contain
one row per reported subject count. The converter assigns stable synthetic IDs per row, then removes
exact pixel duplicates. This controls exact-copy leakage only. It cannot verify that two different
nonidentical rows do not belong to the same person because the source mapping is absent. Preparation
and evaluation both require explicit opt-in, and generated metadata marks the grouping and positive
class as unverified.

IEEE has retracted the publication associated with the archive. The original DOI is
`10.1109/ICBME.2018.8703564`; the retraction notice is
`10.1109/ICBME45317.2018.10207763`. The archive is not treated as scientific evidence and must not be
redistributed until provenance, license, consent, and the implications of the retraction are resolved.

## Model Protocol

Classical models use a pipeline that estimates variance filtering and scaling only from each training
fold. No transform is fitted on the complete dataset. Models use fixed, declared hyperparameters and
the MLP does not create a sample-level internal validation split, so outer test folds are not reused
for model selection.

Deep models use ImageNet initialization by default. Lower layers are frozen, upper blocks and the
binary head are fine-tuned, and class-weighted cross-entropy handles imbalance. Training samples come
from the original fold and receive seeded `ColorJitter` augmentation. The evaluation transform does
not apply that augmentation. Epoch count is fixed, and the test fold is not used for early stopping
or hyperparameter selection. Resize, normalization, and augmentation parameters are recorded in
`deep_run_config.json`.

Canonical metadata is stored inside the NPZ and therefore changes its SHA-256. The JSON sidecar is
checked against the embedded value, preventing a sidecar-only status change from bypassing gates.

Classical robustness models train on original-image features. Deep models also draw from original
training images, with the training-only augmentation described above. Named photometric variants are
used only for test-fold evaluation.

Raw-image segmentation is rejected when the quality score is below threshold, the pupil contour
leaves the iris contour, angular annulus width collapses, or either boundary leaves the image. These
checks validate geometric consistency, not clinical segmentation accuracy; representative manual
annotations remain necessary for a scientific segmentation study.

## Interpretation Limits

- RGB vesselness is a computational proxy. It is not AS-OCTA and has no clinical equivalence claim.
- Collarette and crypt measurements on normalized strips are image morphology proxies.
- Pupil diameter is a relative image measurement unless acquisition scale and illumination are
  controlled.
- The legacy archive cannot support geometric features because normalization discarded the required
  contours and radii.
- Synthetic row IDs in the legacy archive do not establish person-based isolation.
- The legacy positive class means reported diabetic; its DM2 subtype is not verifiable.
- Exact duplicate removal changes the effective legacy sample from 196 rows to 123 unique images.
- Independent external validation is not present and remains required before generalization claims.
- Multiple feature families and model comparisons are exploratory. Local findings require correction
  for multiple comparisons and confirmation on independent data.
- Outputs represent cohort-level research metrics. They are not individual diagnoses.

## Requirements Before Scientific Reporting

Before reporting scientific conclusions, obtain and document:

1. a provenance and license statement for the image archive;
2. confirmation that the positive label specifically represents DM2;
3. real subject identifiers or a custodian-generated group mapping;
4. acquisition metadata, including illumination and camera conditions;
5. an independent external cohort or a clearly stated absence of external validation;
6. a statistical analysis plan for confidence intervals and multiple-comparison correction.