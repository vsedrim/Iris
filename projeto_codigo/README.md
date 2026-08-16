# DM2 Iris Biomarker Research Code

This directory contains the computational protocol described in `../projeto_pgc.tex`.
It evaluates whether conventional iris photographs contain measurable signals associated with type
2 diabetes mellitus (DM2). The code does not implement iridology and is not intended for individual
diagnosis or clinical decisions.

The implementation includes:

- restricted conversion of the legacy Python 2 dataset to NPZ without pickle at experiment time;
- raw-image pupil and iris segmentation with Daugman-style rubber-sheet normalization;
- classical intensity, LBP, and Haralick descriptors;
- HSV/LAB color moments and regional histograms;
- explicit RGB proxies for vascular and normalized-iris morphology;
- pupil/iris geometry when raw images and segmentation contours are available;
- grouped, stratified cross-validation with shared folds across all models;
- logistic regression, SVM, random forest, MLP, and AdaBoost baselines;
- ResNet-18, EfficientNet-B0, and ViT-B/16 transfer learning;
- fold-level predictions, five required metrics, mean, standard deviation, seeds, and photometric
  robustness results.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for requirement traceability and scientific limits.
See [docs/BASELINE_RESULTS.md](docs/BASELINE_RESULTS.md) for the exploratory classical baseline.

## Data Integrity Finding

`Data.zip` contains 108 control rows and 88 reported-diabetic rows in `personBase_invert`, matching
the published counts. The preparation command found 73 exact image duplicates and retained 123 unique
images: 66 reported controls and 57 reported-diabetic samples. Duplicates are removed deterministically
and listed in `metadata.json`. Results from the old sample-level implementation must therefore be
recomputed before comparison.

The archive contains normalized iris strips only. It does not contain raw photographs, subject
identifiers, eye laterality, pupil contours, iris contours, or acquisition metadata. Pupil diameter
and border morphology cannot be reconstructed from this archive, so those fields remain unavailable.
Row IDs are not proof of subject identity, and the positive class cannot be verified as DM2.

## Installation

Python 3.12 is required. From this directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,deep]"
```

Omit `deep` when PyTorch experiments are not needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Legacy Dataset Workflow

Prepare a deduplicated NPZ dataset:

```powershell
.\.venv\Scripts\python.exe -m iris_dm2 prepare `
  --archive .\Data.zip `
  --layout personBase_invert `
  --output .\data\legacy `
  --acknowledge-unverified-legacy-metadata
```

Extract all image-based feature groups and robustness variants:

```powershell
.\.venv\Scripts\python.exe -m iris_dm2 extract `
  --dataset .\data\legacy\dataset.npz `
  --output .\data\legacy\features `
  --feature-groups classic color vascular morphology `
  --variants original brightness_low brightness_high contrast_low contrast_high
```

Run the complete classical evaluation:

```powershell
.\.venv\Scripts\python.exe -m iris_dm2 evaluate `
  --dataset .\data\legacy\dataset.npz `
  --features .\data\legacy\features `
  --output .\results\classical `
  --models logistic_regression svm random_forest mlp adaboost `
  --feature-sets all classic color vascular morphology `
  --variants original brightness_low brightness_high contrast_low contrast_high `
  --seeds 42 123 2025 `
  --folds 5 `
  --allow-unverified-identity `
  --allow-unverified-labels
```

Run transfer learning with the same grouped split protocol:

```powershell
.\.venv\Scripts\python.exe -m iris_dm2 deep `
  --dataset .\data\legacy\dataset.npz `
  --output .\results\deep `
  --models resnet18 efficientnet_b0 vit_b_16 `
  --variants original brightness_low brightness_high contrast_low contrast_high `
  --seeds 42 123 2025 `
  --folds 5 `
  --epochs 10 `
  --device auto `
  --allow-unverified-identity `
  --allow-unverified-labels
```

Pretrained weights are downloaded by TorchVision on first use. Deep cross-validation is
compute-intensive. The `--no-pretrained` switch exists for execution checks and ablation only, not as
the primary protocol described in the project document.

Deep models draw their training samples from the original fold and apply seeded `ColorJitter` during
training. Named brightness and contrast variants are used only for test-fold evaluation. The resize,
normalization, and augmentation parameters are recorded in `deep_run_config.json`.

## Raw Image Workflow

Create a CSV manifest with one row per photograph:

```csv
sample_id,image_path,person_id,label,eye,diagnosis_source,diagnosis_verified
sample-0001,images/person-001-left.jpg,person-001,0,L,clinical_record,true
sample-0002,images/person-001-right.jpg,person-001,0,R,clinical_record,true
sample-0003,images/person-002-left.jpg,person-002,1,L,clinical_record,true
```

`label` must be `0` for control and `1` for DM2. `person_id` is mandatory because it defines the
cross-validation groups. `diagnosis_source` records where the supplied clinical label came from. The
`diagnosis_verified` field records whether the project has completed its external verification of that
label. The preparation step validates and preserves these fields but cannot independently authenticate a
medical record.

```powershell
.\.venv\Scripts\python.exe -m iris_dm2 preprocess `
  --manifest .\raw_manifest.csv `
  --output .\data\raw
```

The command writes normalized images, geometric features, a manifest, metadata, and
`exclusions.json`. The output manifest preserves each source path, SHA-256 hash, and diagnosis source.
Segmentations must pass minimum quality, contour containment, annulus-width, and image-bound checks;
failures are recorded and excluded rather than silently replaced.

## Outputs

| File | Purpose |
|---|---|
| `dataset.npz` | Images, labels, group IDs, provenance, optional geometry, and canonical metadata |
| `manifest.csv` | Sample, person, class, eye, diagnosis source, source path, and source hash |
| `metadata.json` | Source hash, deduplication record, assumptions, and limitations |
| `features_<variant>.npz` | Feature matrix, names, families, sample IDs, and perturbation |
| `metrics_by_fold.csv` | Classical metrics for every model, feature set, seed, fold, and variant |
| `predictions.csv` | Sample-level classical predictions for auditing |
| `fold_assignments.csv` | Exact person-grouped test assignment for every seed |
| `summary.csv` | Mean and standard deviation across folds and seeds |
| `deep_*.csv` | Equivalent reports for the transfer-learning branch |
| `run_config.json` | Runtime, package versions, hashes, parameters, and dataset limitations |

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

The tests cover restricted data conversion, duplicate removal, segmentation, rubber-sheet normalization,
feature schemas, photometric variants, person-group isolation, required metrics, and deep-model heads.

## Dataset Provenance and Retraction

The bundled archive is associated with the following retracted work and is retained only for local
provenance and implementation compatibility checks:

```bibtex
@inproceedings{iridology-icbme2018,
  author    = {Parsa Moradi and Naghme Nazer and Amirhosein Khasahmadi and Hoda Mohammadzadeh and Hasan Khojasteh Jafari},
  title     = {Discovering Informative Regions in Iris Images to Predict Diabetes},
  booktitle = {2018 25th National and 3rd International Iranian Conference on Biomedical Engineering},
  year      = {2018},
  doi       = {10.1109/ICBME.2018.8703564},
  note      = {Retracted. See DOI 10.1109/ICBME45317.2018.10207763}
}
```

IEEE marks the original article as `Retracted: Discovering Informative Regions in Iris Images to
Predict Diabetes`. The formal notice is `Retraction Notice: Discovering Informative Regions in Iris
Images to Predict Diabetes`, DOI `10.1109/ICBME45317.2018.10207763`.

The legacy README reports 88 diabetic and 108 control cases acquired under ophthalmologist supervision
at Farabi Hospital. The archive does not provide enough metadata to independently verify DM2 subtype,
identity, consent, licensing, or acquisition protocol. Do not redistribute it or use it as scientific
evidence without resolving those issues and the retraction with the data custodian.

Legacy ZIP members are checked for encryption, uncompressed size, compression ratio, and declared
size before restricted NumPy-only deserialization. Static opcode inspection is advisory; the runtime
allowlist is the enforcement boundary.

Canonical metadata is embedded in `dataset.npz`, so it contributes to the dataset hash. A human-readable
`metadata.json` sidecar is generated from the same object and must match the embedded copy when present.
