# Brain MRI Classification and Explainable AI

A staged computer-vision project for four-class brain MRI image classification. The first stage is `notebooks/01_eda_and_data_audit.ipynb`, which inventories the dataset, checks image integrity and class balance, exports visualizations, detects exact duplicates, and optionally flags cross-split perceptual near-duplicate candidates. `notebooks/02_custom_cnn.ipynb` trains a custom CNN baseline, uses a hash-group-aware train/validation split, selects checkpoints by validation macro-F1, and reports official plus exact-duplicate-filtered test metrics. `notebooks/03_resnet18_transfer_learning.ipynb` reuses that split manifest to fine-tune an ImageNet-pretrained ResNet-18, perform validation-based checkpoint selection, compare performance with the CNN baseline, and generate a Grad-CAM visualization. `notebooks/04_vit_finetuning.ipynb` reuses the same manifest to fine-tune torchvision ViT-B/16, select a checkpoint by validation macro-F1, evaluate both official and exact-duplicate-filtered test subsets, and update the shared model comparison table.

> **Research/education only:** This project is not a clinical diagnostic system. Dataset quality, possible overlap, source differences, and lack of reliable patient identifiers limit any claims about generalization.

## Dataset

Download the [Brain Tumor MRI Dataset by Masoud Nickparvar on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and extract it under `data/`, for example:

```text
data/
└── brain-tumor-mri-dataset/
    ├── Training/
    │   ├── glioma/
    │   ├── meningioma/
    │   ├── notumor/
    │   └── pituitary/
    └── Testing/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

The EDA notebook attempts to detect `Training`/`Testing` or `train`/`test` folder pairs automatically. If discovery fails, edit `DATA_BASE` near the top of the notebook. Do not commit dataset images to Git.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open and run `notebooks/01_eda_and_data_audit.ipynb` from top to bottom. The perceptual near-duplicate scan can take a little while. Its matches are candidates for visual review, not confirmed duplicate labels.

## Project structure

```text
brain-mri-classification/
├── data/                 # Dataset goes here; do not commit images
├── notebooks/
│   ├── 01_eda_and_data_audit.ipynb     # Implemented
│   ├── 02_custom_cnn.ipynb             # Implemented: custom CNN baseline
│   ├── 03_resnet18_transfer_learning.ipynb  # Implemented: ResNet-18 transfer learning
│   ├── 04_vit_finetuning.ipynb         # Implemented: ViT-B/16 fine-tuning
│   └── 05_clip_blip_experiments.ipynb  # CLIP zero-shot + optional BLIP captioning
├── src/
│   ├── __init__.py
│   └── model_factory.py  # Model definitions shared with the inference app
├── models/               # Local checkpoints; do not commit large weights
├── results/              # CSVs, JSON, plots and experiment outputs
├── app.py                # Gradio inference app with class scores and Grad-CAM
├── requirements.txt
└── README.md
```

## EDA outputs

The first notebook writes outputs under `results/`:

- `image_inventory.csv` — per-image metadata and validity status
- `exact_duplicate_groups.csv` — all rows belonging to exact duplicate groups, if any
- `cross_split_exact_duplicates.csv` — exact duplicates appearing in both splits, if any
- `perceptual_hashes.csv` — pHash cache used to speed up reruns
- `cross_split_near_duplicate_candidates.csv` — candidate pairs with non-identical file hashes, if any
- `data_audit_summary.json` — summary of counts and audit settings
- `visualizations/` — class counts, sample images, image dimensions and candidate review grids

Some output files are created only when relevant matches exist.

## Custom CNN outputs (notebook 02)

- `model_data_split.csv` — train/validation/test manifest. The validation fold is created from the original training data using `StratifiedGroupKFold`, grouping exact SHA-256 hashes so byte-identical images are not split across train and validation. Reuse this manifest for ResNet-18 and ViT to make comparisons fair.
- `custom_cnn_history.csv` — per-epoch training loss and validation metrics.
- `custom_cnn_metrics.json` — configuration, selected checkpoint metadata, and final test metrics.
- `custom_cnn_classification_report.csv` — class-wise metrics on the official test split.
- `custom_cnn_predictions.csv` — class predictions and probabilities for the official test and exact-duplicate-filtered test evaluations.
- `custom_cnn_invalid_images.csv` and `custom_cnn_conflicting_hashes_excluded.csv` — generated if invalid images or exact-image label conflicts are found.
- `models/custom_cnn.pt` — best checkpoint selected with validation macro-F1. Do not commit large weights unless your repository policy allows it.
- `results/visualizations/custom_cnn_*.png` — learning curves and confusion matrices.

The exact-duplicate-filtered test evaluation removes test images whose SHA-256 appears in the original training folder. It does not remove all possible near-duplicates or establish patient-level independence.


## ResNet-18 transfer-learning outputs (notebook 03)

- `models/resnet18.pt` — best checkpoint selected by validation macro-F1; metadata records whether ImageNet pretrained weights were used.
- `resnet18_history.csv` and `resnet18_metrics.json` — training history, configuration, and final evaluations.
- `resnet18_classification_report.csv` and `resnet18_predictions.csv` — per-class metrics and test predictions/probabilities.
- `model_comparison.csv` — custom CNN versus ResNet-18 results, if the CNN summary from notebook 02 exists.
- `visualizations/resnet18_learning_curves.png`, confusion matrices, and `resnet18_gradcam_example.png` — diagnostic visualizations.

The ResNet notebook requires the shared `results/model_data_split.csv` created by notebook 02. Its model-specific transforms use ImageNet normalization. By default, training first updates the classification head, then fine-tunes `layer4` and the head. If pretrained weights cannot be downloaded, setting `USE_PRETRAINED_WEIGHTS = False` is possible for an offline smoke test, but that run must not be described as transfer learning.

## ViT-B/16 fine-tuning outputs (notebook 04)

- `models/vit_b16.pt` — best checkpoint selected by validation macro-F1.
- `vit_history.csv` and `vit_metrics.json` — epoch history, configuration, checkpoint-selection metadata, split sizes, and final evaluations.
- `vit_classification_report.csv` and `vit_predictions.csv` — per-class metrics and per-image predicted probabilities.
- `model_comparison.csv` — updated with ViT-B/16 rows while preserving existing CNN and ResNet-18 results.
- `visualizations/vit_learning_curves.png` and confusion matrices for the official and exact-duplicate-filtered test evaluations.

The ViT uses ImageNet mean/std normalization and torchvision's `ViT_B_16_Weights.DEFAULT` by default. The first run may download pretrained weights. Setting `USE_PRETRAINED_WEIGHTS = False` is intended only for a smoke test unless you explicitly want to study training from scratch; record that change and do not call the run pretrained transfer learning. The default implementation first trains the classification head, then unfreezes the final two encoder blocks and final LayerNorm.

## CLIP zero-shot and BLIP captioning (notebook 05)

- `results/clip_metrics.json` — CLIP checkpoint name, prompts, split counts and metrics.
- `results/clip_predictions.csv` and `results/clip_classification_report.csv` — image-level zero-shot predictions and official-test class metrics.
- `results/model_comparison.csv` — retains previous supervised rows and adds CLIP zero-shot rows when inference is run.
- `results/blip_captions.csv` and `results/visualizations/blip_caption_examples.png` — qualitative BLIP captions for a small, class-balanced test sample, if `RUN_BLIP = True`.

Notebook 05 reuses the split manifest from notebook 02. CLIP is a frozen, general-purpose image-text model evaluated without fine-tuning; BLIP is used only for optional captioning. Prompts are exploratory and not medically validated. CLIP scores are not calibrated clinical probabilities, and BLIP captions can be inaccurate on MRI images. Run notebooks 01–02 first; notebooks 03–04 are recommended to provide the full supervised comparison. The first run downloads model files from Hugging Face and requires internet access unless they are already cached.

## Gradio inference application

After training at least one supervised model, launch the interactive demo from the project root:

```bash
python app.py
```

The app provides a model selector for these checkpoints:

- `models/custom_cnn.pt` — produced by notebook 02
- `models/resnet18.pt` — produced by notebook 03
- `models/vit_b16.pt` — produced by notebook 04

Upload an image to view the predicted dataset label, softmax scores for all classes, and a Grad-CAM overlay. The app loads the fine-tuned checkpoint rather than downloading ImageNet weights again. Missing checkpoints are reported in the interface; run the corresponding training notebook first. The Model comparison tab reads `results/model_comparison.csv` after notebooks 03–05 have written benchmark results.

The Grad-CAM visualization is an exploratory visualization of model sensitivity, not a medical explanation or proof that the model relies on valid clinical features. Softmax scores are not calibrated clinical probabilities. The app and checkpoints are for research/education only, not diagnosis or treatment.

**Implementation note:** the model classes in `src/model_factory.py` reproduce the CNN, ResNet-18, and ViT-B/16 architectures used by notebooks 02–04 so the app can load their checkpoint state dictionaries. If you modify an architecture in a notebook, update the shared factory and retrain or validate checkpoint compatibility before deployment.

## Modeling principles

1. Keep validation data separate from the final test set.
2. Use training data only for fitting weights and augmentation; use validation data for checkpoint and hyperparameter selection.
3. Reuse `results/model_data_split.csv` for the custom CNN, ResNet-18 and ViT benchmarks.
4. Report macro-F1, per-class recall, confusion matrices and inference latency, not accuracy alone.
5. Inspect duplicate candidates before interpreting filtered-test results. A perceptual hash cannot establish patient identity.
6. Treat CLIP/BLIP experiments as exploratory, not medical advice or clinical inference.

## Current status

- [x] Notebook 01: EDA and data-quality audit
- [x] Notebook 02: custom CNN baseline
- [x] Notebook 03: ResNet-18 transfer learning
- [x] Notebook 04: ViT-B/16 fine-tuning
- [x] Notebook 05: CLIP zero-shot evaluation + optional BLIP captioning notebook (requires downloaded model checkpoints)
- [x] Gradio application scaffold with inference and Grad-CAM (requires trained checkpoints)
