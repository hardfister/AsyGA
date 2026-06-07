# AsyGA — Asymmetric Generative Adaptation for Multispectral Palmprint Recognition

## Quick Links

*  For a fast verification of our experimental results, please jump directly to the [Evaluation & Visualization](#evaluation--visualization) section.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Evaluation & Visualization](#evaluation--visualization)
- [Usage](#usage)
- [Dataset Format](#dataset-format)

---
## Overview

Multispectral palmprint recognition captures distinct biometric traits across multiple spectral bands (Red, Green, Blue, and NIR)—ranging from superficial ridge textures to deep subcutaneous vein patterns—delivering industry-grade spoofing resistance for high-security authentication. 

To overcome the challenges of feature degradation, data privacy, and severe hardware/environmental **Spectral Heterogeneity** in distributed edge environments, this project delivers a three-fold solution:

**Advanced Feature Extraction** — Integrates **Learnable Gabor Convolution (LGC)** layers with multi-order competitive blocks and **Coordinate Attention (CoordAtt)**. This architecture dynamically optimizes orientation- and scale-aware bandpass filters end-to-end to extract highly discriminative, topology-sensitive palmprint embeddings.
**Privacy-First Federated Learning** — Enables decentralized, collaborative model training across heterogeneous edge clients without transmitting raw biometric data. By isolating sensitive data on-device, the framework mitigates network bandwidth overhead while strictly maintaining user privacy.
**Asymmetric Personalization (AsyGA)** — Introduces a novel Personalized Federated Learning (pFL) paradigm. Clients leverage on-device generative diffusion models to synthesize single-domain surrogate images, driving local optimization through a hardware-friendly **Fidelity-Preserving Structural Constraint** to counteract cross-spectral domain shifts without overfitting or modality collapse.

## Project Structure

```
AsyGA/
├── main.py                  # Fine-tuning with ablation study control panel
├── psfed.py                 # PSFed-Palm: Federated learning (multiple FL modes)
├── dpfed.py                 # DPFed-Palm: SOTA Federated Learning
├── cache.py                 # Single-model feature extraction & EER evaluation with caching
├── test.py                  # Multi-model comparison: EER heatmaps, ROC, TAR, score distributions
├── tsne.py                  # t-SNE visualization with convex hull shading
├── requirements.txt         # Python dependencies
│
├── models/
│   ├── ccnet.py             # Original CCNet (Gabor CB + SE + ArcFace)
│   ├── ccnet1.py            # CCNet variant (SELayer naming fix: se_mlp → fc)
│   ├── ccnet2.py            # CCNet variant with Center Loss head and s=64 ArcFace
│   ├── VIT.py            # GT-PalmNet: Gabor-Transformer hybrid network
│   ├── compnet.py           # Co3Net: Coordinate-Aware Competitive Coding Network
│   ├── dataset.py           # MyDataset: multispectral ROI image loader with augmentations
│   └── __init__.py
│
├── utils/
│   ├── util.py              # Visualization utilities, feature map extraction, Gabor filter visualization
│   ├── util_contra_feature_mask.py        # Contrastive feature masking strategies
│   ├── util_contra_feature_twostage2.py   # Two-stage contrastive feature pipeline (v2)
│   ├── util_contra_feature_twostage3.py   # Two-stage contrastive feature pipeline (v3)
│   ├── util_contra_feature_twostagefirst.py # Two-stage contrastive feature pipeline (initial)
│   ├── util_new.py          # Additional utility functions
│   └── __init__.py
│
├── generater/               # Domain generalization training module
│   ├── config.py            # Argument parser & dataset configuration hub
│   ├── clf_train.py         # CLIP-conditioned ResNet18 classifier training
│   ├── clf_train.sh         # Shell launcher for distributed DG experiments
│   ├── dermamnist_data.py   # DermaMNIST medical imaging data loader
│   └── diffusers/           # HuggingFace Diffusers library (vendored for SD conditioning)
│
├── datacasia/               # CASIA multispectral palmprint dataset
│   ├── txt.py               # Generates train txt index files (100 IDs × 6 samples × 4 bands)
│   ├── train_RED.txt
│   ├── train_GREEN.txt
│   ├── train_BLUE.txt
│   └── train_NIR.txt
│
└── datapolyu/               # PolyU multispectral palmprint dataset
    └── txt.py               # Generates train txt index files (500 IDs × 12 samples × 4 bands)
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 12.1 (for PyTorch GPU acceleration)
- 8 GB+ GPU VRAM recommended

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd AsyGA

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

| Package | Version | Purpose |
|---|---|---|
| PyTorch | 2.5.1+cu121 | Deep learning framework |
| TorchVision | 0.20.1+cu121 | Image transforms |
| NumPy | 2.4.6 | Numerical computation |
| SciPy | 1.17.1 | Scientific computing, EER interpolation |
| scikit-learn | 1.9.0 | t-SNE, ROC metrics |
| Pandas | 3.0.3 | EER matrix construction |
| OpenCV | 4.13.0 | Image I/O |
| Pillow | 12.2.0 | Image processing |
| Matplotlib | 3.10.9 | Plotting |
| Seaborn | 0.13.2 | Heatmap visualization |
| loss | 0.1.2 | Supervised contrastive loss utilities |

For the domain generalization module (`generater/`), additional dependencies include `transformers` (HuggingFace) and `diffusers`.

---

### Key Hyperparameters

| Parameter | PSFed-Palm | DPFed-Palm | Description |
|---|---|---|---|
| `batch_size` | 128 | 128 | Training batch size |
| `epoch_num` | 3 | 3 | Local training epochs per round |
| `com` | 15 | 13 | Total communication rounds |
| `lr` | 0.001 | 0.001 | Client learning rate |
| `temp` | 0.07 | 0.07 | SupCon temperature τ |
| `mu` | 1e-2 | — | FedProx proximal coefficient |
| `k` | — | 10 | FedAvg warm-up rounds |
| `m_step` | — | 0.1 | PFL m-search step size |

---


### Training Modes

- **Standard**: Single-domain or aggregated training with few-shot support (`num_shot`)
- **Multi-client** (`multiclient`): Federated-style training across domain-specific clients
- **FGL** (`fgl`): Federated Group Learning with per-domain data
- **FedD3** (`fedd3`): Dataset distillation with pre-computed synthetic coreset (`kip.pt`)

### Key Configuration (`config.py`)

```bash
python generater/clf_train.py \
    --dataset domainnet \
    --train_type train_16 \
    --train_batch_size 64 \
    --pretrained \
    --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4
```

---

## Evaluation & Visualization


### Instructions for Reviewers: Reproducing EER and Visualizations

To facilitate a quick and effortless verification of our main EER results and figures, please follow the steps below. 
run
```bash
pip install -r requirements.txt
```

### Step 1: Download Model Checkpoints and Datasets
Please download the pre-trained checkpoints from the links provided below and place them into their respective directories:

1. **SOTA Method (DPFedPalm):** Download [[Link]]([[Click Link Here](https://huggingface.co/hardfister/AsyGA/tree/main)]) and save it to `weightdp\checkpoint\net_params_best.pth`
2. **Our Method (AsyGA):** Download [[Link]]([[Click Link Here](https://huggingface.co/hardfister/AsyGA/tree/main)]) and save it to `save\checkpoint\AsyGA\AsyGA.pth`
3. **PolyU Multispectral Palmprint:** Apply for [[Link]]([[Click Link Here](https://www4.comp.polyu.edu.hk/~csajaykr/database.php)]) and save it to `datapolyu\`and run
```bash
py roi.py
```
4. **CASIA Multispectral Palmprint:** Download [[Link]]([[Click Link Here](https://www.idealtest.org/#/datasetDetail/6)]) and save it to `save\checkpoint\AsyGA\AsyGA.pth`and run
```bash
py txt.py
```
---

### Step 2: Configure Paths and Run `cache.py`
You will need to run the evaluation twice—once for the SOTA baseline and once for our method. Please modify the paths in `cache.py` accordingly:

#### Run 1: Evaluating the SOTA Baseline (DPFedPalm)
Configure `cache.py` as follows:
```python
SAVE_DIR = r'save\dp'
MODEL_PATHS = {
    'Base': r'weightdp\checkpoint\net_params_best.pth'
}
```
Then execute the script in your terminal:
```Bash
python cache.py
```
Run 2: Evaluating Our Method (AsyGA)Comment out the baseline paths and uncomment our method in cache.py:
```Python
SAVE_DIR = r'save\AsyGA'
MODEL_PATHS = {
    'ours': r'save\checkpoint\AsyGA\AsyGA.pth'
}
```
### Then execute the script again:
```Bash
python cache.py
```
Note: By comparing the numerical EER outputs from these two individual runs, you can directly observe the significant performance improvement achieved by our proposed AsyGA over the SOTA DPFedPalm.
#### Step 3: Generate High-Quality FiguresTo visualize the comprehensive evaluation metrics presented in our paper, please run the main evaluation script:
```Bash
python test.py
```



### `test.py` — Multi-Model Comparison Suite
This script will process the cached data and automatically generate four high-quality figures in the save\pic directory. You will see the following progress logs in the console:
Generates 4 publication-quality figures for comparing multiple models side-by-side:

1. **EER Heatmap Matrix** (`EER.pdf`): Grid of Equal Error Rate matrices across spectral band pairs, each with a distinct colormap.
2. **ROC Curves** (`ROC.pdf`): Overlaid ROC curves on log-scale x-axis with distinct colors and line styles for black-and-white print compatibility.
3. **TAR Bar Chart** (`TAR.pdf`): True Acceptance Rate at a strict FAR = 10⁻⁶, with per-model bars.
4. **Score Distributions** (`IMP.pdf`): Genuine vs. Impostor matching score KDE plots with margin-gap annotations.

Configure `PKL_FILES` and `NEW_MODEL_NAMES` in the script before running:

```python
PKL_FILES = [
    r'save\dp\extracted_features_cache.pkl',
    r'save\Asy\extracted_features_cache.pkl'
]
NEW_MODEL_NAMES = ['DPFed-Palm', 'PSFed-Palm-AsyGA']
```

### `tsne.py` — Feature Embedding Visualization

Computes t-SNE dimensionality reduction of feature embeddings and generates:

- **Per-class convex hull shading** to visualize cluster compactness
- **Multi-model comparison** in a 1-row subplot layout
- **Automatic cache** of extracted features to avoid redundant inference

Supports comparing Baseline (pretrained), Traditional Fine-Tuning, and AsyGA fine-tuned models.

```python
MODEL_PATHS = {
    'Baseline':       r'path/to/pretrained.pth',
    'Traditional FT': r'path/to/traditional_ft.pth',
    'AsyGA':          r'path/to/asyga_ft.pth'
}
```

---

## Usage


### Main

Edit the ablation switches at the top of `load_and_finetune()` in `main.py`:

```python
use_feature_anchor = True    # Enable teacher-anchor regularization
use_hard_freeze    = True   # Keep backbone trainable
use_bn_freeze      = True   # Keep BN statistics updating
use_contrastive_loss = True  # Enable SupCon loss
```
makesure the path is already
```
weightps\checkpoint\net_params_best.pth
```
Then run:

```bash
python main.py
```

### 1. Single-Model EER Evaluation

```bash
# Edit MODEL_PATHS and SAVE_DIR in cache.py first
python cache.py
```

### 2. Multi-Model Comparison

```bash
# Edit PKL_FILES and NEW_MODEL_NAMES in test.py first
python test.py
```

### 3. t-SNE Visualization

```bash
# Edit MODEL_PATHS in tsne.py first
python tsne.py
```

### 4. Domain Generalization Training

```bash
# Standard few-shot training on DomainNet
python generater/clf_train.py \
    --dataset domainnet \
    --train_type train_16 \
    --train_batch_size 64 \
    --pretrained

# Federated multi-client training on DermaMNIST
python generater/clf_train.py \
    --dataset dermamnist \
    --train_type multiclient_5_16 \
    --client_num 5 \
    --pretrained
```

---

## Dataset Format

### Palmprint Datasets (CASIA / PolyU)

The project uses multispectral palmprint datasets organized as text index files:

```
path/to/roi_image_001.bmp 0
path/to/roi_image_002.bmp 0
path/to/roi_image_003.bmp 1
...
```

Each line contains an image path and an integer class label separated by a space. Images are single-channel grayscale ROIs, automatically resized to 128×128 pixels. During training, data augmentation includes random choice of: color jitter (contrast ±5%), perspective distortion (scale ≤15%), and rotation (≤10°).

**Dataset generation scripts:**
- `datacasia/txt.py` — Generates train index files for CASIA (100 identities, 6 samples each across Red/Green/Blue/NIR)
- `datapolyu/txt.py` — Generates train index files for PolyU (500 identities, 12 samples each across R/G/B/I spectrum folders)

### Domain Generalization Datasets

The `generater/`  loaded through per-dataset data loader modules.
download stable-diffusion-v1-5 to args.pretrained_model_name_or_path = r"diffmodels\stable-diffusion-v1-5"


