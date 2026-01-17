# Augmenting Cross-View Geo-Localization with Spatial Semantics from Vision Foundation Models


## Acknowledgments

We would like to express our sincere gratitude to:

- The SAMGeo team ([https://samgeo.gishub.org/](https://samgeo.gishub.org/)) for their excellent foundation vision model that provides high-quality spatial priors for our BEV transformation task.
- The Sample4Geo authors ([https://github.com/Skyy93/Sample4Geo](https://github.com/Skyy93/Sample4Geo)) for their innovative work on dynamic similarity sampling strategies, which significantly improved our model's training process.
- The VIGORv2 dataset creators ([https://gitlab.com/vail-uvm/GPG2A](https://gitlab.com/vail-uvm/GPG2A)) for providing comprehensive ground-to-aerial image pairs with rich spatial and textual context, which greatly facilitated our research. 


## Abstract

Current CVGL approaches fall into two categories: (i) **feature-based** methods that learn purely 2-D visual descriptors and obtain strong accuracy but offer little interpretability, and (ii) **spatial-based** methods that model geometry and yield interpretable matches yet lag behind in performance due to shallow spatial reasoning and weak cross-view alignment.

We reformulate CVGL from a spatial perspective and introduce an **auxiliary task-enhanced framework** that couples a primary visual retrieval objective with an auxiliary **Spatial Semantic Alignment (SSA)** task.  The SSA task learns spatial structure by aligning ground-view Bird’s-Eye-View (BEV) predictions with satellite-view spatial semantics produced by a Vision Foundation Model (VFM).  Joint optimisation through shared encoders fuses spatial structure with visual appearance, delivering representations that are both geometrically grounded and discriminative.

Extensive experiments on three public benchmarks show the proposed method markedly surpasses previous spatial-based techniques while remaining competitive with state-of-the-art (SOTA) feature-based systems, achieving **98.48 % R@1** on **CVUSA** and **71.05 % R@1** on **CVACT_test**. Qualitative analyses using pixel-level activation maps and feature-space UMAP plots further confirm the effectiveness and interpretability of our approach.

## Key Contributions

- **Auxiliary task-enhanced network for CVGL** – integrates an SSA auxiliary task that injects explicit geometric guidance into the primary cross-view retrieval objective.
- **Spatial Semantic Alignment (SSA) objective** – aligns ground and satellite spatial representations in pixel space by supervising BEV-derived ground semantics with VFM-generated satellite semantics, enabling consistent scale, orientation and morphology across views.
- **State-of-the-art results for spatial-based methods** – achieves 98.48 % R@1 on CVUSA and 71.05 % R@1 on CVACT_test while closing the gap to leading feature-based approaches; extensive visualisations provide additional insights.

## Method Overview

The framework consists of two synergistic branches trained end-to-end with shared encoders:

1. **Cross-view retrieval branch**
   - Query encoder $E_q$ (ground images) and reference encoder $E_r$ (satellite images)
   - Global descriptors compressed via projector $P$
   - InfoNCE contrastive loss for retrieval
2. **SSA / BEV branch**
   - BEV decoder $D$ transforms ground features into a semantic BEV map
   - Satellite semantic map predicted by a frozen VFM
   - Cross-entropy + Dice loss enforces pixel-wise alignment

The total loss is a weighted sum of contrastive and SSA objectives.

## Repository Structure

```
model_training/
├── sample4geo/            # Training / evaluation core (code adapted from Sample4Geo)
├── eval_*.py              # Benchmark evaluation scripts (CVACT, CVUSA, VIGOR)
├── train_*.py             # Training entry points
├── parse_training_log.py  # Log analysis utility
└── ...
prior_generation/          # Scripts to generate VFM spatial priors
README.md                  # (this file)
```

## Installation

```bash
# clone repo
git clone <repo-url>
cd SSA_CVGL

# create environment and install dependencies
conda create -n ssacvgl python=3.8 -y
conda activate ssacvgl
pip install -r model_training/requirements.txt
```

> Tested with Python 3.8, PyTorch 1.8 and CUDA 11.0.

## Datasets

We evaluate on three standard benchmarks:

| Dataset | Ground images | Satellite images | Extras |
|---------|---------------|------------------|--------|
| CVUSA   | 35 k train / 8 k val | 35 k / 8 k | BEV labels |
| CVACT   | 35 k train / 92 k test | 35 k / 92 k | BEV labels |
| VIGOR   | 102 k train / 25 k val | 102 k / 25 k | Depth + BEV |

Please follow dataset preparation instructions in `model_training/sample4geo/dataset/` or use the provided helper scripts.

## Pre-processing: Generate Spatial Priors

The network relies on satellite-view spatial priors produced by SAMGeo. If you have already downloaded the provided priors, skip this step. Otherwise run:

```bash
# Generate SAMGeo masks for satellite reference images
python prior_generation/autosam_CVUSA.py   # for CVUSA
python prior_generation/autosam_CVACT.py   # for CVACT
python prior_generation/autosam_VIGOR.py   # for VIGOR

# Convert SAMGeo masks to BEV spatial semantic representations
python prior_generation/genre_BEVlabel_CVUSA_ACT.py   # CVUSA & CVACT
python prior_generation/genre_BEVlabel_VIGOR.py       # VIGOR
```

These scripts will create BEV label files under each dataset directory (e.g. `sat_BEV_label/` for CVUSA/CVACT and `map_repro/` for VIGOR) which are consumed during model training.

## Training Examples

```bash
# CVACT
python model_training/train_cvact.py --config configs/cvact.yaml

# CVUSA
python model_training/train_cvusa.py --config configs/cvusa.yaml

# VIGOR (same-area)
python model_training/train_vigor_DP.py --config configs/vigor.yaml
```

## Evaluation Examples

```bash
python model_training/eval_cvact.py  --checkpoint <path>
python model_training/eval_cvusa.py  --checkpoint <path>
python model_training/eval_vigor_same.py --checkpoint <path>
```

## Citation

If you use this repository, please cite our paper:

```

## License

Distributed under the MIT license; see `LICENSE` for details.
