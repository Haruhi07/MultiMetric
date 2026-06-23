# MultiMetric

**Optimising Factual Consistency in Summarisation via Preference Learning from Multiple Imperfect Metrics**

[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-blue)](#)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

Code repository for the EMNLP 2025 paper *Optimising Factual Consistency in Summarisation via Preference Learning from Multiple Imperfect Metrics* by Yuxuan Ye, Raul Santos-Rodriguez, and Edwin Simpson (University of Bristol).

---

## Overview

MultiMetric is a fully automated training pipeline that improves factual consistency in text summarisation **without requiring any human annotations or reference summaries**. It works by:

1. Generating **lexically similar summary pairs** by varying decoding strategies (e.g., beam search top-1 vs. top-2 candidates), so the model learns from subtle factual differences rather than stylistic variation.
2. Scoring both summaries with an **ensemble of imperfect factuality metrics** (SBERTScore + SummaC).
3. Keeping only pairs where **all metrics agree** on which summary is more factual, producing clean preference labels.
4. Fine-tuning the summariser with **Direct Preference Optimization (DPO)** using these labels.

The method generalises across model architectures and scales — from BART to GPT-J, LLaMA, and DeepSeek — and consistently boosts factuality, often bringing smaller models to performance levels comparable to much larger ones.

![Method overview](figs/introduction.png)

---

## Key Results

### Factuality improvements across models (XSUM)

| Model | SFT | MPO | Ours (AlignScore) |
|:------|:---:|:---:|:-----------------:|
| BART  | 61.9 | 62.0 | **86.6** |
| GPT-J | 59.7 | 53.5 | **75.8** |
| LLaMA | 86.1 | 79.8 | **88.7** |
| DeepSeek | 82.5 | 81.3 | **83.2** |

### Error frequency analysis

Training with MultiMetric reduces key error types (Noun, Predicate, Quantifier) across datasets and models.

| ![XSUM error frequency](figs/xsum.png) | ![TL;DR error frequency](figs/tldr.png) |
|:---:|:---:|
| Error frequencies on XSUM | Error frequencies on TL;DR |

### Training dynamics

![DPO evaluation accuracy during training](figs/curve.png)
*Evaluation accuracies over pairwise labels during DPO training (BART on XSUM).*

---

## Repository Structure

```
MultiMetric/
├── README.md              # This file
├── figs/                  # Figures from the paper
├── configs/               # Training configuration files
├── src/
│   ├── generate_pairs.py  # Generate lexically similar summary pairs
│   ├── annotate.py        # Score pairs with metrics and resolve conflicts
│   ├── train_dpo.py       # DPO training loop
│   └── evaluate.py        # Evaluation with AlignScore, BARTScore, FactCC
├── data/                  # Dataset scripts
└── scripts/               # Reproduce experiments
```

> Note: The code is being prepared for release and will be populated shortly.

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/MultiMetric.git
cd MultiMetric

# Install dependencies
pip install -r requirements.txt   # coming soon
```

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{ye2025multimetric,
  title={Optimising Factual Consistency in Summarisation via Preference Learning from Multiple Imperfect Metrics},
  author={Ye, Yuxuan and Santos-Rodriguez, Raul and Simpson, Edwin},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025}
}
```

---

## Contact

For questions or collaborations, please reach out to Yuxuan Ye at `yuxuan.ye@bristol.ac.uk`.

---

**Intelligent Systems Laboratory, University of Bristol**
