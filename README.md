# MINT — Protein Interaction Demo

A Streamlit demo for the **MINT (Multimeric INteraction Transformer)** model.  
Predicts whether two proteins physically interact, and lets you simulate the effect of a single amino-acid mutation.

**Live demo:** https://mint-interaction-demo.hf.space

---

## What it does

### Interaction Predictor
Select a pre-loaded protein pair (TP53+MDM2, EGFR+GRB2, PCNA+p21, MYC+MAX, BRCA1+BARD1) and click **Show prediction** to see the MINT interaction score.  
Each positive pair is compared against its scrambled-sequence negative control.

### In-silico Mutagenesis
After a prediction, click **Introduce Missense Mutation** to apply a single amino-acid substitution to Protein A.  
The score updates instantly, showing how one residue change can abolish binding (e.g. TP53 W23A: 88.2% → 12.0%).

### Live Mode
Uncheck demo mode to input any arbitrary protein sequence and run real MINT inference (~60 s on CPU).  
On the cloud deployment, the model is downloaded automatically on first use (~3 min), then cached for all subsequent users.

---

## Model

| Component | Details |
|---|---|
| Backbone | ESM2-650M (Facebook Research) |
| Architecture | Multimeric INteraction Transformer (MINT) |
| Classifier | Bernett MLP (Linear 2560→2560→1) |
| Training data | 96 M protein–protein interactions (STRING) |
| Paper | Varun Ullanat et al. — [github.com/VarunUllanat/mint](https://github.com/VarunUllanat/mint) |

Checkpoints are hosted on HuggingFace: [varunullanat2012/mint](https://huggingface.co/varunullanat2012/mint)

---

## Local setup

```powershell
# 1. Clone this repo
git clone https://github.com/plastic0105/mint-interaction-demo.git
cd mint-interaction-demo

# 2. Create conda environment
conda env create -f env.yml
conda activate mint-demo

# 3. Download model checkpoints (~3.3 GB total)
python download_checkpoints.py

# 4. Generate precomputed embeddings (optional, data/pairs.csv already included)
python precompute.py

# 5. Launch app
streamlit run app.py
```

---

## Repository structure

```
mint-interaction-demo/
├── app.py                    — Streamlit app (main entry point)
├── precompute.py             — Generates data/ from MINT model (run once)
├── download_checkpoints.py   — Downloads mint.ckpt and bernett_mlp.pth
├── generate_qr.py            — Generates QR code PNG for any URL
├── Dockerfile                — For HuggingFace Spaces deployment
├── requirements.txt          — Python dependencies
├── env.yml                   — Conda environment (Python 3.9, PyTorch 1.12 CPU)
├── setup.ps1                 — One-click local setup (Windows)
├── data/
│   └── pairs.csv             — 10 protein pairs with pre-computed MINT scores
└── mint_repo/mint/           — MINT source code (with compatibility fixes)
```

---

## Deployment

The app is deployed on **HuggingFace Spaces** (Docker, 16 GB RAM) at:  
https://huggingface.co/spaces/mint-interaction/demo

---

## Fixes applied to upstream MINT code

1. `mint_repo/mint/helpers/extract.py` — added missing `import random`
2. `mint_repo/mint/helpers/extract.py` — `torch.load` wrapped in try/except for PyTorch 2.6 compatibility
3. `env.yml` — pinned `numpy<2` (NumPy 2.x incompatible with PyTorch 1.12)
