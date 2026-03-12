# Image Reconstruction vs Texture and Masked Area

## Overview
This project studies how image reconstruction quality depends on:
1. Texture type
2. Masked area size

Datasets:
- Carpet textures
- Natural images
- Uniform regions (sky, desert, etc.)

The goal is to run controlled experiments with reproducible training and evaluation.

---

## Research Question
How does reconstruction quality change depending on:

- Image texture
- Size of the missing region

---

## Experimental Setup

Resolution:

256 × 256

Mask protocol:

- 5 blocks per image
- Block sizes: 1, 10, 20, 40, 80

Evaluation:

- 5 placements per size
- 25 masks per image

---

## Project Structure

project/
│
├── configs/
├── data/
├── runs/
│
├── models/
├── masks/
├── .gitignore
├── requirements.txt
└── README.md

---

## Environment Setup

Create virtual environment:

```python -m venv .venv```

Activate:

Windows
```.venv\Scripts\activate```

Linux / macOS
```source .venv/bin/activate```

Install dependencies:

```pip install -r requirements.txt```

---

## Dataset Setup

Example:

data/
    carpet/
    natural/
    uniform/

---

## Running Training

```python train.py```

Outputs will appear in:

`runs/`

---

## Evaluation

Each sample is evaluated with:

5 mask sizes × 5 placements = 25 masks

Metrics:

- PSNR (masked region)
- SSIM (masked region)

---

## Reproducibility

Each experiment saves:

- config
- dataset split
- seeds
- metrics
- visual outputs

---

## Future Work

- Add LaMa model
- Add diffusion models
- Add more mask distributions
- Evaluate cross-dataset generalization

```
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

set DATA_PATH=path/to/data