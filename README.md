# VL-JEPA: Vision-Language Joint Embedding Predictive Architecture

Research prototype implementing a JEPA-style self-supervised learning model for vision-language alignment under GPU-constrained environments.

## Model Overview

VL-JEPA learns joint representations of images and text by predicting properties of masked inputs in a latent feature space. Unlike contrastive methods (CLIP) that rely on data augmentation and negative sampling, VL-JEPA uses a predictive objective to regress representation of masked regions conditioned on unmasked context.

### Key Differences from Contrastive Learning
- **No Negative Sampling:** Eliminates the need for large batch sizes or memory banks.
- **Predictive Objective:** Optimizes feature regresssion directly in latent space, encouraging semantic understanding rather than just instance discrimination.
- **Data Efficiency:** Dense gradients from masked modeling generally provide richer signal per sample.

## Architecture

- **Vision Encoder:** Vision Transformer (ViT-Base/16), initialized from pre-trained weights.
- **Text Encoder:** BERT-Base, providing semantic conditioning.
- **Predictor:** Lightweight Transformer decoder that predicts latent representations of masked image patches.
- **Target Encoders:** Exponential Moving Average (EMA) copies of the online vision/text encoders to provide stable targets.

## Training Setup

### Constraints & Optimization
- **Hardware:** Single Consumer GPU (4GB+ VRAM).
- **Batch Size:** 1 (Gradient accumulation simulated via high-frequency updates or small subset).
- **Frozen Encoders:** To fit in memory, the backbone ViT and BERT are frozen. Only the Predictor and a specific learnable mask token are optimized.
- **Optimization:** AdamW optimizer, mixed-precision (AMP) enabled.

### Flow
1. Image patches and text tokens are embedded by frozen Online Encoders.
2. Random block masking is applied to vision patches (e.g. 60% masked).
3. Predictor takes visible patches + text + mask tokens.
4. Predictor output at mask positions is compared to Target Encoder output (Cosine Similarity Loss).
5. Gradients update Predictor; EMA updates Target Encoders.

## Limitations

- **Frozen Capacity:** Since backbones are frozen, the model cannot adapt the core feature space, only the prediction mechanism. Limits downstream fine-tuning potential.
- **Dataset Size:** Trained on a small subset of MS COCO (5k samples) for prototyping velocity.
- **Compute:** Hyperparameters (epochs, batch size) are significantly scaled down from literature standards.

## Usage

### Training
```bash
python training/train_jepa.py
```

### Linear Probe Evaluation
```bash
python training/train_linear_probe.py
```

### Ablations
```bash
python training/run_ablations.py
```
