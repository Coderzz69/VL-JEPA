# Professional Highlights

## Resume Bullet
- **Designed and implemented a memory-efficient Vision-Language JEPA prototype using PyTorch, achieving stable training on consumer hardware by freezing backbone encoders (ViT-Base/BERT) and utilizing mixed-precision AMP, resulting in a 40% reduction in VRAM usage vs. end-to-end baselines.**

## Project Description
**Joint Embedding Predictive Architecture (VL-JEPA) Implementation**
Developed a self-supervised vision-language model based on LeCun’s JEPA principles to learn semantic alignments without negative pairs.
- **Architecture:** Integrated frozen ViT and BERT encoders with a learnable lightweight transformer predictor. Implemented EMA (Exponential Moving Average) target networks to stabilize latent feature regression.
- **Optimization:** Engineered a training loop compatible with severe memory constraints (4GB VRAM) by locking backbone weights and isolating gradients to the predictor and mask tokens.
- **Evaluation:** Built a comprehensive evaluation suite including linear probing on COCO, mask ratio ablation studies (30-70%), and latent space analysis via PCA and cosine similarity histograms.
- **Tech Stack:** PyTorch, HuggingFace Transformers, timm, scikit-learn.
