# SCTG: Semantically Constrained Transfer Gradients

![](https://img.shields.io/badge/python-3.11-green)
![](https://img.shields.io/badge/pytorch-2.0.1-green)
![](https://img.shields.io/badge/cuda-11.7-green)

This repository provides a reference implementation of **SCTG (Semantically Constrained Transfer Gradients)**, a model-free domain adaptation framework for concealed multimodal LLMs. A learnable visual prompt (border perturbation) is trained on a frozen **surrogate MLLM** using only unlabeled target-domain images, then transferred and evaluated across **concealed MLLMs** (MiniGPT-4, InstructBLIP, BLIVA) without accessing their parameters, source data, or outputs during training.

## Setup

### Environment

```shell
conda create --name sctg python=3.11
conda activate sctg
pip install -r requirements.txt
```

### Model Weights

Pretrained weights are **not included** in this code package and must be downloaded separately to `./model_weights/`:

* MiniGPT-4: Follow [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and prepare `MiniGPT-4-Vicuna-7B`
* InstructBLIP: Follow [LAVIS](https://github.com/salesforce/LAVIS) and prepare `InstructBLIP-Vicuna-7b`
* BLIVA: Follow [BLIVA](https://github.com/mlpc-ucsd/BLIVA#prepare-weight) and prepare `BLIVA-Vicuna-7B`
* Vicuna v0 7B: Follow [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and prepare `vicuna-7b-v0` as LLM
* Vicuna v1.1 7B: Follow [LAVIS](https://github.com/salesforce/LAVIS) and prepare `vicuna-7b-v1.1` as LLM
* CLIP ViT-B/32: Download from [OpenAI CLIP](https://github.com/openai/CLIP)

```
model_weights/
├── ViT-B-32.pt
├── vicuna-7b-v0/
├── vicuna-7b-v1.1/
├── MiniGPT-4-Vicuna-7B/
├── InstructBLIP-Vicuna-7b/
└── BLIVA-Vicuna-7B/
```

Paths are configured in `util/*_utils.py`, `transfer_cls.py` (`--clip_weight_path`), and the YAML configs under `models/*/configs/`.

### Datasets

Only split files (`.txt`) are included. Download the full datasets separately to `./datasets/`:

* PACS: 4 domains, 7 classes (dog, elephant, giraffe, guitar, horse, house, person)
* OfficeHome: 4 domains, 65 classes of everyday objects

Run the split generation scripts (adjust internal paths if needed):

```shell
python datasets/generate_pacs_splits.py
python datasets/generate_officehome_splits.py
```

Expected directory structure under `--root ./datasets`:

```
datasets/
├── pacs/                              # split files (generated)
│   ├── art_painting_train.txt, art_painting_test.txt
│   ├── cartoon_train.txt, cartoon_test.txt
│   ├── photo_train.txt, photo_test.txt
│   └── sketch_train.txt, sketch_test.txt
├── PACS/                              # raw images
│   └── {domain}/{class}/*.jpg
├── officehome/                        # split files (generated)
│   ├── Art_train.txt, Art_test.txt
│   ├── Clipart_train.txt, Clipart_test.txt
│   ├── Product_train.txt, Product_test.txt
│   └── Real World_train.txt, Real World_test.txt
└── OfficeHome/                        # raw images
    └── {domain}/{class}/*.jpg
```

Each `.txt` contains one line per image: `<relative_path> <label_id>`, e.g., `art_painting/dog/pic_001.jpg 0`.

## Usage

### Training

```shell
CUDA_VISIBLE_DEVICES=0 python transfer_cls.py \
  --dataset $DATASET --root ./datasets --target_domain $DOMAIN \
  --model_name $SURROGATE --learning_rate $LR --fca $FCA \
  --lambda_clip 0.5 --da 1 --alpha 0.01 --epochs 5 --Nc $NC \
  --batch_size_train 8 --use_grad_noise \
  --lambda_align $ALIGN --lambda_balance $BAL --grad_noise_scale $GNS
```

Checkpoint saved to `./save/{model_name}_domain_{target_domain}_.../checkpoint_best.pth`.

### Evaluation

```shell
# Zero-shot baseline (no checkpoint)
CUDA_VISIBLE_DEVICES=0 python transfer_cls.py \
  --dataset PACS --root ./datasets --target_domain sketch \
  --model_name minigpt-4 --evaluate --batch_size_eval 32

# Concealed-model transfer (prompt learned on surrogate minigpt-4 → tested on concealed instructblip)
CUDA_VISIBLE_DEVICES=0 python transfer_cls.py \
  --dataset PACS --root ./datasets --target_domain sketch \
  --model_name instructblip --model_llm vicuna7b \
  --checkpoint ./save/.../checkpoint_best.pth --evaluate \
  --use_grad_noise --batch_size_eval 8
```

> Omit `--model_llm vicuna7b` when evaluating minigpt-4. Set `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` before InstructBLIP/BLIVA runs.

## Experiment Matrix

For each (dataset, surrogate) pair, train once then evaluate on all 3 concealed models:

| Dataset | Surrogate (train) | → Concealed |
|---------|---------------|-----------|
| PACS | minigpt-4 | minigpt-4, instructblip, bliva |
| PACS | instructblip | minigpt-4, instructblip, bliva |
| OfficeHome | minigpt-4 | minigpt-4, instructblip, bliva |
| OfficeHome | instructblip | minigpt-4, instructblip, bliva |

## Hyperparameter Reference

| Flag | Default | Description |
|:-----|:--------|:------------|
| `--dataset` | `DomainNet` | `PACS`, `OfficeHome`, or `DomainNet` |
| `--root` | `../data` | Dataset root directory |
| `--target_domain` | `clipart` | Target domain for adaptation |
| `--model_name` | `minigpt-4` | `minigpt-4`, `instructblip`, or `bliva` |
| `--model_llm` | `vicuna7b` | LLM backbone for InstructBLIP/BLIVA |
| `--learning_rate` | 1.0 | SGD learning rate |
| `--batch_size_train` | 16 | Training batch size |
| `--batch_size_eval` | 32 | Evaluation batch size |
| `--epochs` | 5 | Number of epochs |
| `--Nc` | 300 | Batches per CMPL chunk |
| `--fca` | 0.003 | Feature consistency weight |
| `--da` | 1.0 | SHOT domain adaptation weight |
| `--lambda_clip` | 0.5 | CLIP pseudo-label weight |
| `--alpha` | 0.01 | CMGA gradient shift coefficient |
| `--lambda_align` | 0.0 | CLIP cosine alignment weight |
| `--lambda_balance` | 0.0 | CLIP gradient balance factor |
| `--use_grad_noise` | off | Enable gradient noise injection |
| `--grad_noise_scale` | 1e-7 | Gradient noise scale |
| `--seed` | 42 | Random seed |
| `--evaluate` | off | Evaluation-only mode (needs `--checkpoint`) |
| `--checkpoint` | — | Checkpoint path for evaluation |

## Project Structure

```
├── transfer_cls.py              # Main script: training, CMPL pseudo-labeling, CMGA gradient alignment, evaluation
├── dataset.py                   # PACS / DomainNet / OfficeHome dataset classes with train/val/test splits
├── requirements.txt
├── datasets/
│   ├── generate_pacs_splits.py          # Generate PACS split files
│   ├── generate_officehome_splits.py    # Generate OfficeHome split files
│   ├── pacs/                            # PACS split files
│   └── officehome/                      # OfficeHome split files
├── util/
│   ├── minigpt4_utils.py        # MiniGPT-4 loader, loss, inference
│   ├── instructblip_utils.py    # InstructBLIP loader, loss, inference
│   ├── bliva_utils.py           # BLIVA loader, loss, inference
│   ├── tool.py                  # refine_classname, add_weight_decay, loss functions
│   ├── utils.py                 # pil_loader, accuracy, AverageMeter, cosine_lr
│   └── get_index.py             # Per-class prototype indices
├── models/
│   ├── minigpt4/                # MiniGPT-4 (Qformer, Blip2, LLaMA)
│   ├── instruct_blip/           # InstructBLIP (Blip2, Qformer, T5/Vicuna)
│   └── bliva/                   # BLIVA (FlanT5/Vicuna, LoRA)
└── save/                        # Checkpoint output
```
