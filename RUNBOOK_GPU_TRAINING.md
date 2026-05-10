# GPU training setup (GTX 1650 etc.)

The current results (100% subject-disjoint accuracy on CASIA-FASD) were trained entirely on CPU using a 242K-parameter `SmallCNN`. If you want to push further with a larger backbone (ResNet-18 / MobileNetV3 / ViT) or run the entire benchmark suite end-to-end, a GPU helps significantly.

## Local GTX 1650 setup (4 GB VRAM)

The GTX 1650 is a 1650-Super-class consumer card. Adequate for FAS:
- ResNet-18 at 96×96 input, batch 64: ~2 GB VRAM, ~15 sec/epoch on CASIA-FASD train
- MobileNetV3-small at 96×96, batch 128: ~1.2 GB VRAM, ~10 sec/epoch
- ResNet-50 at 224×224, batch 32: ~3.5 GB VRAM, ~45 sec/epoch (right at the 4 GB limit)

### One-time install

```bash
# 1. Verify NVIDIA driver visible to Linux
nvidia-smi  # should show GTX 1650 + CUDA version >= 11.8

# 2. Install CUDA-enabled PyTorch (pick based on your driver's CUDA version)
# For CUDA 12.1 (modern):
pip install --break-system-packages torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older driver):
pip install --break-system-packages torch torchvision \
    --index-url https://download.pytorch.org/whl/cu118

# 3. Verify
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: cuda: True NVIDIA GeForce GTX 1650
```

### Training on GPU

The existing scripts auto-detect CUDA — no flag changes needed.

```bash
# SmallCNN (242K params) — fast even on CPU, but ~3x faster on GTX 1650
python -m tests.benchmark.train_cnn \
    --train paper/figures/captures/casia_fasd_train_crops.npz \
    --test  paper/figures/captures/casia_fasd_test_crops.npz \
    --out   paper/figures/cnn_casia_fasd_gpu.json \
    --epochs 50 --batch-size 128 --lr 1e-3 --arch small

# ResNet-18 (11M params, GPU recommended)
python -m tests.benchmark.train_cnn \
    --train paper/figures/captures/casia_fasd_train_crops.npz \
    --test  paper/figures/captures/casia_fasd_test_crops.npz \
    --out   paper/figures/resnet18_casia_fasd_gpu.json \
    --epochs 60 --batch-size 64 --lr 5e-4 --arch resnet18

# MobileNetV3-small (2.5M params, fast inference)
python -m tests.benchmark.train_cnn \
    --train paper/figures/captures/casia_fasd_train_crops.npz \
    --test  paper/figures/captures/casia_fasd_test_crops.npz \
    --out   paper/figures/mobilenetv3_casia_fasd_gpu.json \
    --epochs 50 --batch-size 128 --lr 1e-3 --arch mobilenetv3
```

### What we expect from a bigger model

The `SmallCNN` is overfitting on the 1655-frame train set (val AUC = 1.0 from epoch 9). Larger models trained with the same data may not improve raw test accuracy — the limit is data quantity, not model capacity. **Where bigger helps:**

1. **Cross-dataset transfer**: ResNet-18 pretrained on ImageNet (set `weights="DEFAULT"` in the model factory) transfers better to OULU-NPU / SiW after fine-tuning.
2. **Larger inputs**: 224×224 captures more facial detail vs 96×96. Use `--crop-size 224` when extracting and bump `train_cnn.py`'s tensor preprocessing.
3. **Augmentation budget**: GPU lets you afford heavier augmentation (random erasing, mixup, color jitter at full strength) without throughput loss.

### Throughput estimates on your GTX 1650

| Architecture | Params | Input | Epoch time | 30-epoch run |
|---|---:|---|---:|---:|
| SmallCNN (CPU baseline) | 242K | 96×96 | 10 s | 5 min |
| **SmallCNN on GTX 1650** | 242K | 96×96 | **~3 s** | **~1.5 min** |
| MobileNetV3-small on GTX 1650 | 2.5M | 96×96 | ~10 s | ~5 min |
| MobileNetV3-small on GTX 1650 | 2.5M | 224×224 | ~25 s | ~12 min |
| ResNet-18 on GTX 1650 | 11M | 96×96 | ~15 s | ~8 min |
| ResNet-18 on GTX 1650 | 11M | 224×224 | ~45 s | ~22 min |

### Memory tips for 4 GB VRAM

If you hit `CUDA out of memory` on ResNet-18 at 224×224:

```bash
# Halve batch size or use gradient accumulation
--batch-size 16

# Enable mixed precision (reduces VRAM ~40%)
# See pytorch's torch.amp.autocast — currently not in train_cnn.py.
# Add `with torch.amp.autocast(device_type="cuda"):` around the forward+loss.
```

## Cloud GPU (if more compute needed)

| Provider | Instance | GPU | $/hr | Reasonable for |
|---|---|---|---:|---|
| **Hetzner** | GEX44 | RTX 4000 (8 GB) | ~$0.40 | full benchmark suite end-to-end |
| **Vast.ai** | community | RTX 3090 (24 GB) | ~$0.20 | retraining ResNet-50 / ViT-B |
| **Lambda Labs** | A10G | A10 (24 GB) | ~$0.50 | reproducing CDCN / FAS-SGTD |
| **Google Colab Free** | — | T4 (16 GB) | $0 | one-off experiments |
| **Google Colab Pro** | — | A100 (40 GB) | $10/mo | sustained large-model training |

## Reproducibility on GPU

Set seeds for determinism:

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Note: `cudnn.deterministic=True` slows training ~10–20% on GTX 1650. For paper numbers (where reproducibility matters more than speed) it's the right tradeoff.

## What to run first on your GTX 1650

```bash
# 1. Verify install
nvidia-smi
python -c "import torch; assert torch.cuda.is_available()"

# 2. Re-run the existing winning recipe on GPU (sanity check)
python -m tests.benchmark.train_cnn \
    --train paper/figures/captures/casia_fasd_train_crops.npz \
    --test  paper/figures/captures/casia_fasd_test_crops.npz \
    --out   paper/figures/cnn_casia_fasd_gpu_seed42.json \
    --epochs 30 --seed 42
# Should reproduce the 100% on subjects 21-30

# 3. Try ResNet-18 for cross-dataset transfer
python -m tests.benchmark.train_cnn \
    --train paper/figures/captures/casia_fasd_train_crops.npz \
    --test  paper/figures/captures/casia_fasd_test_crops.npz \
    --out   paper/figures/resnet18_casia_fasd_gpu.json \
    --epochs 60 --batch-size 64 --arch resnet18

# 4. Run leave-one-subject-out CV (20 folds, ~3 min on GTX 1650 vs ~10 min on CPU)
python -m tests.benchmark.loso_cv \
    --train-analyzers paper/figures/captures/casia_fasd_train_ensemble.json \
    --test-analyzers  paper/figures/captures/casia_fasd_test_ensemble.json \
    --train-crops     paper/figures/captures/casia_fasd_train_crops.npz \
    --test-crops      paper/figures/captures/casia_fasd_test_crops.npz \
    --out paper/figures/loso_cv_gpu.json \
    --cnn-epochs 30
```
