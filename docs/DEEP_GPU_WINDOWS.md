# Deep Learning (GPU) on Windows (RTX 4070 Ti Super)

This backend supports an optional PyTorch-based deep learning model (Transformer encoder) for training + inference.

The base install (`requirements.txt`) intentionally **does not** include PyTorch so the project stays lightweight and tests can run on CPU-only machines.

## 1) Prereqs

- Windows 10/11
- Python 3.12 (matches this repo)
- Latest NVIDIA GPU driver installed (Game Ready or Studio)

You do **not** need to install the full CUDA Toolkit separately for PyTorch; the PyTorch CUDA wheels ship the needed CUDA runtime.

## 2) Install PyTorch with CUDA

From the repo root (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1

# Option A (recommended): use PyTorch CUDA wheels via requirements-deep.txt
pip install -r requirements-deep.txt
```

If you already have a CPU-only torch installed and want GPU:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-deep.txt
```

## 3) Verify CUDA is available

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Expected: `cuda available True` and a device name like `NVIDIA GeForce RTX 4070 Ti SUPER`.

## 4) Use the deep endpoints

- Train: `POST /api/learning/train-deep`
- Status: `GET /api/learning/status-deep`
- Predict: `GET /api/learning/predict-deep`

Notes:
- Deep models are stored in SQLite `trained_models` using a key prefix `deep::...`.
- Inference prefers deep models when present; otherwise it falls back to the classic ridge model.
