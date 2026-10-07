#!/bin/bash
# Install TraceExtract dependencies into the currently active (conda) Python env.
# Building pointops2 needs a CUDA toolkit (nvcc) matching PyTorch's CUDA 12.8.
set -e
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# PyTorch (CUDA 12.8). Comment this out to keep an existing compatible build.
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# ffmpeg binary: video inputs (.mp4/.webm/...) are decoded through it.
if ! command -v ffmpeg >/dev/null 2>&1; then
    if [ -n "${CONDA_PREFIX:-}" ]; then
        "${CONDA_EXE:-conda}" install -y -p "$CONDA_PREFIX" -c conda-forge ffmpeg
    else
        echo "WARNING: ffmpeg not found; install it (e.g. 'apt install ffmpeg') to read video files." >&2
    fi
fi

pip install git+https://github.com/EasternJournalist/utils3d.git@fb135440dc5eb327805a6e377f3a6e3f9c9edb7d

# Core pipeline (depth/pose, 3D tracking, DINO keypoints, I/O)
pip install kornia==0.8.1 huggingface_hub hydra-core omegaconf \
    timm einops jaxtyping "python-box[all]~=7.0" \
    opencv-python-headless Pillow matplotlib mediapy av imageio scipy scikit-learn \
    sophuspy tqdm rich loguru flow_vis moviepy==1.0.0 easydict

# Dataset loaders: h5py (EgoDex sidecars), zstandard (EgoVerse Zarr shards)
pip install h5py zstandard

# Visualization (visualize_single_image.py)
pip install viser

# Optional caption generation (see caption/README.md)
pip install python-dotenv openai google-genai

# pointops2 CUDA extension (KNN queries in the 3D tracker). Compiled for every GPU
# generation PyTorch 2.8 (cu128) supports, not only the build machine's GPU, so the
# env also runs on other GPU types. Override by exporting TORCH_CUDA_ARCH_LIST.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX}"
cd third_party/pointops2
python setup.py install
