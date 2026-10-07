# Checkpoints

Model weights are **not** bundled with this release. Only the TAPIP3D tracker
checkpoint must be downloaded manually; the other weights (VGGT4Track, DINOv2,
CoTracker3) are fetched automatically on first run via Hugging Face / `torch.hub`
(about 5 GB in total, mostly VGGT4Track), so the first run needs internet access.

## Required: TAPIP3D tracker (~296 MB)
```bash
wget -O checkpoints/tapip3d_final.pth \
  https://huggingface.co/zbww/tapip3d/resolve/main/tapip3d_final.pth
```
This is the default for `--checkpoint` in `infer.py`.

## Auto-downloaded on first run (no action needed)
- VGGT4Track (depth + camera pose) — Hugging Face `Yuxihenry/SpatialTrackerV2_Front`
- DINOv2 ViT-B/14 (keypoint features) — `torch.hub` (`facebookresearch/dinov2`)
- CoTracker3 offline (initializes the tracker's CNN encoder before the TAPIP3D
  checkpoint is loaded) — `torch.hub` (`facebookresearch/co-tracker`)

`torch.hub` caches code and weights under `$TORCH_HOME/hub` (default
`~/.cache/torch/hub`); Hugging Face caches under `$HF_HOME` (default
`~/.cache/huggingface`).
