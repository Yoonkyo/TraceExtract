# `infer.py` Output Structure

This document describes everything that `infer.py` writes to disk and the
exact meaning, shape, dtype, and value range of each field. The relevant
saving code lives in `save_structured_data` (`infer.py`).

---

## 1. On-disk layout

When `infer.py` finishes processing a video, the result for that video is
written under:

```
<output_dir>/
└── <video_name>/
    ├── images.npy              (T, H, W, 3) uint8 — all RGB frames
    ├── depth.npy               (T, H, W)    float16 — metric depth in meters
    ├── cameras.npz             per-video intrinsics + extrinsics
    ├── acceleration.npz        per-frame acceleration
    ├── movement_statistics.npz keypoint displacement stats
    ├── description.txt         language instruction(s), one per line (agibot / droid / egodex)
    ├── meta.json               source-dataset metadata (agibot / droid / egodex)
    └── samples/
        ├── frame_indices.npy   (F,)   int32   — which frames have data
        ├── offsets.npy          (F+1,) int64   — row boundaries in arrays below
        ├── keypoints.npy       (N_total, 2)            float16
        ├── cluster_ids.npy     (N_total,)              int32
        ├── is_moving.npy       (N_total,)              bool
        ├── visibs.npy          (N_total,)              float16
        ├── traj.npy            (N_total, future_len, 3) float16
        ├── traj_history.npy    (N_total, history_len, 3) float16
        ├── valid_steps.npy     (N_total, future_len)   bool
        ├── valid_steps_history.npy (N_total, history_len) bool
        ├── raw_traj.npy        (N_total, future_len, 3) float16
        ├── raw_traj_history.npy(N_total, history_len, 3) float16
        ├── raw_valid_steps.npy (N_total, future_len)   bool
        └── raw_valid_steps_history.npy (N_total, history_len) bool
```

### Top-level files

- `images.npy` — All RGB frames of the **full** video, stored as a single
  NumPy array of shape `(T, H, W, 3)` with dtype `uint8`. Load with
  `np.load("images.npy", mmap_mode="r")` for lazy, memory-mapped access.
- `depth.npy` — Per-frame metric depth from the depth/pose model. Shape
  `(T, H, W)`, dtype `float16`, values in **meters**. Supports mmap.
- `cameras.npz` — Per-frame camera parameters for the **full** video.
  Loaded with `np.load(...)`, exposes:
  - `intrinsics`: `(T, 3, 3)` `float32` — pinhole `K` per frame.
  - `extrinsics`: `(T, 4, 4)` `float32` — **world-to-camera** matrix
    per frame. Invert it to get camera-to-world.
  - `height`: `int32` — image height.
  - `width`: `int32` — image width.

### `samples/` directory

All trajectory data is stored as **unified `.npy` files**. The keypoints
from all frames are concatenated along the first axis (`N_total`), and an
offset array records where each frame's data begins and ends.

- `frame_indices.npy` — `(F,)` `int32`. The list of frame indices `t` that
  have sample data. `F` is the number of frames with at least one valid
  keypoint. Not every frame has data (frames near the end of the video with
  no future trajectory are skipped).
- `offsets.npy` — `(F+1,)` `int64`. For the `i`-th entry in
  `frame_indices`, the keypoints for that frame are rows
  `offsets[i] : offsets[i+1]` in all the arrays below.

All other files share the row dimension `N_total` (total keypoints across
all frames):

| File | Shape | dtype | Description |
|---|---|---|---|
| `keypoints.npy` | `(N_total, 2)` | `float16` | Pixel positions `(x, y)` at each track's frame. |
| `cluster_ids.npy` | `(N_total,)` | `int32` | DINO cluster ID. |
| `is_moving.npy` | `(N_total,)` | `bool` | True if the track was classified as moving. |
| `visibs.npy` | `(N_total,)` | `float16` | Visibility at the track's frame (0/1). |
| `traj.npy` | `(N_total, future_len, 3)` | `float16` | Forward trajectory, arc-length retargeted. Padded with `-inf`. |
| `traj_history.npy` | `(N_total, history_len, 3)` | `float16` | Backward trajectory, arc-length retargeted. Padded with `-inf`. |
| `valid_steps.npy` | `(N_total, future_len)` | `bool` | Per-track mask for `traj`. |
| `valid_steps_history.npy` | `(N_total, history_len)` | `bool` | Per-track mask for `traj_history`. |
| `raw_traj.npy` | `(N_total, future_len, 3)` | `float16` | Forward trajectory **before retargeting**, frame-aligned. Padded with `-inf`. |
| `raw_traj_history.npy` | `(N_total, history_len, 3)` | `float16` | Backward trajectory before retargeting. Padded with `-inf`. |
| `raw_valid_steps.npy` | `(N_total, future_len)` | `bool` | Per-track mask for `raw_traj`. |
| `raw_valid_steps_history.npy` | `(N_total, history_len)` | `bool` | Per-track mask for `raw_traj_history`. |

`future_len` is a CLI argument (`--future_len`, default **128**) and is the
fixed second dimension of `traj` / `valid_steps`. `history_len` is a
separate CLI argument (`--history_len`, default **32**).

---

## Migration from legacy per-frame format

> If you have a dataloader or script that was built for the **previous
> output format** (`images/*.png`, `depth/*.png`, `samples/*.npz`), this
> section explains what changed and how to update.

### What changed

| | Legacy format | New format |
|---|---|---|
| RGB frames | `images/<video>_<t>.png` (one PNG per frame) | `images.npy` — `(T, H, W, 3)` uint8 |
| Depth | `depth/<video>_<t>.png` (uint16, `depth * 10000`) | `depth.npy` — `(T, H, W)` float16, meters |
| Samples | `samples/<video>_<t>.npz` (one file per frame) | `samples/*.npy` — unified arrays + offset index |
| `is_task_relevant` field | Present in each `.npz` | **Removed** (VLM filter no longer used) |
| `image_path` field | Present in each `.npz` (`"images/<video>_<t>.png"`) | **Removed** (use `frame_indices.npy` instead) |

### Migrating your dataloader

**Images / Depth** — Replace per-frame file opens with mmap slicing:

```python
# OLD
image = np.array(Image.open(f"images/{video}_{t}.png"))
depth = np.array(Image.open(f"depth/{video}_{t}.png")).astype(np.float32) / 10000.0

# NEW
images = np.load("images.npy", mmap_mode="r")  # open once, reuse across frames
image  = np.array(images[t])                     # (H, W, 3) uint8

depths = np.load("depth.npy", mmap_mode="r")
depth  = np.array(depths[t]).astype(np.float32)  # (H, W) stored as float16, cast to float32
```

**Samples** — Replace per-frame `.npz` loads with offset-based slicing:

```python
# OLD
data = np.load(f"samples/{video}_{t}.npz")
keypoints = data["keypoints"]     # (N, 2)
traj      = data["traj"]          # (N, future_len, 3)
frame_idx = data["frame_index"]   # (1,)

# NEW
frame_indices = np.load("samples/frame_indices.npy")      # (F,)
offsets       = np.load("samples/offsets.npy")             # (F+1,)
slot = np.where(frame_indices == t)[0][0]
lo, hi = int(offsets[slot]), int(offsets[slot + 1])

keypoints = np.load("samples/keypoints.npy", mmap_mode="r")[lo:hi]  # (N_t, 2)
traj      = np.load("samples/traj.npy",      mmap_mode="r")[lo:hi]  # (N_t, future_len, 3)
frame_idx = t
```

**`is_task_relevant`** — If your code filters on this field, either drop the
filter or replace it with `True`:

```python
# OLD
keep = data["is_moving"] & data["is_task_relevant"]

# NEW
keep = data["is_moving"]
```

### Backward-compatible loading

If your code needs to work with **both** old and new outputs, use the
helpers in `utils/output_format.py`. They auto-detect the format:

```python
from utils.output_format import load_sample, load_image, load_depth

data  = load_sample(video_dir, video_name, frame_idx=42)   # dict, same keys
image = load_image(video_dir, video_name, frame_idx=42)     # (H, W, 3) uint8
depth = load_depth(video_dir, video_name, frame_idx=42)     # (H, W) float32
```

These return the same data shapes regardless of which format is on disk.

---

## 2. How to access per-frame data from `samples/`

All per-frame arrays (keypoints, trajectories, masks, etc.) are
**concatenated** along the first axis across all frames. The two index
files tell you which rows belong to which frame:

```
frame_indices = [10, 15, 20, 25, ...]   # F frames that have data
offsets       = [ 0, 48, 120, 185, 240, ...]  # F+1 boundary indices
```

Frame `frame_indices[i]` owns rows `offsets[i]` to `offsets[i+1]` (exclusive)
in every array. The number of keypoints per frame (`offsets[i+1] - offsets[i]`)
varies because different frames are covered by different tracking segments.

### 2a. Load a single frame

```python
import numpy as np

video_dir = "output/<video_name>"
samples_dir = f"{video_dir}/samples"

# Load the two index arrays (small, fast)
frame_indices = np.load(f"{samples_dir}/frame_indices.npy")   # (F,)
offsets       = np.load(f"{samples_dir}/offsets.npy")          # (F+1,)

# Look up frame t=42
t = 42
slot = np.where(frame_indices == t)[0][0]   # position in frame_indices
lo, hi = int(offsets[slot]), int(offsets[slot + 1])
N_t = hi - lo   # number of keypoints at this frame

# Slice into the concatenated arrays (mmap avoids loading everything)
keypoints = np.load(f"{samples_dir}/keypoints.npy", mmap_mode="r")[lo:hi]  # (N_t, 2)
traj      = np.load(f"{samples_dir}/traj.npy",      mmap_mode="r")[lo:hi]  # (N_t, future_len, 3)
mask      = np.load(f"{samples_dir}/valid_steps.npy",mmap_mode="r")[lo:hi]  # (N_t, future_len)

# Load the corresponding RGB frame (also mmap)
images = np.load(f"{video_dir}/images.npy", mmap_mode="r")
image  = images[t]   # (H, W, 3) uint8
```

### 2b. Iterate over all frames

```python
frame_indices = np.load(f"{samples_dir}/frame_indices.npy")
offsets       = np.load(f"{samples_dir}/offsets.npy")

# Open mmap handles once (no data is read yet)
all_keypoints = np.load(f"{samples_dir}/keypoints.npy", mmap_mode="r")
all_traj      = np.load(f"{samples_dir}/traj.npy",      mmap_mode="r")

for i, t in enumerate(frame_indices):
    lo, hi = int(offsets[i]), int(offsets[i + 1])
    kp   = all_keypoints[lo:hi]   # (N_t, 2)
    traj = all_traj[lo:hi]        # (N_t, future_len, 3)
    # ... process frame t ...
```

### 2c. Using the helper in `utils/output_format.py`

```python
from utils.output_format import load_sample, load_image, load_depth

data  = load_sample(video_dir, video_name, frame_idx=42)
# data is a dict: {"keypoints": ..., "traj": ..., "valid_steps": ..., ...}

image = load_image(video_dir, video_name, frame_idx=42)  # (H, W, 3) uint8
depth = load_depth(video_dir, video_name, frame_idx=42)  # (H, W) float32 meters
```

The `load_*` functions auto-detect whether the directory uses the unified
`.npy` format or the legacy per-frame PNG/NPZ format, so they work with
both old and new outputs.

---

## Lazy loading / dataloader tips

> **Note:** This section is just a helper for writing lazy dataloaders that
> need to efficiently iterate over many frames or many videos in parallel.
> If you already have a dataloader design that works well for your use case,
> or you are simply loading a single episode at a time (e.g. for
> visualization or debugging), you can ignore this section entirely and
> just use the `load_sample` / `load_image` helpers or plain `np.load`
> as shown in section 2.

All `.npy` files are saved with `np.save`, so they can be memory-mapped
with `np.load(..., mmap_mode="r")`. This means the OS pages in only the
bytes you actually read — opening a 2 GB `images.npy` is instant and costs
near-zero RAM until you slice into it.

### Open handles once in `__init__`, slice in `__getitem__`

```python
class TraceExtractDataset(torch.utils.data.Dataset):
    def __init__(self, video_dir):
        samples_dir = os.path.join(video_dir, "samples")

        # These are instant — no data is read yet
        self.images     = np.load(os.path.join(video_dir, "images.npy"), mmap_mode="r")
        self.depths     = np.load(os.path.join(video_dir, "depth.npy"),  mmap_mode="r")
        self.frame_indices = np.load(os.path.join(samples_dir, "frame_indices.npy"))
        self.offsets       = np.load(os.path.join(samples_dir, "offsets.npy"))
        self.keypoints  = np.load(os.path.join(samples_dir, "keypoints.npy"),  mmap_mode="r")
        self.traj       = np.load(os.path.join(samples_dir, "traj.npy"),       mmap_mode="r")
        self.valid_steps= np.load(os.path.join(samples_dir, "valid_steps.npy"),mmap_mode="r")
        # ... open other arrays as needed ...

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        t  = int(self.frame_indices[idx])
        lo = int(self.offsets[idx])
        hi = int(self.offsets[idx + 1])

        # Only these slices hit disk (OS page cache makes repeated access fast)
        image = np.array(self.images[t])             # (H, W, 3) uint8
        depth = np.array(self.depths[t]).astype(np.float32)  # stored float16, cast up
        kp    = np.array(self.keypoints[lo:hi])      # (N_t, 2)
        traj  = np.array(self.traj[lo:hi])           # (N_t, future_len, 3)
        mask  = np.array(self.valid_steps[lo:hi])    # (N_t, future_len)
        return image, depth, kp, traj, mask
```

### Practical notes

- **`np.array(mmap_slice)`**: Slicing an mmap returns a *view* that still
  points at the file. Wrap with `np.array(...)` to copy into RAM when you
  need a regular tensor (e.g. before `torch.from_numpy`). Without the copy,
  PyTorch's `DataLoader` with `num_workers > 0` can hit "cannot pickle
  mmap" errors.
- **`num_workers`**: numpy mmap handles are safe to share across forked
  workers (read-only file descriptors survive `fork()`). Each worker's page
  faults are independent and benefit from the shared OS page cache.
- **`float16` arrays**: `traj`, `keypoints`, `visibs` are stored as
  `float16` to halve file size. Cast to `float32` before feeding to your
  model: `traj = np.array(self.traj[lo:hi]).astype(np.float32)`.
- **`-inf` in float16**: The `-inf` padding sentinel is preserved under
  `float16` (IEEE 754 half-precision has infinity). After casting to
  `float32`, `-inf` stays `-inf`. Combine with `valid_steps` mask to skip
  padded slots.
- **Multi-video dataset**: For a dataset spanning many videos, open all
  mmap handles in `__init__` and store them in a list indexed by video.
  The OS page cache handles the rest — even hundreds of mmap files are
  fine since no data is resident until accessed.

---

## 3. What `traj` and `traj_history` actually contain

For a single frame `t`:

- `traj[i]` is the **future trajectory** of track `i`, expressed in the
  camera coordinate frame of frame `t`. Index 0 along the time axis is the
  position at frame `t` itself; subsequent indices walk forward in time.
- `traj_history[i]` is the **past trajectory** of track `i`, also expressed
  in frame-`t` camera coordinates, but **time-reversed**. Index 0 is the
  position at frame `t`; subsequent indices walk *backward* in time.
- Both are 3D points in the form `(x_pixel, y_pixel, z_depth)`.

### Coordinate convention

The 3D tracks are reprojected into frame `t`'s camera before being saved:

1. The track's world-space 3D position is transformed into frame `t`'s
   camera coordinate frame.
2. It is projected to a 2D pixel `(x, y)` by frame `t`'s intrinsics `K`.
3. The track's metric depth `z` (the camera-space Z, in meters) is appended.

So a single entry `traj[i, k] = (x, y, z)` means: "If you stood at frame
`t`'s camera and looked at track `i`'s position at the k-th time step, it
would project to pixel `(x, y)` and lie at depth `z` meters in front of the
camera."

### Value ranges

| Channel | Range | Notes |
|---|---|---|
| `traj[..., 0]` (x) | pixels, nominally `[0, image_width]` | Not clipped: points that leave the frame keep off-screen values. |
| `traj[..., 1]` (y) | pixels, nominally `[0, image_height]` | Not clipped: points that leave the frame keep off-screen values. |
| `traj[..., 2]` (z) | metric depth in meters | **Not clipped, not normalized.** |
| `-inf` rows | sentinel | Any time step where the track is not active. |

> **Important:** The pixel coordinates are in the *original image
> resolution*, not normalized to `[0, 1]`.

---

## 4. `-inf` padding and the per-track validity masks

`traj` and `traj_history` are *not* densely populated. Each row has a
finite prefix and an `-inf` tail:

```
traj[i] = [(x0,y0,z0), ..., (xL,yL,zL), (-inf,-inf,-inf), ...]
           └── valid_steps[i] True ──┘└── valid_steps[i] False ─┘
```

The truth source for "is this slot real?" is the boolean mask:

```python
traj = np.load("samples/traj.npy", mmap_mode="r")[lo:hi]
mask = np.load("samples/valid_steps.npy", mmap_mode="r")[lo:hi]

valid_points = traj[i][mask[i]]   # (L_i, 3) — only real samples
```

Always combine the boolean mask with a finiteness check before doing math
that propagates NaN/inf:

```python
finite = np.isfinite(traj[i]).all(axis=1)
keep   = mask[i] & finite
pts    = traj[i][keep]
```

For `raw_traj` / `raw_traj_history`, also require positive depth
(`traj[i][:, 2] > 0`): in rare cases a point that passes behind the camera
keeps a `True` mask with a non-positive depth or a non-finite projection.

---

## 5. Trajectory retargeting

**`traj[i, k]` is not the position of track `i` at frame `t + k`**. The
time axis is an *arc-length* axis, not a frame axis.

`retarget_trajectories` replaces the frame-aligned representation with an
**arc-length-aligned** one. Downstream consumers (e.g. action models) often
want samples that are roughly equidistant in space, not equidistant in time.

If you need a strict frame-aligned representation, use `raw_traj` /
`raw_traj_history` — they preserve the upstream-tracker output before any
retargeting.

### `raw_traj` / `raw_traj_history`

| Property | Retargeted (`traj`) | Raw (`raw_traj`) |
|---|---|---|
| Time axis | Equal arc-length steps | Real video frame offsets |
| Slot `k` | k-th arc-length sample | Frame `t + k` (future) or `t - k` (history) |
| x, y range | Not clipped — off-screen values possible | Not clipped — off-screen values possible |
| Padding | `-inf` in trailing slots | `-inf` in trailing slots |
| Time dim | `future_len` (constant) | `future_len` (padded to match) |

---

## 6. Visualization scripts

All viewers support both the unified `.npy` format and the legacy per-frame
PNG/NPZ format via auto-detection in `utils/output_format.py`.

```bash
# New format (recommended):
python visualize_single_image.py \
    --video_dir <output_root>/<video> \
    --frame_index <frame>

# Legacy format:
python visualize_single_image.py \
    --npz_path   <output_root>/<video>/samples/<video>_<frame>.npz \
    --image_path <output_root>/<video>/images/<video>_<frame>.png \
    --depth_path <output_root>/<video>/depth/<video>_<frame>.png

# Batch checker (auto-detects format):
python checker/batch_process_result_checker_3d.py <output_root>
```

Add `--raw` to use `raw_traj` instead of the retargeted `traj`.
Add `--visualize_history` to view past trajectories.

---

## 7. Quick consumption recipe

```python
import numpy as np
from utils.output_format import load_sample

data = load_sample("output/<video>", "<video>", frame_idx=42)
N = data["keypoints"].shape[0]

for i in range(N):
    if not data["is_moving"][i]:
        continue

    mask   = data["valid_steps"][i]
    finite = np.isfinite(data["traj"][i]).all(axis=1)
    pts    = data["traj"][i][mask & finite]   # (L_i, 3)

    # pts[:, 0:2] are pixel coords in the frame-42 image
    # pts[:,   2] is metric depth (meters) from the frame-42 camera
    ...
```

For frame-aligned access, swap `traj` -> `raw_traj` and `valid_steps` ->
`raw_valid_steps`.
