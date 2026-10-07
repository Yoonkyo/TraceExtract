# Long-Video Consistent 3D Tracking Pipeline

This document explains the long-video pipeline in TraceExtract, which handles arbitrarily long videos without VRAM limits or temporal inconsistency.

## Motivation

The standard pipeline runs VGGT (depth + pose estimation) on the entire video at once, which is bounded by GPU memory. For long videos this fails with OOM errors. A naive fix — chopping the video into independent chunks — introduces coordinate drift: each chunk lives in its own coordinate frame, so 3D tracks don't stitch together.

The long-video pipeline solves both problems with two components:

1. **Hybrid chunked VGGT** — globally consistent depth and poses within bounded VRAM.
2. **Progressive keypoint tracking** — seamless 3D tracks across chunk boundaries.

---

## Pipeline Overview

```
Full video (T frames)
  │
  ├─ Phase 1: Load all frames to CPU
  │
  ├─ Phase 2: Chunked VGGT (depth + camera poses)
  │    ├─ Global sparse pass (≤ sparse_max uniformly sampled frames)
  │    └─ Per-chunk dense passes, each SE3-aligned to global anchors
  │
  ├─ Phase 3: Progressive tracking (chunk by chunk)
  │    ├─ Chunk 0: DINO keypoints → TAPIP3D tracking
  │    └─ Chunk N≥1: propagate existing tracks → DINO new keypoints
  │         → proximity filter → track new keypoints → merge
  │
  ├─ Phase 4: Assemble global trajectories
  │
  └─ Phase 5: Motion detection (is_moving per keypoint)
```

---

## Phase 2: Hybrid Chunked VGGT

### Problem

VGGT processes a `(1, T, 3, H, W)` tensor and returns per-frame depth maps, camera extrinsics (camera-to-world), and intrinsics. Processing all T frames at once exceeds VRAM for long videos.

### Solution: Global Sparse + Local Dense

```
Full video: [frame 0] [frame 1] ... [frame T-1]
                |          |                |
Sparse pass:  [f0]      [f15]    ...     [f_{T-1}]     ← ~sparse_max frames
                ↓          ↓                ↓
         Globally consistent anchor poses (extrinsics)

Dense pass:  [chunk 0: f0..f19] [chunk 1: f20..f39] [chunk 2: f40..f59] ...
                    ↓                   ↓                   ↓
              chunk-local          chunk-local          chunk-local
              depth+poses          depth+poses          depth+poses
                    ↓                   ↓                   ↓
              SE3 align to         SE3 align to         SE3 align to
              global anchors       global anchors       global anchors
```

#### Step 1: Global Sparse Pass

Uniformly subsample up to `sparse_max` frames from the full video. Run VGGT once on these frames. This yields globally consistent anchor extrinsics — all in the same coordinate frame because VGGT sees them together.

#### Step 2: Per-Chunk Dense Passes

Split the full video into non-overlapping chunks of `chunk_size` frames. For each chunk:
1. Run VGGT → chunk-local depth, extrinsics, intrinsics.
2. Find **anchor frames**: frames that appear in both the sparse set and this chunk.
3. Compute least-squares SE3 alignment using these anchor pairs.
4. Apply the alignment transform to all frames in the chunk.

#### Why No Drift

Every chunk is aligned **directly to the global sparse anchors** — not chained through previous chunks. There is no sequential error accumulation. Even chunk 50 is aligned to the same global reference as chunk 0.

#### Intrinsics: Global Average

All frames share a single intrinsics matrix averaged from the sparse pass. Since it is the same physical camera, per-chunk intrinsics are nearly identical, and averaging produces a consistent value. Using per-chunk intrinsics was tested but caused visible position discontinuities at chunk boundaries — the slight intrinsics differences between adjacent chunks introduced jumps in the projected 3D tracks. A single global intrinsics avoids this.

#### SE3 Alignment Details

The alignment transform `T` is found by solving:

```
For each anchor pair (chunk_local_idx, sparse_idx):
    sparse_extrs[sparse_idx] ≈ T @ chunk_extrs[chunk_local_idx]
```

This uses:
- **Rotation averaging**: Eigendecomposition of the quaternion outer-product matrix (Markley et al. 2007).
- **Translation averaging**: Arithmetic mean.
- **Outlier rejection**: Anchor pairs whose alignment residual exceeds 2σ are discarded. Requires ≥4 anchors to activate.

Implementation: `utils/se3_utils.py` → `align_chunk_to_global()`.

---

## Phase 3: Progressive Keypoint Tracking

### Problem

Standard tracking runs DINO + TAPIP3D on each chunk independently. Keypoints discovered in chunk 0 are lost at chunk boundaries — there's no continuity. Objects visible across the entire video would be re-discovered in each chunk, producing duplicate tracks.

### Solution: Propagate + Mask + Generate

#### Chunk 0 (First Chunk)

Normal pipeline:
1. **DINO**: Extract DINOv2 features → bipartite clustering → identify peak frames (where each cluster is most visible) → sample keypoints at peak frames.
2. **TAPIP3D**: Track keypoints through the chunk. Output: `coords (T_chunk, N, 3)` in world space + `visibs (T_chunk, N)`.
3. Register as trajectory groups. All become "active" for the next chunk.

#### Chunk N >= 1 (Subsequent Chunks)

```
Active groups from previous chunks
        │
        ▼
  Step A: PROPAGATION
  │  For each active group, take the last frame's 3D world coordinates.
  │  Build 3D queries: [t=0, world_x, world_y, world_z]
  │  Run TAPIP3D → tracks through this chunk.
  │  Append results to each group's trajectory.
  │
  Step B: COMPUTE PATCH MASK + DINO CLUSTERING
  │  Project all propagated tracks to 2D at each frame.
  │  Mark which DINO patches (16x16 grid) are "covered" by visible tracks.
  │  Run DINO feature extraction on full images (needs spatial context).
  │  Cluster all patches (bipartite), but exclude covered patches from
  │  peak-frame counting and point sampling.
  │  → Only genuinely new (uncovered) regions produce keypoints.
  │
  Step C: TRACK NEW KEYPOINTS
  │  Track new keypoints through this chunk via TAPIP3D.
  │  Register as new trajectory groups.
  │
  Step D: UPDATE ACTIVE GROUPS
     All groups (old + new) become active for next chunk.
```

### Key Design Decisions

**3D propagation, not 2D.** When propagating across chunk boundaries, we pass world-space 3D coordinates as queries to TAPIP3D (not 2D pixel reprojections). This is robust to large camera motion between chunks.

**All keypoints propagated, regardless of visibility.** Even occluded keypoints are propagated — TAPIP3D handles visibility internally and the point may become visible again in later frames.

**Mask-then-generate, not generate-then-filter.** Rather than extracting all DINO keypoints and filtering by proximity to existing tracks, we mask out DINO patches that already have tracks *before* peak-frame selection and point sampling. This is more principled: clustering sees the full scene for semantic coherence, but only uncovered regions produce keypoints. A cluster whose only presence is in already-tracked patches gets no peak frame and generates zero keypoints.

**Patch coverage.** A DINO patch is "covered" if at least one visible propagated track projects into it. The 16x16 patch grid naturally provides the right granularity — one patch ≈ `image_size / 16` pixels, which is the minimum meaningful distance between distinct objects in DINO feature space.

### Variable Lifespans

Groups registered in different chunks have different lifespans:

```
Group registered at chunk 0:  [chunk 0 tracked] [chunk 1 propagated] [chunk 2 propagated] ...
Group registered at chunk 2:                                          [chunk 2 tracked] [chunk 3 propagated] ...
```

Each group stores `frame_indices` — the list of global frame indices it covers. At assembly time, per-chunk coordinate arrays are concatenated and `frame_indices` tracks the mapping.

---

## GPU Memory Management

Only one model is on GPU at a time:

| Phase | On GPU | On CPU |
|---|---|---|
| VGGT (Phase 2) | VGGT (~5-6 GB) | Tracker, DINO |
| Propagation (Step A) | Tracker (~3-4 GB) | VGGT, DINO |
| DINO clustering (Step B) | DINO (~1-2 GB) | VGGT, Tracker |
| New keypoint tracking (Step C) | Tracker (~3-4 GB) | VGGT, DINO |

Models are moved via `_move_model_to()` which handles device transfer and CUDA cache clearing. This keeps peak VRAM at ~6 GB regardless of video length.

Additionally, `sparse_max` is automatically clamped to `chunk_size` so the sparse VGGT pass never exceeds the per-chunk VRAM budget.

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--chunk_size` | 60 | Max frames per VGGT dense chunk |
| `--tracking_chunk_size` | `chunk_size` | Max frames per tracking chunk (tracking needs less VRAM, so this can be larger) |
| `--sparse_max` | 150 | Max frames for global sparse VGGT pass (auto-clamped to chunk_size) |
| `--frame_step` | 1 | Take every Nth frame (1 = no skip, 2 = every other frame, etc.) |
| `--target_fps` | 10 | Subsample video-file inputs to ~this fps (0 = use `--frame_step`) |

### Example

```bash
python infer.py \
    --video_path <input_directory> \
    --out_dir outputs/custom \
    --batch_process --scan_depth 1 \
    --frame_step 2 --chunk_size 20 \
    --debug
```

---

## File Reference

| File | Role |
|---|---|
| `infer.py` → `process_long_video()` | Main orchestrator |
| `utils/video_depth_pose_utils.py` → `chunked_inference()` | Hybrid VGGT |
| `utils/se3_utils.py` | SE3 alignment (quaternions, least-squares) |
| `utils/dino_keypoints.py` | DINO feature extraction + clustering |
