# Caption Generation

Generate motion captions from TraceExtract episodes and visualize them over the
RGB frames. This is an **optional** post-processing step: it turns the 3D
trajectory output of `infer.py` into per-chunk and window-level natural-language
motion descriptions for downstream training.

The expected input is the output format produced by `infer.py`: each episode
directory contains `images.npy`, `acceleration.npz`, and a `samples/` directory
with `keypoints.npy`, `cluster_ids.npy`, `raw_traj.npy`, and the related validity
masks. See [`docs/output_structure.md`](../docs/output_structure.md) for the full
on-disk schema.

All commands below are run from the TraceExtract repo root, so the scripts live
under `caption/`.

## Setup

Install the caption-only dependencies (the core pipeline installer covers
`numpy`, `pillow`, `opencv-python`, and `scipy`):

```bash
pip install python-dotenv openai google-genai
```

`scipy` is optional but recommended for smoother chunk-boundary detection.

Provide API keys via a `.env` file (copy the template and fill it in) or by
exporting them in your shell:

```bash
cp caption/.env.example caption/.env   # then edit caption/.env
# or export directly:
export OPENAI_API_KEY=your_key_here     # OpenAI backend (default)
export GEMINI_API_KEY=your_key_here     # optional Gemini backend
```

Optional model overrides (`OPENAI_CAPTION_MODEL`, `OPENAI_MERGE_MODEL`,
`GEMINI_CAPTION_MODEL`, `GEMINI_MERGE_MODEL`) are listed in
[`.env.example`](.env.example). On the OpenAI path the merge
stage automatically promotes `merge_type == "full"` windows to `gpt-4o-mini`,
and any short-merge call that returns empty/truncated/malformed JSON falls back
once to `gpt-4o-mini`.

## DROID-style caption curation

`caption/curate_captions.py` works on generic per-episode folders (e.g. DROID,
EgoDex, EgoVerse). It splits each episode into motion chunks, builds motion
masks, sends chunk images to a multimodal model, and then merges adjacent chunk
captions into longer window-level captions.

```bash
# All episodes under a root
python caption/curate_captions.py --root outputs/egoverse

# One or more specific episodes
python caption/curate_captions.py --root outputs/egoverse --video episode_11980
python caption/curate_captions.py --root outputs/egoverse --video episode_11980 --video episode_11987

# Use Gemini instead of OpenAI
python caption/curate_captions.py --root outputs/egoverse --backend gemini

# Dry-run chunking + masks without API calls
python caption/curate_captions.py --root outputs/egoverse --dry-run

# Re-run captioning even when cached per-chunk JSON already exists
python caption/curate_captions.py --root outputs/egoverse --overwrite
```

Common chunking controls:

```bash
python caption/curate_captions.py \
  --root outputs/egoverse \
  --min-chunk-len 8 \
  --max-chunk-len 60 \
  --max-chunks 6 \
  --max-keypoints-per-frame 300
```

Per-episode outputs:

- `curated/chunks_meta.json` — chunk boundaries and metadata.
- `curated/segments_meta.json` — noisy-frame filtering and segment metadata.
- `curated/masks/` — motion masks and overlays for chunks.
- `curated/frames/` — selected first, middle, and last frames.
- `curated/chunk_captions/chunk_XXX.json` — cached model output per chunk.
- `curated/merged_windows.json` — adjacent chunk merge outputs.
- `curated_captions.json` — full debug-oriented caption record.
- `curated_training_texts.json` — lean export for downstream training.

## AgiBot caption curation

`caption/curate_captions_agibot.py` is tuned for AgiBot episodes, which are
already task-level segments. It uses more conservative chunking defaults
(usually one to three chunks per episode) and reads each episode's
`description.txt` as a visual hint while still asking the model to rely on image
evidence.

```bash
# All AgiBot episodes
python caption/curate_captions_agibot.py --root outputs/agibot

# One episode
python caption/curate_captions_agibot.py --root outputs/agibot --video 648544_0_540
```

Useful defaults: `--min-chunk-len 25`, `--max-chunk-len 300`,
`--min-prominence-ratio 0.40`, `--max-chunks 3`. Override them only when an
episode is being split too aggressively or not enough:

```bash
python caption/curate_captions_agibot.py \
  --root outputs/agibot \
  --max-chunks 2 \
  --min-prominence-ratio 0.50
```

Outputs match the DROID pipeline, with extra AgiBot fields in
`curated_training_texts.json`: `task_description`, `chunk_index`, `num_chunks`,
`start_frame`, `end_frame`, and `selected_frame_indices`.

## Debug video visualization

After curation, `caption/visualize_debug.py` renders MP4s with caption text and
raw frame-aligned trajectories overlaid on the RGB frames. It uses
`raw_traj.npy` / `raw_traj_history.npy` (not the retargeted `traj.npy`) because
raw trajectories align with real video frames.

```bash
# One episode, red future trajectories
python caption/visualize_debug.py outputs/egoverse/episode_11980

# Future (red) + history (green dashed)
python caption/visualize_debug.py outputs/egoverse/episode_11980 --visualize_history

# Color by cluster ID
python caption/visualize_debug.py outputs/egoverse/episode_11980 --classify_cluster

# Choose a caption field
python caption/visualize_debug.py outputs/egoverse/episode_11980 --caption-field instruction_2

# Change horizon / playback FPS / output scale
python caption/visualize_debug.py outputs/egoverse/episode_11980 --horizon 16 --fps 10 --scale 2.0

# Every episode under a root
python caption/visualize_debug.py outputs/egoverse --all-videos
```

With `--all-videos`, outputs go to
`<root>/visualize_caption/<episode_name>_inst1.mp4`; for a single episode the
default is `<episode>/curated/debug_instruction_1.mp4`. Use `--output-dir` to
override.

## Typical workflow

1. Run `infer.py` and confirm each episode follows the
   [output structure](../docs/output_structure.md).
2. Run the appropriate curation script:
   ```bash
   python caption/curate_captions.py --root outputs/egoverse
   # or, for AgiBot:
   python caption/curate_captions_agibot.py --root outputs/agibot
   ```
3. Inspect `curated_captions.json` and `curated_training_texts.json`.
4. Generate debug videos:
   ```bash
   python caption/visualize_debug.py outputs/egoverse --all-videos --visualize_history
   ```
5. Review the MP4s for chunk boundaries, caption quality, and trajectory
   alignment.

## Notes

- Default backend is OpenAI; use `--backend gemini` for Gemini models.
- `--dry-run` still produces chunk metadata, masks, and selected frames, so it
  is useful before spending API calls.
- `--overwrite` forces regeneration of cached chunk captions.
- Caption overlays wrap text to the output frame width and display up to four
  lines.
