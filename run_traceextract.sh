#!/bin/bash
# Unified TraceExtract launcher for every supported dataset.
#
# Usage:
#   ./run_traceextract.sh <dataset> <input_root> <out_dir> [extra infer.py args...]
#
#   <dataset>     agibot | droid | egodex | egoverse | video
#                 ("video" = generic video files / frame folders, no dataset loader)
#   <input_root>  raw dataset root (see README "Processing datasets" for layouts)
#   <out_dir>     where per-episode output directories are written
#   extra args    passed through to infer.py after the preset, so options that
#                 take a value override it (e.g. `--frame_step 2`, `--max_episodes 1`)
#
# Examples:
#   ./run_traceextract.sh droid    /data/droid       outputs/droid --droid_num_shards 1
#   ./run_traceextract.sh egodex   /data/egodex      outputs/egodex
#   ./run_traceextract.sh video    /data/my_videos   outputs/custom --scan_depth 0
#
# Optional environment variables:
#   MIN_FREE_VRAM_MIB=23000  wait until GPU 0 has this much free VRAM before
#                            each attempt (default 0 = start immediately)
#   MAX_RETRIES=3            re-launch infer.py when it exits with an error (an
#                            episode failed, or it was killed, e.g. OOM); finished
#                            episodes are skipped, so each retry resumes where the
#                            last one stopped (default 0 = run once)
#   RETRY_SLEEP=600          seconds to wait between retries (default 600)
#
# Run with the TraceExtract conda env active (see README "Installation").

set -uo pipefail

if [ $# -lt 3 ]; then
    sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

DATASET=$1
INPUT_ROOT=$2
OUT_DIR=$3
shift 3

# infer.py is called by absolute path (no cd), so relative paths in the extra
# arguments (e.g. --episode_list) resolve against the caller's directory.
REPO_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
INPUT_ROOT=$(realpath -m "$INPUT_ROOT")
OUT_DIR=$(realpath -m "$OUT_DIR")

# Shared settings: 60-frame VGGT chunks, 360-frame tracking chunks, 60-frame
# global sparse pass — sized for a ~24 GB GPU. Re-runs skip finished episodes.
COMMON=(--skip_existing --chunk_size 60 --tracking_chunk_size 360 --sparse_max 60)

# Per-dataset presets (frame rates chosen to sample each source at ~5-10 Hz).
case "$DATASET" in
    agibot)   PRESET=(--dataset_name agibot --target_fps 5) ;;
    droid)    PRESET=(--dataset_name droid --frame_step 3 --target_fps 0) ;;   # DROID is 15 Hz → 5 Hz
    egodex)   PRESET=(--dataset_name egodex --target_fps 10 --target_hw 360 640
                      --num_points_per_entity 144) ;;                          # 1080p → 360p
    egoverse) PRESET=(--dataset_name egoverse --target_fps 10) ;;
    video)    PRESET=(--batch_process --scan_depth 1 --target_fps 10 --frame_step 3) ;;
    *)
        echo "Unknown dataset '$DATASET' (expected: agibot, droid, egodex, egoverse, video)" >&2
        exit 1 ;;
esac

MIN_FREE_VRAM_MIB=${MIN_FREE_VRAM_MIB:-0}
MAX_RETRIES=${MAX_RETRIES:-0}
RETRY_SLEEP=${RETRY_SLEEP:-600}

wait_for_vram() {
    [ "$MIN_FREE_VRAM_MIB" -gt 0 ] || return 0
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && [ "$free_mib" -gt "$MIN_FREE_VRAM_MIB" ]; then
            echo "Free VRAM ${free_mib} MiB > ${MIN_FREE_VRAM_MIB} MiB, starting."
            return 0
        fi
        echo "Waiting for free VRAM > ${MIN_FREE_VRAM_MIB} MiB (now: ${free_mib} MiB)..."
        sleep 60
    done
}

attempt=0
while true; do
    wait_for_vram
    python "$REPO_DIR/infer.py" \
        --video_path "$INPUT_ROOT" \
        --out_dir "$OUT_DIR" \
        "${COMMON[@]}" \
        "${PRESET[@]}" \
        "$@"
    code=$?
    if [ $code -eq 0 ]; then
        echo "Finished: outputs in $OUT_DIR"
        exit 0
    fi
    if [ $attempt -ge "$MAX_RETRIES" ]; then
        echo "infer.py exited with code $code" >&2
        exit $code
    fi
    attempt=$((attempt + 1))
    echo "infer.py exited with code $code; retry $attempt/$MAX_RETRIES in ${RETRY_SLEEP}s..."
    sleep "$RETRY_SLEEP"
done
