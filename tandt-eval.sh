set -euo pipefail

# Keep this tag short and update it whenever the experiment purpose changes.
EXPERIMENT_NAME="raytrace"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
COMMIT_DATE="$(git -C "$SCRIPT_DIR" show -s --date=format:%m%d --format=%cd HEAD)"
COMMIT_HASH="$(git -C "$SCRIPT_DIR" rev-parse --short=6 HEAD)"
MODEL_ROOT=~/autodl-tmp/model/rtsplat/tandt
MODEL_DIR="${MODEL_ROOT}/${EXPERIMENT_NAME}_${COMMIT_DATE}_${COMMIT_HASH}/truck"

if [[ -n "$(git -C "$SCRIPT_DIR" status --porcelain --untracked-files=no)" ]]; then
    echo "Warning: tracked files contain uncommitted changes; the output tag identifies HEAD only." >&2
fi

echo "Experiment output: $MODEL_DIR"

# Fail before the long training run if the pinned official CUDA/OptiX backend
# or its OptiX headers are unavailable.
git -C "$SCRIPT_DIR" submodule update --init submodules/3dgrut
git -C "$SCRIPT_DIR/submodules/3dgrut" submodule update --init threedgrt_tracer/dependencies/optix-dev
python -m secondary_raytracer.build_3dgrt

# SOTA setting: generated normal priors are debug-only, not a training loss.
TRAIN_CMD="python train.py \
    -s ~/autodl-tmp/truck \
    -m $MODEL_DIR \
    --prior_path ~/autodl-tmp/truck/priors \
    --eval \
    --env_scope_center -0.943 -0.083 0.514 \
    --env_scope_radius 1 \
    --init_until_iter 700 \
    --norm_loss_from_iter 700 \
    --xyz_axis 2.0 1.0 0.0 \
    --fastgs_on \
    --fastgs_prune \
    --fastgs_protect_prune \
    --fastgs_grad_thresh 0.0002 \
    --fastgs_grad_abs_thresh 0.0002 \
    --fastgs_dense 0.01 \
    --fastgs_loss_thresh 0.5 \
    --fastgs_min_importance 5 \
    --fastgs_num_cams 10 \
    --lambda_lpips 0 \
    --lambda_edge_aware 0.1 \
    --edge_aware_from_iter 10000 \
    --lambda_normal_prior 0 \
    --normal_prior_from_iter 1000 \
    --normal_prior_warmup_iters 2000 \
    --normal_prior_axis_sign -1.0 1.0 1.0 \
    --normal_prior_edge_suppression 2.0 \
    --normal_prior_pool_size 3 \
    --glossy_prior_on \
    --glossy_from_iter 5000 \
    --glossy_interval 500 \
    --glossy_num_cams 8 \
    --glossy_min_views 3 \
    --glossy_wavelet_levels 2 \
    --glossy_wavelet_far_levels 4 \
    --glossy_depth_scale_start 0.35 \
    --glossy_depth_scale_end 0.85 \
    --glossy_guided_filter_radius 1 \
    --glossy_guided_filter_iterations 2 \
    --glossy_plane_consensus_threshold 0.30 \
    --glossy_plane_consensus_downsample 8 \
    --glossy_plane_consensus_radius 5 \
    --glossy_plane_consensus_majority 0.60 \
    --glossy_plane_consensus_blend 0.75 \
    --glossy_plane_consensus_depth_floor 0.50 \
    --glossy_angle_reference_spread 0.01 \
    --glossy_angle_max_compensation 4.0 \
    --glossy_threshold 0.15 \
    --glossy_prior_roughness_threshold 0.45 \
    --glossy_specular_boost 1.0 \
    --glossy_target_roughness 0.15 \
    --glossy_target_reflectance 0.70 \
    --secondary_raytrace_on \
    --secondary_raytrace_strength 0.8 \
    --secondary_raytrace_glossy_low 0.15 \
    --secondary_raytrace_glossy_high 0.25 \
    --secondary_raytrace_roughness_max 0.25 \
    --secondary_raytrace_origin_epsilon 0.02 \
    --secondary_raytrace_thickness_ratio 0.10 \
    --secondary_raytrace_rebuild_interval 1 \
    --secondary_raytrace_min_transmittance 0.03 \
    --secondary_raytrace_from_iter 30000"

mkdir -p "$MODEL_DIR"
echo "$TRAIN_CMD" > "$MODEL_DIR/train_cmd.txt"

eval "$TRAIN_CMD"

python render.py -m "$MODEL_DIR" --skip_train --skip_mesh

python metrics.py -m "$MODEL_DIR"
