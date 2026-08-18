MODEL_DIR=~/autodl-tmp/model/rtsplat/tandt/test_priors_sota_captured_probe_128/truck
set -euo pipefail

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
    --glossy_specular_boost 1.0 \
    --glossy_target_roughness 0.15 \
    --glossy_target_reflectance 0.70 \
    --local_probe_on \
    --local_probe_count 4 \
    --local_probe_resolution 128 \
    --local_probe_strength 0.8 \
    --local_probe_radiance_max 4.0 \
    --local_probe_glossy_low 0.10 \
    --local_probe_glossy_high 0.20 \
    --local_probe_scope_radius 15.0 \
    --local_probe_surface_offset 0.25 \
    --local_probe_query_radius 5.0 \
    --local_probe_from_iter 30000"

mkdir -p $MODEL_DIR
echo "$TRAIN_CMD" > $MODEL_DIR/train_cmd.txt

eval "$TRAIN_CMD"

python render.py -m $MODEL_DIR --skip_train --skip_mesh --render_tag probe_on

python render.py -m $MODEL_DIR --skip_train --skip_mesh --disable_local_probe --render_tag probe_off

python metrics.py -m $MODEL_DIR
