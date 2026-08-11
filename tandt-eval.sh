MODEL_DIR=~/autodl-tmp/model/rtsplat/tandt/test_priors_surface_consistent/truck
set -euo pipefail

TRAIN_CMD="python train.py \
    -s ~/autodl-tmp/truck \
    -m $MODEL_DIR \
    --prior_path ~/autodl-tmp/truck/priors \
    --eval \
    --env_scope_center -0.943 -0.083 0.514 \
    --env_scope_radius 1000.0 \
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
    --glossy_angle_reference_spread 0.01 \
    --glossy_angle_max_compensation 4.0 \
    --glossy_threshold 0.15 \
    --glossy_specular_boost 1.0"

mkdir -p $MODEL_DIR
echo "$TRAIN_CMD" > $MODEL_DIR/train_cmd.txt

eval "$TRAIN_CMD"

python render.py -m $MODEL_DIR

python metrics.py -m $MODEL_DIR
