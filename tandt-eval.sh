MODEL_DIR=~/autodl-tmp/model/rtsplat/tandt/fastgsV2_prune_protect_stddensify/truck

TRAIN_CMD="python train.py \
    -s ~/autodl-tmp/data/tandt/tandt/truck \
    -m $MODEL_DIR \
    --eval \
    --env_scope_center -0.943 -0.083 0.514 \
    --env_scope_radius 0 \
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
    --edge_aware_from_iter 10000"

eval $TRAIN_CMD

mkdir -p $MODEL_DIR
echo "$TRAIN_CMD" > $MODEL_DIR/train_cmd.txt

python render.py -m $MODEL_DIR

python metrics.py -m $MODEL_DIR
