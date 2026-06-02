python train.py \
    -s ~/autodl-tmp/data/tandt/tandt/truck \
    -m ~/autodl-tmp/model/rtsplat/tandt/fastgs_off+remove_lpips/truck \
    --eval \
    --env_scope_center -0.943 -0.083 0.514 \
    --env_scope_radius 1 \
    --init_until_iter 700 \
    --norm_loss_from_iter 700 \
    --xyz_axis 2.0 1.0 0.0 \
    --fastgs_on \
    --fastgs_densify \
    --fastgs_prune \
    --fastgs_grad_thresh 0.0002 \
    --fastgs_grad_abs_thresh 0.0002 \
    --fastgs_dense 0.01 \
    --fastgs_loss_thresh 0.5 \
    --fastgs_min_importance 5 \
    --fastgs_num_cams 10 \
    --lambda_lpips 0

python render.py -m ~/autodl-tmp/model/rtsplat/tandt/fastgs_off+remove_lpips/truck

python metrics.py -m ~/autodl-tmp/model/rtsplat/tandt/fastgs_off+remove_lpips/truck
