python train.py -s ~/autodl-tmp/data/tandt/tandt/truck \
-m ~/autodl-tmp/model/tandt/truck --eval --env_scope_center -0.943 -0.083 0.514 --env_scope_radius 1 --init_until_iter 700 --norm_loss_from_iter 700 --xyz_axis 2.0 1.0 0.0

python render.py -m ~/autodl-tmp/model/tandt/truck

python metrics.py -m ~/autodl-tmp/model/tandt/truck