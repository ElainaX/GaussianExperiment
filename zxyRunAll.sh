export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py \
-s ~/autodl-tmp/data/mipnerf360/room  \
-m ~/autodl-tmp/output/models/exp_0.1.0/train02/room \
--quiet \
--eval \
--solidification_start 20000 \
--lambda_conn_max 0.01 \
--lambda_opaque_max 0.005 \
--lambda_area 1e-4 \
--tau_edge 0.1 \
--k_edge 3 \
--iterations 30000 \
--position_lr_max_steps 30000 \
--densify_from_iter 500 \
--densify_until_iter 10000 \
--start_pruning 4000 \
--start_opacity_floor 5000 \
--start_vertex_opt 12000 \
--start_upsampling 25001 \
--max_points 1000000 \
--final_opacity_iter 24000 \
--sigma_until 30000 \
--iteration_mesh 5000 \
--skip_delaunay



python render.py  -m ~/autodl-tmp/output/models/exp_0.1.0/train02/room
python metrics.py -m ~/autodl-tmp/output/models/exp_0.1.0/train02/room
python create_ply.py ~/autodl-tmp/output/models/exp_0.1.0/train02/room/point_cloud/iteration_30000