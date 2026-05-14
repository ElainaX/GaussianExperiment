export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py \
-s ~/autodl-tmp/data/mipnerf360/room  \
-m ~/autodl-tmp/output/models/exp_0.1.0/room \
--quiet \
--eval \
--indoor \
--solidification_start 1000 \
--lambda_conn_max 0.01 \
--lambda_opaque_max 0.005 \
--lambda_area 1e-4 \
--tau_edge 0.1 \
--k_edge 5


python render.py  -m ~/autodl-tmp/output/models/exp_0.1.0/room
python metrics.py -m ~/autodl-tmp/output/models/exp_0.1.0/room