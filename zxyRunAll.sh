export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py \
-s ~/autodl-tmp/data/mipnerf360/room  \
-m ~/autodl-tmp/output/models/room \
--quiet \
--eval \
--indoor \
--solidification_start 20000 \
--lambda_conn_max 0.01 \
--lambda_opaque_max 0.005 \
--lambda_area 1e-4 \
--tau_edge 0.1 \
--k_edge 5


python render.py  -m ~/autodl-tmp/output/models/room
python metrics.py -m ~/autodl-tmp/output/models/room