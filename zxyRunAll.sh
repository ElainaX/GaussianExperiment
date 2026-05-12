export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py \
-s ~/autodl-tmp/data/mipnerf360/room  \
-m ~/autodl-tmp/output/models/room \
--quiet \
--eval \
--indoor \
--test_iterations 30000


python render.py  -m ~/autodl-tmp/output/models/room
python metrics.py -m ~/autodl-tmp/output/models/room