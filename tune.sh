#CUDA_LIBRARY_PATH=/usr/local/cuda-10.2/lib64
CUDA_VISIBLE_DEVICES=0,1#,2,3
export PORT=25001
#python tools/train.py ./configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py
#./tools/dist_train.sh ./configs/lvis/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_lvis_v1_tune.py 2
#python tools/train.py ./configs/censor/mask_rcnn_x101_32x4d_fpn_sample1e-3_mstrain_1x_censor_v1_tune.py
#python tools/train.py ./configs/censor/faster_rcnn_x101_32x4d_fpn_1x_censor.py
#./tools/dist_train.sh ./configs/censor/faster_rcnn_x101_32x4d_fpn_1x_censor-freeze-s1.py 2
#python tools/train.py  ./configs/censor/faster_rcnn_x101_32x4d_fpn_1x_censor-freeze-s1.py
python tools/train.py  ./configs/censor/faster_rcnn_x101_32x4d_fpn_1x_rotdat.py

#./tools/dist_train.sh ./configs/censor/faster_rcnn_x101_64x4d_fpn_1x_censor.py 4
