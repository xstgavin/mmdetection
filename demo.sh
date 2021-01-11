#CUDA_LIBRARY_PATH=/usr/local/cuda-10.2/lib64
CUDA_VISIBLE_DEVICES=0,1#,2,3
export PORT=25001
#python tools/train.py ./configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py
./tools/dist_train.sh ./configs/lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py 2

