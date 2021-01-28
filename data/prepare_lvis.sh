####------------- coco
# mkdir /home/jovyan/fast-data/coco
# cp /home/jovyan/data-vol-polefs-2/coco/*2017.zip /home/jovyan/fast-data/coco/
# cd /home/jovyan/fast-data/coco
# cp /home/jovyan/vision/mmdetection/data/lvis/*.zip /home/jovyan/fast-data/coco/

# unzip test2017.zip
# unzip val2017.zip
# unzip train2017.zip
# unzip lvis_v1_image_info_test_challenge.json.zip
# unzip lvis_v1_image_info_test_dev.json.zip
# unzip lvis_v1_train.json.zip
# unzip lvis_v1_val.json.zip
# unzip annotations_trainval2017.zip
# unzip image_info_test2017.zip
# unzip stuff_annotations_trainval2017.zip
# rm -rf *.zip 
# mv lvis*.json ./annotations/
####---------- censor -------

mkdir /home/jovyan/fast-data/censor
cp -r /home/jovyan/data-vol-2/data/censor_det/val /home/jovyan/fast-data/censor/
cp -r /home/jovyan/data-vol-2/data/censor_det/scrolller/random_12.tar /home/jovyan/fast-data/censor/
cp -r /home/jovyan/data-vol-2/data/censor_det/scrolller/annotations.tar /home/jovyan/fast-data/censor/
cd /home/jovyan/fast-data/censor/
tar -xf random_12.tar && mv random_12 train
tar -xf annotations.tar && mv ./annotations/random_12.json ./train/train.json
rm *.tar 
rm -rf annotations

cp -r /home/jovyan/data-vol-1/xiaoshengtao/data/tusou_test.tar /home/jovyan/fast-data/ && cd /home/jovyan/fast-data && tar -xf tusou_test.tar && rm tusou_test.tar




