mkdir /home/jovyan/fast-data/censor
# copy val dataset
#cp -r /home/jovyan/data-vol-2/data/censor_det/val /home/jovyan/fast-data/censor/

#### tusou sample test dataset
cp -r /home/jovyan/data-vol-1/xiaoshengtao/data/tusou_test.tar /home/jovyan/fast-data/ && cd /home/jovyan/fast-data && tar -xf tusou_test.tar && rm tusou_test.tar
####----------random-12 version training set
# cp -r /home/jovyan/data-vol-2/data/censor_det/scrolller/random_12.tar /home/jovyan/fast-data/censor/
# cp -r /home/jovyan/data-vol-2/data/censor_det/scrolller/annotations.tar /home/jovyan/fast-data/censor/
# cd /home/jovyan/fast-data/censor/
# tar -xf random_12.tar && mv random_12 train
# tar -xf annotations.tar && mv ./annotations/random_12.json ./train/train.json

####----------rand12_pornpic2k training set
#cp -r /home/jovyan/data-vol-2/data/censor_det/scrolller/rand12_pornpic2k.tar /home/jovyan/fast-data/censor/
#cp -r /home/jovyan/data-vol-2/data/censor_det/scrolller/annotations.tar /home/jovyan/fast-data/censor/
#cd /home/jovyan/fast-data/censor/
#tar -xf rand12_pornpic2k.tar && mv rand12_pornpic2k train
#tar -xf annotations.tar && mv ./annotations/rand12_pornpic2k.json ./train/train.json
#
#rm *.tar 
#rm -rf annotations
#### --------- rand12_pornpic2k_hm3k_abby_javporn_fc2phto --
cp -r /home/jovyan/data-vol-2/data/censor_det/censor_dat/val.tar /home/jovyan/fast-data/censor/
cp -r /home/jovyan/data-vol-2/data/censor_det/censor_dat/train.tar /home/jovyan/fast-data/censor/
cp -r /home/jovyan/data-vol-2/data/censor_det/censor_dat/annotations_add.tar /home/jovyan/fast-data/censor/
cd /home/jovyan/fast-data/censor/
tar -xf val.tar && rm val.tar
tar -xf train.tar && rm train.tar 
tar -xf annotations_add.tar  && rm annotations_add.tar
cp annotations_add/random12_pornpic2k_hm3k_abby_javporn_fc2photo_jointRot_bgrd.json ./train/
cp annotations_add/val_jointRot_bgrd.json ./val/


#### test-set


