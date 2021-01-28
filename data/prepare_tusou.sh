#mkdir /home/jovyan/fast-data/tusou/
#sudo mkdir /fast-data && sudo chmod +777 /fast-data

mkdir /fast-data/tusou
#cp /home/jovyan/data-vol-polefs-1/data/tusou_data/tusou_data_440.tar.gz /fast-data/tusou/
cp /home/jovyan/data-vol-polefs-1/data/tusou_data/tusou_data_440_tars/porn.tar /fast-data/tusou/
cp /home/jovyan/data-vol-polefs-1/data/tusou_data/train.lst /fast-data/tusou/
cd /fast-data/tusou/ && tar -xf porn.tar
#rm -rf /fast-data/tusou/tusou_data_440.tar.gz
