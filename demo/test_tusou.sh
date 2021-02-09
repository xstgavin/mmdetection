# nohup python test_tusou.py --device cuda:0 --groups 8 --pt 0 2>&1 &
# nohup python test_tusou.py --device cuda:0 --groups 8 --pt 1 2>&1 &
# nohup python test_tusou.py --device cuda:0 --groups 8 --pt 2 2>&1 &
# nohup python test_tusou.py --device cuda:0 --groups 8 --pt 3 2>&1 &
# nohup python test_tusou.py --device cuda:1 --groups 8 --pt 4 2>&1 &
# nohup python test_tusou.py --device cuda:1 --groups 8 --pt 5 2>&1 &
# nohup python test_tusou.py --device cuda:1 --groups 8 --pt 6 2>&1 &
# nohup python test_tusou.py --device cuda:1 --groups 8 --pt 7 2>&1 

nohup python test_image_list.py --device cuda:0  --image-list /home/jovyan/data-vol-2/data/tusou_part_list/all_valid_0.txt --prefix /home/jovyan/fast-data/tusou_data --threshold 0.5  --groups 8 --pt 0 2>&1 &
nohup python test_image_list.py --device cuda:0  --image-list /home/jovyan/data-vol-2/data/tusou_part_list/all_valid_0.txt --prefix /home/jovyan/fast-data/tusou_data --threshold 0.5  --groups 8 --pt 1 2>&1 &
nohup python test_image_list.py --device cuda:1  --image-list /home/jovyan/data-vol-2/data/tusou_part_list/all_valid_0.txt --prefix /home/jovyan/fast-data/tusou_data --threshold 0.5  --groups 8 --pt 2 2>&1 &
nohup python test_image_list.py --device cuda:1  --image-list /home/jovyan/data-vol-2/data/tusou_part_list/all_valid_0.txt --prefix /home/jovyan/fast-data/tusou_data --threshold 0.5  --groups 8 --pt 3 2>&1 &
nohup python test_image_list.py --device cuda:2  --image-list /home/jovyan/data-vol-2/data/tusou_part_list/all_valid_0.txt --prefix /home/jovyan/fast-data/tusou_data --threshold 0.5  --groups 8 --pt 4 2>&1 &
nohup python test_image_list.py --device cuda:2  --image-list /home/jovyan/data-vol-2/data/tusou_part_list/all_valid_0.txt --prefix /home/jovyan/fast-data/tusou_data --threshold 0.5 --groups 8 --pt 5 2>&1 &
nohup python test_image_list.py --device cuda:3  --image-list /home/jovyan/data-vol-2/data/tusou_part_list/all_valid_0.txt --prefix /home/jovyan/fast-data/tusou_data --threshold 0.5  --groups 8 --pt 6 2>&1 &
nohup python test_image_list.py --device cuda:3  --image-list /home/jovyan/data-vol-2/data/tusou_part_list/all_valid_0.txt --prefix /home/jovyan/fast-data/tusou_data --threshold 0.5 --groups 8 --pt 7 2>&1 
