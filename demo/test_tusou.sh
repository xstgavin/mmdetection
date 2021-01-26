nohup python test_tusou.py --device cuda:0 --groups 8 --pt 0 2>&1 &
nohup python test_tusou.py --device cuda:0 --groups 8 --pt 1 2>&1 &
nohup python test_tusou.py --device cuda:0 --groups 8 --pt 2 2>&1 &
nohup python test_tusou.py --device cuda:0 --groups 8 --pt 3 2>&1 &
nohup python test_tusou.py --device cuda:1 --groups 8 --pt 4 2>&1 &
nohup python test_tusou.py --device cuda:1 --groups 8 --pt 5 2>&1 &
nohup python test_tusou.py --device cuda:1 --groups 8 --pt 6 2>&1 &
nohup python test_tusou.py --device cuda:1 --groups 8 --pt 7 2>&1 