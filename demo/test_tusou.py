from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import sys
from tqdm import tqdm

def get_tusou_data(fpath='/home/jovyan/fast-data/tusou_test/script/valid.lst'):
    lines = open(fpath,'r').readlines()
    PORN6=['hentai','anime','norm','porn','sexy','very_sexy']
    dat={}
    print(len(lines))
    for line in lines:
        a = line.rstrip().split(' ')
        file = a[0]
        file = file.replace('home/jovyan/','')
        file = file.replace('fast-data/tusou_data_440/train','fast-data/tusou')
        annt = PORN6[int(a[1])]
        if annt not in dat.keys():
            dat[annt]=[]
            dat[annt].append(file)
        else:
            dat[annt].append(file)
    return dat

def check_result(result):
    for i in range(7):
        if result[i].shape[0]>0:
            return True
    return False
    
def process_image(model,imageList, gt_is_porn=True,grp=4,pt=0):
    dat_list = []
    for idx, imgfil in tqdm(enumerate(imageList)):
        if idx%grp == pt:
            try:
                result = inference_detector(model, imgfil)
                is_porn = check_result(result)
            except:
                continue
            if is_porn != gt_is_porn:
                dat_list.append(imgfil)
        else:
            #print(imgfil,' ',idx)
            continue
    return dat_list

    
def main():
  
    
    parser = ArgumentParser()
    #parser.add_argument('img', help='Image file')
    parser.add_argument('--config', 
                        default = '../configs/censor/faster_rcnn_x101_32x4d_fpn_1x_censor.py', 
                        help='Config file')
    parser.add_argument('--checkpoint', 
                        default = '../work_dirs/faster_rcnn_x101_32x4d_fpn_1x_censor/epoch_48.pth', 
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--groups', type=int, default=4, help='groups of threads')
    parser.add_argument(
        '--pt', type=int, default=0, help='current thread index')
    parser.add_argument(
        '--category', default='porn', help='class category to test')    
    args = parser.parse_args()
    print(args.groups, args.pt)
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    img_dat = get_tusou_data('/fast-data/tusou/train.lst')
    false_list = process_image(model,img_dat[args.category], grp=args.groups, pt=args.pt)
    #fid = open('/fast-data/tusou/train_det/%s_%d_%d_%s.txt'%(args.category,args.groups,args.pt,args.device),'w')
    fid = open('./tusou_train_det/%s_%d_%d_%s.txt'%(args.category,args.groups,args.pt,args.device),'w')
    for img in false_list:
        fid.write(img+'\n')
    fid.close()
    # show the results
    #show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
