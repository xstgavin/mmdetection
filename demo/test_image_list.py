from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import sys
from tqdm import tqdm
classes=('penis','vagina','breast','butt','naked-body','naked-bottom','underwear','bra','dildo','cum')


def get_tusou_data(fpath='/home/jovyan/fast-data/tusou_test/script/valid.lst'):
    lines = open(fpath,'r').readlines()
    PORN6=['hentai','anime','norm','porn','sexy','very_sexy']
    dat={}
    #print(len(lines))
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

import cv2
import os
import json
#lines = open('../data/censor/test/abby_fc2_test.txt').readlines()
#lines = open('../data/censor/test/fc2_photos.txt').readlines()
# lines = open('/home/shengtao/Data/tusou_test/tusou_1.txt').readlines()
# test_data={}
# for line in lines:
#     img_path = line.rstrip('\n')
#     img_tag = img_path.split('/')[-2]
#     if img_tag  not in test_data.keys():
#         test_data[img_tag]=[img_path]
#     else:
#         test_data[img_tag].append(img_path)
# print(test_data.keys())
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm

def get_test_data(fpath):
    lines = open(fpath).readlines()
    test_data={}
    for line in lines:
        img_path = line.rstrip('\n')
        img_tag = img_path.split('/')[-2]
        if img_tag  not in test_data.keys():
            test_data[img_tag]=[img_path]
        else:
            test_data[img_tag].append(img_path)
    #print(test_data.keys())
    return test_data

def check_result(result,thr):
    checkList = [0,1,2,3,4,5,6,7,9,10,11] # without hand
    for i in checkList:
        if result[i].shape[0]>0:
            for j in range(result[i].shape[0]):
                if result[i][j,4]>=thr:
                    return True
    return False

def draw_rects(imageFile,ret,thre=0.3, classes_name=''):
    #print(len(ret))
    if classes_name == '':
        classes=( 'penis', 'vagina','breast', 'butt','naked-body', 'naked-bottom', 'underwear','bra','hand','dildo','cum','tongue')
    else:
        classes = classes_name
    #print(len(classes))
    img = cv2.imread(imageFile)
    if img is None:
        return img
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #bb = an['bbox']
    for i in range(12):
        #print(ret[i])
        if ret[i].shape[0] == 0:
            continue
        else:
            for j in range(ret[i].shape[0]):
                
                bb = ret[i][j,:]
                #print(bb)
                pt1= (int(bb[0]),int(bb[1]))
                pt2=(int(bb[2]), int(bb[3]))
                conf = bb[4]
                if conf <thre:
                    continue
                img=cv2.rectangle(img,pt1,pt2,(255,0,0),3)
                img=cv2.putText(img,classes[i]+' | %.2f'%conf,pt1, cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2,cv2.LINE_AA)
    #print('done')
    return img

def test_list(model,fpath, saveRoot='/home/shengtao/Data/coco/ret/',thr=0.3, saveAllPos=False,saveAllNeg=False,prefix='',group=4,pid=0,cuda='cuda:0'):
    if not os.path.exists(saveRoot):
        os.mkdir(saveRoot)
    test_data = get_test_data(fpath)
    testKeys = list(test_data.keys())
    fid = open('./'+fpath.split('/')[-1]+'_ret_%d_cuda_%s.json'%(pid,cuda),'w')
    retDat = {}

    for idx,tag in enumerate(testKeys):
        if not idx%group == pid:
            continue
        #count = count+1
        if tag not in retDat.keys():
            retDat[tag]={}
        #for imgfile in tqdm(test_data[tag]):
        for imgfile in test_data[tag]:  
            imgfile = os.path.join(prefix,imgfile.split(' ')[0])
            #print(imgfile)
            try:
            #if True:
                result = inference_detector(model, imgfile)
                ret = check_result(result,thr)
                cat = imgfile.split('/')[-2]
                if saveAllPos:
                    if ret:
                        img = draw_rects(imgfile,result,thr,classes_name=classes)
                        if img is not None:
                            cv2.imwrite(saveRoot+'%d_'%ret+cat+'_'+imgfile.split('/')[-1].split('.')[0]+'.jpg',img)  
                if saveAllNeg:
                    if not ret:
                        img = draw_rects(imgfile,result,thr,classes_name=classes)
                        if img is not None:
                            cv2.imwrite(saveRoot+'%d_'%ret+cat+'_'+imgfile.split('/')[-1].split('.')[0]+'.jpg',img)                 
                retDat[tag][imgfile]={}
                for ij in range(10):
                    #print(result[ij])
                    crtClass = classes[ij]
                    retDat[tag][imgfile][crtClass]=[]
                    for ix in range(result[ij].shape[0]):
                        #retDat[tag][imgfile][crtClass].append(list(result[ij][ix]))
                        crtRet=result[ij][ix].tolist()
                        retDat[tag][imgfile][crtClass].append(crtRet)
            except:
                print(imgfile+' failed')
                retDat[tag][imgfile]={}
                continue
    #print(retDat)            
    json.dump(retDat,fid,indent=4)
    fid.close()
    return retDat
#test_list(model,'/home/shengtao/Data/tusou_test/fc2_failed_photo.txt')

    
def main():
  
    
    parser = ArgumentParser()
    #parser.add_argument('img', help='Image file')
    parser.add_argument('--config', 
                        default = '../configs/censor/faster_rcnn_x101_32x4d_fpn_1x_fc2pv_rotdat-v2.py', 
                        help='Config file')
    parser.add_argument('--checkpoint', 
                        default = '../work_dirs/faster_rcnn_x101_32x4d_fpn_1x_fc2pv_rotdat-v2/epoch_18.pth', 
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
    parser.add_argument(
        '--image-list',default='/home/jovyan/fast-data/porn_sel.txt',help='path to image list')
    parser.add_argument('--save-root',default='/home/jovyan/fast-data/tusou/')
    parser.add_argument('--threshold',default=0.3, help='detection threshold')
    parser.add_argument('--prefix',default='',help='image prefix path')
    args = parser.parse_args()
    print(args.groups, args.pt)
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image

    test_list(model,fpath=args.image_list,saveRoot= args.save_root,thr=float(args.threshold),saveAllPos=True,prefix=args.prefix,
              group=args.groups,pid=args.pt,cuda=args.device)
    # show the results
    #show_result_pyplot(model, args.img, result, score_thr=args.score_thr)

if __name__ == '__main__':
    main()
