from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import argparse
import csv
import pandas as pd
import collections
import cv2
import sys
import torch
import datetime
from itertools import count
import numpy as np
from glob import glob1, glob
import json
import time
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
import re
from pysot.utils import bbox as bbx
import argparse
import time
from sys import platform
import csv
from models import *
from utils.datasets import *
from utils.utils import *
import itertools


torch.set_num_threads(5)
def get_frames(video_name):

    if video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        vid_name = "./videos/{}".format(video_name)
        cap = cv2.VideoCapture(vid_name)#(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        print('Video read fail')
        # images = glob(os.path.join(video_name, '*.jp*'))
        # images = sorted(images,key=lambda x: int(x.split('/')[-1].split('.')[0]))
        # for img in images:
        #     frame = cv2.imread(img)
        #     yield frame


# def detect(cfg1,
#            data_cfg,
#            weights,
#            images='data/samples',  # input folder
#            output='output',  # output folder
#            fourcc='mp4v',  # video codec
#            img_size=416,
#            conf_thres=0.5,
#            nms_thres=0.5,
#            save_txt=True,
#            save_images=True,
#            webcam=False
def main():
    cfg1 = './cfg/yolov3-spp.cfg'
    data_cfg = './data/coco.data'
    weights = './weights/yolov3-spp.weights'     
    images = './videos'  
    output='./output'
    fourcc='mp4v'
    img_size=416
    conf_thres=0.5
    nms_thres = 0.5
    save_images=True


    #yolo Initialize
    device = torch_utils.select_device()
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    #yolo Initialize model
    if ONNX_EXPORT:
        s = (320, 192)  # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
        model = Darknet(cfg1, s)
    else:
        model = Darknet(cfg1, img_size)

    #yolo Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    #yolo Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    #yolo Eval mode
    model.to(device).eval()

    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    #Set Dataloader
    vid_path, vid_writer = None, None
    dataloader = LoadImages(images, img_size=img_size)
    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    for i, (path, img, im0, vid_cap) in enumerate(dataloader):    
        the_list = []
        t = time.time()
        save_path = str(Path(output) / Path(path).name)
        video_name_yolo = os.path.basename(save_path)
        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                       
             # Print results to screen
            print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            for *xyxy,conf,cls_conf,cls in det:
                if cls == 0 :
                    if conf >= 0.75:
                        my_list_det = ('%g,' * 4) % (*xyxy,)
                        #my_list = ('%g ' * 6) % (*xyxy, cls,conf)

                        my_list_list = [my_list_det]
                        my_list_strp = my_list_list[0][:-1]
                        my_list_int = np.array(my_list_strp.split(",")).astype('int').tolist()
                        my_int = np.array(my_list_strp.split(","))
                        the_list.append(my_list_int)
                        with open('./videos/'+video_name_yolo+'_yolo.csv', 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile, quoting=0, delimiter = ",")#,quotechar='',escapechar='')
                            writer.writerow(my_int)
                                            
                    else:
                        continue
                else:
                    continue

                #Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
            
        print('Done. (%.3fs)' % (time.time() - t))

        if save_images:  # Save image with detections
            if dataloader.mode == 'images':
                cv2.imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
                vid_writer.write(im0)


                   
    if save_images:
        print('Results saved to %s' % os.getcwd() + os.sep + output)
        if platform == 'darwin':  # macos
            os.system('open ' + output + ' ' + save_path)



    #siam load config

    cfg.merge_from_file('./experiments/siamrpn_r50_l234_dwxcorr_8gpu/config.yaml')
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    #siam create model
    model = ModelBuilder()

    #siam load model
    model.load_state_dict(torch.load('./experiments/siamrpn_r50_l234_dwxcorr_8gpu/model.pth', map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    #siam build tracker
    tracker = build_tracker(model)
    #video_list = []
    video_list_mp4 = glob1("./videos/", "*.mp4")
    #video_list.append(video_list_mp4)
    video_list_avi = glob1("./videos/","*.avi")
    #video_list.append(video_list_avi)
    
    #print(video_list)

    for video_name in itertools.chain(video_list_mp4, video_list_avi):
        print(video_name)
        #with open('./demo/vids/'+video_name+'_yolo.csv') as vid:
        #df = pd.read_csv('./demo/vids/'+video_name+'_yolo.csv', delimiter = ',', dtype=int)
            # read = csv.reader(vid, delimiter = ',', newline='')
            # for row in read:
            #    cords = row[0]
        #print(cords)
        #cords = [list(x) for x in df.values]
        #cords = [1,2,3,4]
        try:
            df = pd.read_csv('./videos/'+video_name+'_yolo.csv', delimiter=',', header=None)
            cords = [list(x) for x in df.values]
        except:
            continue
        master_tracking_list = []
        object_counter = 0
        for cord in cords:
            cord = [cord[0],cord[1],cord[2]-cord[0],cord[3]-cord[1]]
            try:
                iou = [bbx.IoU(track_cor,cord) for track_cor in master_tracking_list]
            except:
                print("IOU calculation fail")
                pass

            if len(iou) == 0:
                print("appending cord")
                master_tracking_list.append(cord)
                iou = [0.0]
             
            if max(iou) <= 0.2:    
                print(max(iou))         
                object_counter = object_counter + 1
                first_frame = True
            # if video_name:#args.video_name:
            #     video_name = video_name.split('/')[-1].split('.')[0]
            #     print(video_name)
            #     #video_name = args.video_name.split('/')[-1].split('.')[0]

            # else:
            #     exit()
            
            #print(cord)
                frame_count = 0
                #mylist = [frame_count,object_counter,cord[0],cord[1],cord[2],cord[3],video_name]
                for frame in get_frames(video_name):#(args.video_name):
                    if first_frame:
                        try:
                            init_rect = cord
                        except:
                            exit()
                        tracker.init(frame, init_rect)
                        first_frame = False
                    else:
                        outputs = tracker.track(frame)

                        if 'polygon' in outputs:
                            exit()
                        else:
                            #crds = map(int,outputs['bbox'])
                            bbox = list(map(int,outputs['bbox']))
                            master_tracking_list.append(bbox)
                            cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),3)   
                            frame_count = frame_count + 1   
                            mylist = [frame_count,object_counter,bbox[0],bbox[1],bbox[2],bbox[3],video_name]

                            vid_writer.release()
                            save = './videos/out/'+str(object_counter)+'_'+str(frame_count)+'.vid.mp4'
                            out = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
                            #print(frame_count)
                            out.write(frame)

            else:
                print("Coordinate Ignored")
                continue
                

if __name__ == '__main__':
    main()

# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg1', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    # parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    # parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    # parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    # parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    # parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    # parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    # parser.add_argument('--output', type=str, default='output', help='specifies the output path for images and videos')
    # opt = parser.parse_args()
    # print(opt)

    # with torch.no_grad():
    #     detect(opt.cfg1,
    #            opt.data_cfg,
    #            opt.weights,
    #            images=opt.images,
    #            img_size=opt.img_size,
    #            conf_thres=opt.conf_thres,
    #            nms_thres=opt.nms_thres,
    #            fourcc=opt.fourcc,
    #            output=opt.output)
