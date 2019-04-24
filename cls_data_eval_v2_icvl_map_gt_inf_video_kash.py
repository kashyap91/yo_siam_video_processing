'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''

import sys
sys.path.insert(0,'/home/user/caffe-3d/python')
import caffe
from dataset.icvl_inf_eval_video_Kash import icvl_inf_eval_video_Kash
import numpy as np
from utils.cython_bbox import bbox_overlaps
from utils.bbox_transform import bbox_transform

import scipy.misc
import cv2
import os
import datetime
import json

def bbox_iou_mihai(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        #b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    # else:
    #     # Transform from center and width to exact coordinates
    #     b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    #     b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    #     b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    #     b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1) 
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    #inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    #inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    inter_area = np.maximum(0, inter_rect_x2 - inter_rect_x1) * np.maximum(0, inter_rect_y2 - inter_rect_y1)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    #print(pred_cls)
    #print(target_cls)
    #print(tp)
    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects
        #print(c)
        #print(n_gt)
        #print(n_p)
        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i]).astype(np.float)
            tpc = np.cumsum(tp[i]).astype(np.float)
            # print(i)
            # print("FPC...")
            # print(fpc)
            # print("TPC")
            # print(tpc)
            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))
            # print("Recall")
            # print(recall_curve)
            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))
            # print("Precision..")
            # print(precision_curve)
            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

class DataEval():
  def __init__(self, net, model, _eval_test_with_detection, depth):
    self._batch_size = 1
    self._depth = depth
    self._height = 384
    self._width = 608
    self.dataset = icvl_inf_eval_video_Kash('val', [self._height, self._width], _eval_test_with_detection, split=1, )
    gpu_id = 1
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)   #setting the gpu id
    self._net = caffe.Net(net, model, caffe.TEST)  
    self.model = model
    self.net = net
    
    self._classes = ('__BG__',
                         'climbing', 'drop_bag', 'fall_down', 'fighting', 'get_into_car', 'get_off_car',
                         'not_group', 'not_phone_call', 'group_talking', 'group_walking',
                         'phone_call', 'pickup_bag', 'running', 'smoking', 'not_fall_down',
                         'not_smoking', "walking", "Sitting", "Standing")  #'walking_watching_mobile'
    
    self.thickness = 1
    self._eval_test_with_detection = _eval_test_with_detection  

  #def forward(self, total_mat_clip, total_clip_class, total_mat_video, total_video):
  def forward(self, mAPs, mR, mP, AP_accum_count, AP_accum, nC, total_mat_clip,total_clip_class):
    totaltime_matching = 0
    [video, gt_bboxes, gt_label, vid_name, pred, ratio_video, is_last] = self.dataset.next_rec_video(self._depth)
    num_frames = video.shape[0]                #reading the number of frames
    n_clips = num_frames / self._depth         #
    print(vid_name)
    #pred = pred #/ 1.25


    scores = np.zeros((n_clips, 3))
    scores_pred = np.zeros((1, nC)) 
    
    if not(os.path.isdir('/tarik/tarik/behavior-analysis/TCNN/Kashyap/{}'.format(vid_name))):
        os.makedirs('/tarik/tarik/behavior-analysis/TCNN/Kashyap/{}'.format(vid_name))

    fourcc_string="MJPG"
    fps = 8
    fourcc = cv2.VideoWriter_fourcc(*fourcc_string)   
    
    output_filename = '/tarik/tarik/behavior-analysis/TCNN/Kashyap/{}/{}.avi'.format(vid_name,vid_name)
    video_writer = cv2.VideoWriter(
            output_filename, fourcc, fps, (self._width*2, self._height*2))

    display = 0
    gt_len = len(gt_bboxes)
    gt_lab = len(gt_label)
    count = 0
    
    in_threshold = 0.5
    # outer_threshold = 0.6
    backgorund_threshold = 0.5

    scores_all = {}
    scores_pred_all = {}
    scire_c_all = {}
    scores_gt_all = {}
    thrwshold_all_i = {}

    scores_copy_all = {}
    scores_pred_copy_all = {}
   
    correct = []
    prediction_class = []
    prediction_score = []
    target_cls = []

    #print()
    gt_bboxes = np.array(gt_bboxes)   
    #print(gt_bboxes)
    cout_iou_best_zero = 0
    data_my = []
    count_frame_detect = 0
    last_frame_check = num_frames - self._depth 
    hop = 4
    for i in xrange(0, num_frames - self._depth, hop):#(n_clips):#(num_frames - self._depth): #n_clips
        in_start_in = datetime.datetime.now()
        
        te = video[i: i + self._depth].astype(np.float32,  copy=True)
        batch_clip = video[i: i + self._depth].transpose((3, 0, 1, 2))

        # pos_in_pred =  i * self._depth  #current clip first frame
        pos_in_pred = i
        object_pos = gt_bboxes[:, 0]  #Reading the index
        box_index_for_frame = np.where(object_pos==pos_in_pred)[0]
        if len(box_index_for_frame) ==0:
            continue
        
        curr_boxes = np.zeros((len(box_index_for_frame),4))
        #do the same for tube id
        curr_tub_id = np.zeros((len(box_index_for_frame),1))
        co = 0
        for findex in box_index_for_frame:
            curr_boxes[co] = gt_bboxes[findex,1:5] / 16.0
            curr_tub_id[co] = gt_bboxes[findex,6]
            co += 1
        #print(curr_boxes)
        mul=0
        
        frame_curr_number_box = len(curr_boxes)
        obj_dim_number = 1024
        #print(frame_curr_number_box)
        if frame_curr_number_box > obj_dim_number:
            print("Too many boxes found........!!!!!!")
            print("Converted to 1024; first 1024 are considered")
            frame_curr_number_box = obj_dim_number
        batch_tois = np.empty((obj_dim_number, 5))  #do it runtime bounding box detection
        #batch_tois = np.empty((frame_curr_number, 5))  #do it runtime bounding box detection
        for j in xrange(frame_curr_number_box):
            batch_tois[j] = np.concatenate((np.ones(1) * mul, curr_boxes[j]))
        for j in xrange(frame_curr_number_box, obj_dim_number):  #ading the last one until 24 rectangle filled
            batch_tois[j] = np.concatenate((np.ones(1) * mul, curr_boxes[frame_curr_number_box-1]))
        #batch_tois[0] = np.concatenate((np.ones(1) * mul, curr_bbox))
        
        self._net.blobs['data'].data[...] = batch_clip.astype(np.float32,
                                                                copy=False)
        self._net.blobs['tois'].data[...] = batch_tois.astype(np.float32,
                                                                copy=False)
        self._net.forward()
        s = self._net.blobs['loss'].data[...]

        scores_i = np.zeros((frame_curr_number_box))
        scores_pred_i = np.zeros((frame_curr_number_box))
        score_c_i = np.zeros((frame_curr_number_box))
        scores_gt_i = np.zeros((frame_curr_number_box))
        thresh_i_ = np.zeros((frame_curr_number_box))

        score_i_copy = np.zeros((frame_curr_number_box))
        scores_pred_i_copy = np.zeros((frame_curr_number_box))

        #read prediction for each candidate box
        max_iou_pos = -1
        iou_thres = 0.5
        #ground truth for current clips 
        gt_frame_pos = gt_bboxes[:, 0]  #Reading the index
        boxes_gt_index_frame = np.where(gt_frame_pos==pos_in_pred)[0]
        no_gt = False
        if len(boxes_gt_index_frame)==0:
            no_gt = True
        else:
            gt_boxes_frames = np.zeros((len(boxes_gt_index_frame),4))
            gt_label_frames = np.zeros((len(boxes_gt_index_frame),1))
            no_gt = False
        co_i = 0
        for findex in boxes_gt_index_frame:
            gt_boxes_frames[co_i] = (gt_bboxes[findex,1:5]).astype(np.uint32) #pred[findex,1:5] / 16
            gt_label_frames[co_i] = (gt_bboxes[findex,5]).astype(np.uint32) #pred[findex,1:5] / 16
            target_cls.append((gt_bboxes[findex,5]).astype(np.uint32))
            if gt_label_frames[co_i] != gt_label[findex]:
                print("Not same: %d != %d"%(gt_label_frames[co_i], gt_label[findex]))
            co_i += 1

        im1 = te[0].astype(np.int32)
        im1 = np.array(scipy.misc.toimage(im1))    
        
        frame_dict = {"frame":[], "objects":[],}   

        if_inserted = False
        detected = []
        
        s_copy = np.copy(s)
        # print("Processing frame: %d"%i)
        #for all the tube in the current clip 
        for j in xrange(frame_curr_number_box): #frame_curr_number
            pred_i = s[j].argmax()
            score_i =  s[j].max()
            already_change = False
            
            if already_change == False and pred_i == 0 and score_i < backgorund_threshold:
                s[j][pred_i]=0
                s_copy[j][pred_i]=0
                pred_i = s[j].argmax()
                score_i =  s[j].max()
                
                s_copy[j][pred_i]=0
                pred_i_s = s_copy[j].argmax()
                score_i_s =  s_copy[j].max()

                #print("..")
                in_threshold = 0.40
            else:
                in_threshold = 0.50
                s_copy[j][pred_i]=0
                pred_i_s = s_copy[j].argmax()
                score_i_s =  s_copy[j].max()      
            
            testing_the_candidate = True
            _bbox_ = curr_boxes[j] * 16.0
            _bbox_[0] = _bbox_[0] / ratio_video[1]
            _bbox_[2] = _bbox_[2] / ratio_video[1]
            _bbox_[1] = _bbox_[1] / ratio_video[0]
            _bbox_[3] = _bbox_[3] / ratio_video[0]
            _box_height = _bbox_[3] - _bbox_[1]+1
            if _box_height <80:  #100 pixels
                testing_the_candidate = False
            #in_threshold = 0.2
            # print("%d %f"%(pred_i,score_i))
            #"positions": [{"bbox": [696, 548, 952, 846], "id": 0, "behaviour_list": ["fighting"]}]
            if pred_i >= 0 and testing_the_candidate==True: #and score_i > in_threshold:  #if it is not a background
                in_frame_dict = {"score":[], "position":[], "behaviour":[], "id":[]}

                if no_gt == True:
                    
                    if pred_i > 0 and score_i > in_threshold:
                        current_pred_i = pred_i
                        if gt_label_frames[best_i]==9 or gt_label_frames[best_i]==10 or gt_label_frames[best_i]==16:
                            current_gt_label_frame_inside = 10

                        if current_pred_i==9 or current_pred_i==10 or current_pred_i==16:
                            current_pred_i = 10

                        correct.append(0)
                        prediction_class.append(current_pred_i)
                        prediction_score.append(score_i)

                        score_c_i[j] = 0
                        scores_i[j] = current_pred_i
                        scores_pred_i[j] = score_i
                        scores_gt_i[j] = 0
                        thresh_i_[j]=in_threshold

                        score_i_copy[j] = pred_i_s
                        scores_pred_i_copy[j] = score_i_s
                        
                        total_mat_clip[0,current_pred_i] += 1
                        total_clip_class[0,0] += 1
                        #for json file
                        in_frame_dict["behaviour"] = str(self._classes[current_pred_i])  # .append(self._classes[pred_i])
                        in_frame_dict["score"] = str(score_i)  # .append(score_i)

                        list_bbox_list = curr_boxes[j] * 16.0
                        list_bbox_list[0] = list_bbox_list[0] / ratio_video[1]
                        list_bbox_list[2] = list_bbox_list[2] / ratio_video[1]
                        list_bbox_list[1] = list_bbox_list[1] / ratio_video[0]
                        list_bbox_list[3] = list_bbox_list[3] / ratio_video[0]
                        list_bbox_list_i = list_bbox_list.astype(np.int32)
                        list_bbox_f = []
                        list_bbox_f.append(int(list_bbox_list_i[0]))
                        list_bbox_f.append(int(list_bbox_list_i[1]))
                        list_bbox_f.append(int(list_bbox_list_i[2]))
                        list_bbox_f.append(int(list_bbox_list_i[3]))
                        in_frame_dict["position"]=list_bbox_f

                        #id of the object
                        in_frame_dict["id"]=int(curr_tub_id[j])
                    #if_inserted = True
                    else:
                        scores_i[j] = 0
                        scores_pred_i[j] = 0
                        score_c_i[j] = 0
                        scores_gt_i[j] = 0
                        thresh_i_[j]=in_threshold

                        score_i_copy[j] = 0
                        scores_pred_i_copy[j] = 0
                else:
                    #print("Frame: %d pred= %f"%(i,score_i))
                    boxc = curr_boxes[j] * 16.0
                    boxcc = np.zeros((1,4))
                    boxcc[0] = boxc#.astype(np.int)
                    iou = bbox_iou_mihai(boxcc, gt_boxes_frames)
                    
                    # Compute iou with target boxes
                    
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    boxcc_c = boxcc.astype(np.int) 
                    # confusion matrix. If the iou between predicted box
                    # and gt box is larger than thr, take into consideration
                    # miou[best_i] - mean iou between each box and each gt box. if tube does not have a corresponding box to the gt, iou[i[ = 0
                    #if best_iou is zero donot append: cand not be
                    # if best iou is less than 10 it should be ignored
                    current_gt_label_frame_inside = gt_label_frames[best_i]
                    current_pred_i = pred_i
                    if gt_label_frames[best_i]==9 or gt_label_frames[best_i]==10 or gt_label_frames[best_i]==16:
                        current_gt_label_frame_inside = 10
                    if current_pred_i==9 or current_pred_i==10 or current_pred_i==16:
                        current_pred_i = 10

                    # if iou[best_i]<0.10 and pred_i >=0:
                    if iou[best_i] < 0.10 and current_pred_i >=0:
                        #if background ignore donot add it
                        #if pred_i is object and less than threshold; donot add it

                        #if score_i is greater than threshold add it to 
                        #if pred_i > 0 and score_i >= in_threshold:
                        if current_pred_i > 0 and score_i >= in_threshold:
                            correct.append(0)
                            #prediction_class.append(pred_i)
                            prediction_class.append(current_pred_i)
                            prediction_score.append(score_i)

                            score_c_i[j] = 0
                            scores_i[j] = current_pred_i #pred_i
                            scores_pred_i[j] = score_i
                            scores_gt_i[j] = 0
                            thresh_i_[j]=in_threshold

                            score_i_copy[j] = pred_i_s
                            scores_pred_i_copy[j] = score_i_s

                            cout_iou_best_zero += 1
                            #total_mat_clip[0, pred_i] += 1
                            total_mat_clip[0, current_pred_i] += 1
                            total_clip_class[0, 0] += 1

                            # for json file
                            #in_frame_dict["behaviour"] = str(self._classes[pred_i])  # .append(self._classes[pred_i])
                            in_frame_dict["behaviour"] = str(self._classes[current_pred_i])  # .append(self._classes[pred_i])
                            in_frame_dict["score"] = str(score_i)  # .append(score_i)

                            list_bbox_list = curr_boxes[j] * 16.0
                            list_bbox_list[0] = list_bbox_list[0] / ratio_video[1]
                            list_bbox_list[2] = list_bbox_list[2] / ratio_video[1]
                            list_bbox_list[1] = list_bbox_list[1] / ratio_video[0]
                            list_bbox_list[3] = list_bbox_list[3] / ratio_video[0]
                            list_bbox_list_i = list_bbox_list.astype(np.int32)
                            list_bbox_f = []
                            list_bbox_f.append(int(list_bbox_list_i[0]))
                            list_bbox_f.append(int(list_bbox_list_i[1]))
                            list_bbox_f.append(int(list_bbox_list_i[2]))
                            list_bbox_f.append(int(list_bbox_list_i[3]))
                            in_frame_dict["position"]=list_bbox_f
                            #id of the object
                            in_frame_dict["id"]=int(curr_tub_id[j])
                        else:
                            scores_i[j] = 0
                            scores_pred_i[j] = 0
                            score_c_i[j] = 0
                            scores_gt_i[j] = 0
                            thresh_i_[j]=in_threshold

                            score_i_copy[j] = 0
                            scores_pred_i_copy[j] = 0
                        #if_inserted = True
                    #it should be higher then the threshold to be considered as true poisitve; 0.5 or 0.3
                    #elif iou[best_i] > iou_thres and pred_i == gt_label_frames[best_i] and score_i >= in_threshold: #and best_i not in detected:
                    elif iou[best_i] > iou_thres and current_pred_i == current_gt_label_frame_inside and score_i >= in_threshold: #and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                        #prediction_class.append(pred_i)
                        prediction_class.append(current_pred_i)
                        prediction_score.append(score_i)

                        score_c_i[j] = 1
                        scores_i[j] = current_pred_i #pred_i
                        scores_pred_i[j] = score_i
                        scores_gt_i[j] = gt_label_frames[best_i]
                        thresh_i_[j]=in_threshold

                        score_i_copy[j] = pred_i_s
                        scores_pred_i_copy[j] = score_i_s

                        # cv2.rectangle(im1, (boxcc_c[0,0], boxcc_c[0,1]), (boxcc_c[0,2], boxcc_c[0,3]),
                        #         color=(0, 0, 255))
                        #total_mat_clip[int(gt_label_frames[best_i]), pred_i] += 1
                        total_mat_clip[int(gt_label_frames[best_i]), current_pred_i ] += 1
                        total_clip_class[int(gt_label_frames[best_i]), 0] += 1

                        # for json file
                        #in_frame_dict["behaviour"] = str(self._classes[pred_i])  # .append(self._classes[pred_i])
                        in_frame_dict["behaviour"] = str(self._classes[current_pred_i])  # .append(self._classes[pred_i])
                        in_frame_dict["score"] = str(score_i)  # .append(score_i)

                        list_bbox_list = curr_boxes[j] * 16.0
                        list_bbox_list[0] = list_bbox_list[0] / ratio_video[1]
                        list_bbox_list[2] = list_bbox_list[2] / ratio_video[1]
                        list_bbox_list[1] = list_bbox_list[1] / ratio_video[0]
                        list_bbox_list[3] = list_bbox_list[3] / ratio_video[0]
                        list_bbox_list_i = list_bbox_list.astype(np.int32)
                        list_bbox_f = []
                        list_bbox_f.append(int(list_bbox_list_i[0]))
                        list_bbox_f.append(int(list_bbox_list_i[1]))
                        list_bbox_f.append(int(list_bbox_list_i[2]))
                        list_bbox_f.append(int(list_bbox_list_i[3]))
                        in_frame_dict["position"]=list_bbox_f
                        #if_inserted = True
                        #id of the object
                        in_frame_dict["id"]=int(curr_tub_id[j])
                    else:
                        ignore_the_candidate = False
                        if iou[best_i] > iou_thres and current_pred_i  >= 0: # if iou >= threshold and pred_i >= 0; false positive
                            ignore_the_candidate = False
                        elif iou[best_i] <= iou_thres and score_i >= in_threshold: # if iou <= threshold and score_i > threshold; false positive
                            ignore_the_candidate = False
                        elif iou[best_i] <= iou_thres and current_pred_i == 0 and score_i < in_threshold: 
                            ignore_the_candidate = True
                        elif iou[best_i] <= iou_thres and current_pred_i > 0 and score_i < in_threshold: 
                            ignore_the_candidate = True
                        
                        # if iou is less than threshold and bg and score < threshold; ignore
                        # if iou is less than threshold and not bg and score < threshold; ignore 
                        if ignore_the_candidate == False:
                            correct.append(0)
                            # cv2.rectangle(im1, (boxcc_c[0,0], boxcc_c[0,1]), (boxcc_c[0,2], boxcc_c[0,3]),
                            #         color=(255, 255, 255))

                            #prediction_class.append(pred_i)   
                            prediction_class.append(current_pred_i)   
                            prediction_score.append(score_i)

                            score_c_i[j] = 0
                            scores_i[j] = current_pred_i
                            scores_pred_i[j] = score_i
                            scores_gt_i[j] = gt_label_frames[best_i]
                            thresh_i_[j]=in_threshold

                            score_i_copy[j] = pred_i_s
                            scores_pred_i_copy[j] = score_i_s

                            #print("others: %s gt: %s score: %f"%(self._classes[int(pred_i)], self._classes[int(gt_label_frames[best_i])], score_i))
                            #total_mat_clip[int(gt_label_frames[best_i]), pred_i] += 1
                            total_mat_clip[int(gt_label_frames[best_i]), current_pred_i] += 1
                            total_clip_class[int(gt_label_frames[best_i]), 0] += 1

                            # for json file
                            #in_frame_dict["behaviour"] = str(self._classes[pred_i])  # .append(self._classes[pred_i])
                            in_frame_dict["behaviour"] = str(self._classes[current_pred_i])  # .append(self._classes[pred_i])
                            in_frame_dict["score"] = str(score_i)  # .append(score_i)

                            list_bbox_list = curr_boxes[j] * 16.0
                            list_bbox_list[0] = list_bbox_list[0] / ratio_video[1]
                            list_bbox_list[2] = list_bbox_list[2] / ratio_video[1]
                            list_bbox_list[1] = list_bbox_list[1] / ratio_video[0]
                            list_bbox_list[3] = list_bbox_list[3] / ratio_video[0]
                            list_bbox_list_i = list_bbox_list.astype(np.int32)
                            list_bbox_f = []
                            list_bbox_f.append(int(list_bbox_list_i[0]))
                            list_bbox_f.append(int(list_bbox_list_i[1]))
                            list_bbox_f.append(int(list_bbox_list_i[2]))
                            list_bbox_f.append(int(list_bbox_list_i[3]))
                            in_frame_dict["position"]=list_bbox_f

                            #id of the object
                            in_frame_dict["id"]=int(curr_tub_id[j])
                        else:
                            scores_i[j] = 0
                            scores_pred_i[j] = 0
                            score_c_i[j] = 0
                            scores_gt_i[j] = 0
                            thresh_i_[j]=in_threshold

                            score_i_copy[j] = 0
                            scores_pred_i_copy[j] = 0

                    #detection_box.append(boxcc)  #adding it detection box
                #for jason append
                frame_dict["objects"].append(in_frame_dict)
                frame_dict["frame"]=i
                if_inserted = True
                if i == last_frame_check-1:  #just the last frame for video check
                    frame_dict_1 = {"frame":[], "objects":[],}   
                    frame_dict_1["objects"].append(in_frame_dict)
                    frame_dict_1["frame"]=i+1

                    frame_dict_2 = {"frame":[], "objects":[],}   
                    frame_dict_2["objects"].append(in_frame_dict)
                    frame_dict_2["frame"]=i+2

                    frame_dict_3 = {"frame":[], "objects":[],}   
                    frame_dict_3["objects"].append(in_frame_dict)
                    frame_dict_3["frame"]=i+3

                    frame_dict_4 = {"frame":[], "objects":[],}   
                    frame_dict_4["objects"].append(in_frame_dict)
                    frame_dict_4["frame"]=i+4

                    frame_dict_5 = {"frame":[], "objects":[],}   
                    frame_dict_5["objects"].append(in_frame_dict)
                    frame_dict_5["frame"]=i+5

                    frame_dict_6 = {"frame":[], "objects":[],}   
                    frame_dict_6["objects"].append(in_frame_dict)
                    frame_dict_6["frame"]=i+6

            else:
                scores_i[j] = 0
                scores_pred_i[j] = 0
                score_c_i[j] = 0
                scores_gt_i[j] = 0
                thresh_i_[j]=in_threshold

                score_i_copy[j] = 0
                scores_pred_i_copy[j] = 0

        # cout_iou_best_zero
        # Compute Average Precision (AP) per class

        duration_in = datetime.datetime.now() - in_start_in  
        totaltime_matching += duration_in.total_seconds()*1000

        scores_all[i] = scores_i
        scores_pred_all[i] = scores_pred_i
        scire_c_all[i] = score_c_i
        scores_gt_all[i] = scores_gt_i  
        thrwshold_all_i[i] = thresh_i_   

        scores_copy_all[i] = score_i_copy
        scores_pred_copy_all[i] = scores_pred_i_copy

        
        if if_inserted:
            data_my.append(frame_dict)   
            if i == last_frame_check-1:  #just the last frame for video check
                data_my.append(frame_dict_1)
                data_my.append(frame_dict_2)   
                data_my.append(frame_dict_3)
                data_my.append(frame_dict_4)
                data_my.append(frame_dict_5)
                data_my.append(frame_dict_6)

        count_frame_detect += 1
        #cv2.imshow('Image-',im1)
        #cv2.waitKey(0)

    print("Saving the Json file")
    basename, _ = os.path.splitext(os.path.basename(vid_name))
    if self._eval_test_with_detection == True:
        json_name = "Kashyap/jason_folder/"+basename+".json"
    else:
        json_name = "Kashyap/jason_folder/"+basename+".json"
    with open(json_name, 'wb') as outfile:
        json.dump(data_my, outfile)

    correct = True
    for i in xrange(0, num_frames - self._depth, hop): #(n_clips): #(num_frames - self._depth)

        in_start_in = datetime.datetime.now()
        te = video[i: i + self._depth].astype(np.float32, copy=True)

        pos_in_pred = i

        object_pos = gt_bboxes[:, 0] 
        box_index_for_frame = np.where(object_pos==pos_in_pred)[0]
        if len(box_index_for_frame) ==0:
            continue
        curr_boxes = np.zeros((len(box_index_for_frame),4))
        curr_tub_id = np.zeros((len(box_index_for_frame),1))
        co = 0
        for findex in box_index_for_frame:
            curr_boxes[co] = gt_bboxes[findex,1:5].astype(np.int)
            curr_tub_id[co] = gt_bboxes[findex,6]
            co += 1
        mul=0
        frame_curr_number = len(curr_boxes)
        

        all_sc_i = scores_all.get(i)
        all_pred_i = scores_pred_all.get(i)
        all_c_i = scire_c_all.get(i)
        all_gt_i = scores_gt_all.get(i)
        thresh_i_ = thrwshold_all_i.get(i)

        all_sc_copy_i = scores_copy_all.get(i)
        all_pred_copy_i = scores_pred_copy_all.get(i)
        
        #1st
        im1 = te[0].astype(np.int32)
        im1 = np.array(scipy.misc.toimage(im1))
        
        #print(scores[i])
        draw_box = True

        #if max_iou_pos != -1:
        #    draw_box = True
            #print("Draw boxes")
        one = False
        two = False
        three = False
        four = False
        five = False
        six = False
        seven = False
        eight = False
        
        if draw_box == True:
            curr = (curr_boxes[max_iou_pos]).astype(np.uint32)
            if curr[0] == None or curr[1]==None:
                print("None object found....")
                continue
            # print(curr)
            for j in xrange(frame_curr_number):
                if all_sc_i[j] == 1:
                    c_array = (0, 51, 0)
                elif all_sc_i[j] == 2:
                    c_array = (51, 204, 51)
                elif all_sc_i[j] == 3:
                    c_array = (255, 204, 255)
                elif all_sc_i[j] == 4:
                    c_array = (255, 0, 255)
                elif all_sc_i[j] == 5:
                    c_array = (102, 0, 102)
                elif all_sc_i[j] == 6:
                    c_array = (204, 0, 101)
                elif all_sc_i[j] == 7:
                    c_array = (102, 51, 153)
                elif all_sc_i[j] == 8:
                    c_array = (51, 102, 102)
                elif all_sc_i[j] == 9:
                    c_array = (0, 102, 255)
                elif all_sc_i[j] == 10:
                    c_array = (255, 102, 51)
                elif all_sc_i[j] == 11:
                    c_array = (255, 255, 255)
                elif all_sc_i[j] == 12:
                    c_array = (255, 204, 0)
                elif all_sc_i[j] == 13:
                    c_array = (153, 255, 255)
                elif all_sc_i[j] == 14:
                    c_array = (255, 0, 125)
                elif all_sc_i[j] == 15:
                    c_array = (255, 125, 125)
                elif all_sc_i[j] == 16:
                    c_array = (125, 255, 255)
                elif all_sc_i[j] == 17:
                    c_array = (50, 255, 125)
                elif all_sc_i[j] == 18:
                    c_array = (50, 50, 125)
                elif all_sc_i[j] == 19:
                    c_array = (102, 153, 255)
                else:
                    c_array = (153, 51, 0)

                if all_sc_i[j] >= 0 and all_c_i[j]==1:
                    try:
                        in_threshold_i = thresh_i_[j]
                        curr_1 = (curr_boxes[j]).astype(np.uint32)
                        c_id = (curr_tub_id[j]).astype(np.uint32)
                        pt1 = int(curr_1[0]), int(curr_1[1])
                        pred_text = "%d %s->%.2f"%(c_id, self._classes[int(all_sc_i[j])], all_pred_i[j])
                        text_size = cv2.getTextSize(
                             pred_text, cv2.FONT_HERSHEY_PLAIN, 0.5, self.thickness)
                        center = pt1[0] + 10, pt1[1] + text_size[0][1]
                        center_1 = pt1[0] + 10, pt1[1] + text_size[0][1] + text_size[0][1] + text_size[0][1]
                        pred_text_1 = "%s" % (self._classes[int(all_gt_i[j])])

                        pred_text_2 = "%s->%.2f"%(self._classes[int(all_sc_copy_i[j])], all_pred_copy_i[j])
                        center_2 = pt1[0] + 10, pt1[1] + text_size[0][1] + text_size[0][1]

                        if int(all_gt_i[j]) != 0 and all_pred_i[j] > in_threshold_i:  
                            cv2.rectangle(im1, (curr_1[0], curr_1[1]), (curr_1[2], curr_1[3]),
                                    color=c_array)
                            cv2.putText(im1, pred_text, center, cv2.FONT_HERSHEY_PLAIN,
                                    0.5, c_array, self.thickness)
                            cv2.putText(im1, pred_text_2, center_2, cv2.FONT_HERSHEY_PLAIN,
                                    0.5, c_array, self.thickness)
                            if self._eval_test_with_detection == False:
                                cv2.putText(im1, pred_text_1, center_1, cv2.FONT_HERSHEY_PLAIN,
                                        0.5, c_array, self.thickness)
                        elif int(all_gt_i[j]) == 0  and all_pred_i[j] > in_threshold_i and all_sc_i[j]>=0: # 
                            cv2.rectangle(im1, (curr_1[0], curr_1[1]), (curr_1[2], curr_1[3]),
                                          color=c_array)
                            cv2.putText(im1, pred_text, center, cv2.FONT_HERSHEY_PLAIN,
                                        0.5, c_array, self.thickness)
                            cv2.putText(im1, pred_text_2, center_2, cv2.FONT_HERSHEY_PLAIN,
                                        0.5, c_array, self.thickness)
                            if self._eval_test_with_detection == False:
                                 cv2.putText(im1, pred_text_1, center_1, cv2.FONT_HERSHEY_PLAIN,
                                             0.5, c_array, self.thickness) 
                    except:
                        mine = False
                elif all_sc_i[j] >= 0 and all_c_i[j]==0:
                    try:
                        #     '_BG_',
                        #  'climbing', 'drop_bag', 'fall_down', 'fighting', 'get_into_car', 'get_off_car',
                        #  'get_off_cycle', 'get_onto_cycle',
                        #  'group_talking', 'group_walking',
                        #  'phone_call', 'pickup_bag', 'running', 'smoking', 'walking_watching_mobile',
                        #  'walking_with_child', "walking", "Sitting", "Standing"
                        in_threshold_i = thresh_i_[j]
                        curr_1 = (curr_boxes[j]).astype(np.uint32)
                        c_id = (curr_tub_id[j]).astype(np.uint32)
                        # print(curr_1)
                        pt1 = int(curr_1[0]), int(curr_1[1])
                        # pred_text = "%s->%.2f"%(self._classes[int(all_sc_i[j])], all_pred_i[j])
                        pred_text = "%d %s->%.2f"%(c_id, self._classes[int(all_sc_i[j])], all_pred_i[j])
                        text_size = cv2.getTextSize(
                             pred_text, cv2.FONT_HERSHEY_PLAIN, 0.5, self.thickness)
                        center = pt1[0] + 10, pt1[1] + text_size[0][1]
                        center_1 = pt1[0] + 10, pt1[1] + text_size[0][1] + text_size[0][1] + text_size[0][1]
                        pred_text_1 = "%s" % (self._classes[int(all_gt_i[j])])

                        pred_text_2 = "%s->%.2f"%(self._classes[int(all_sc_copy_i[j])], all_pred_copy_i[j])
                        center_2 = pt1[0] + 10, pt1[1] + text_size[0][1] + text_size[0][1]

                        # if int(all_gt_i[j]) != 0 and all_pred_i[j] > in_threshold_i:
                        cv2.rectangle(im1, (curr_1[0], curr_1[1]), (curr_1[2], curr_1[3]),
                                color= c_array)#(255, 255, 255))
                        cv2.putText(im1, pred_text, center, cv2.FONT_HERSHEY_PLAIN,
                                0.5, c_array, self.thickness) #(0, 255, 0), self.thickness)
                        cv2.putText(im1, pred_text_2, center_2, cv2.FONT_HERSHEY_PLAIN,
                                0.5, c_array, self.thickness) #(0, 255, 0), self.thickness)
                        # if self._eval_test_with_detection == False:
                        #     cv2.putText(im1, pred_text_1, center_1, cv2.FONT_HERSHEY_PLAIN,0.5, c_array, self.thickness)  #(0, 255, 0)
                        # elif int(all_gt_i[j]) == 0  and all_sc_i[j]>=0 and all_pred_i[j] > in_threshold_i:  
                        #     cv2.rectangle(im1, (curr_1[0], curr_1[1]), (curr_1[2], curr_1[3]),
                        #                   color=c_array) #(0, 255, 255))
                        #     cv2.putText(im1, pred_text, center, cv2.FONT_HERSHEY_PLAIN, 0.5, c_array, self.thickness) #
                        #     cv2.putText(im1, pred_text_2, center_2, cv2.FONT_HERSHEY_PLAIN, 0.5, c_array, self.thickness) #
                        #     if self._eval_test_with_detection == False:
                        #         cv2.putText(im1, pred_text_1, center_1, cv2.FONT_HERSHEY_PLAIN, 0.5, c_array, self.thickness) # (0, 255, 255)
                    except:
                        mine = False
                        print("error")
            one = True
            

        #8th
        #if one == True:
        display = 0
        #else:
        #    display = 1
        if display == 1:
            cv2.imshow('image1',im1)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()  

        if video_writer is not None:
            if one == True:
                if self._eval_test_with_detection == True:
                    r1 = 2
                    r2 = 2
                    im1_1 = cv2.resize(im1, None, None, fx=r2, fy=r1,
                                        interpolation=cv2.INTER_LINEAR)
                    video_writer.write(im1_1)
                else:
                    r1 = 2
                    r2 = 2
                    im1_1 = cv2.resize(im1, None, None, fx=r2, fy=r1,
                                        interpolation=cv2.INTER_LINEAR) 
                    video_writer.write(im1_1)
           
        
        batch_clip = None   
        batch_tois = None
        te = None
        s=None
    #pred_1 = s.argmax()
    video_writer.release()
    
    return is_last, correct #, total_mat_clip, total_clip_class
