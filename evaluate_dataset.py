from os.path import join, realpath, dirname, exists, isdir
from os import listdir
import os
import logging
import glob
import numpy as np
import json
from collections import OrderedDict
import cv2
from PIL import Image, ImageColor
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import time
from scipy import ndimage
import scipy.misc
from skimage.metrics import structural_similarity
import pickle
import threading

# own imports 
from utility import *
import sys
sys.path.append("../SiamMask")

from utils.log_helper import init_log, add_file_handler
from utils.load_helper import load_pretrain
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from utils.benchmark_helper import dataset_zoo
from utils.anchors import Anchors
from utils.tracker_config import TrackerConfig
from utils.config_helper import load_config
from utils.pyvotkit.region import vot_overlap, vot_float2str

from tools.test import *

import argparse
import logging

import torch
from torch.autograd import Variable
import torch.nn.functional as F

thrs = np.arange(0.40, 0.45, 0.05)


parser = argparse.ArgumentParser(description='Test SiamMask')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('--config', dest='config', required=True, help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--dataset', dest='dataset', default='a2d2',
                    help='datasets')
parser.add_argument('--datapath', dest='datapath', default='.',
                    help='datapath')

def MultiBatchIouMeter(thrs, outputs, targets, rgb, end_tracks, start=None, end=None):
    targets = np.array(targets)
    outputs = np.array(outputs)
    num_frame = targets.shape[0]
    num_object = len(rgb)
    res = np.zeros((num_object, len(thrs)), dtype=np.float32)
    output_max_id = np.argmax(outputs, axis=0).astype('uint8')+1
    outputs_max = np.max(outputs, axis=0)
    for k, thr in enumerate(thrs):
        output_thr = outputs_max > thr
        for j in range(num_object):
            target_j = [np.logical_and.reduce(target == rgb[j], axis = -1) for target in targets]
            start_frame = 0
            end_frame = end_tracks[j]
            iou = []
            for i in range(start_frame, end_frame):
                pred = (output_thr[i] * output_max_id[i]) == (j+1)
                mask_sum = (pred == 1).astype(np.uint8) + (target_j[i] > 0).astype(np.uint8)
                intxn = np.sum(mask_sum == 2)
                union = np.sum(mask_sum > 0)
                if union > 0:
                    iou.append(intxn / union)
                elif union == 0 and intxn == 0:
                    iou.append(1)
            avg = np.mean(iou)
            res[j, k] = np.mean(iou)
    return res

def get_instances(im):
    img = np.array(Image.open(im))
    obj_ids = np.unique(img)
    return [obj_id % 1000 for obj_id in obj_ids]


def eval_kitti(model, data, hp, mask_enable=True, refine_enable=True, mot_enable=False, device='cpu'):
    gold_stop_track_dict = {}
    estimate_gold_stop_track_dict = {}
    pred_stop_track_dict = {}
    images_to_consider = 50
    for scene in data:
        start = 0 
        end = len(data[scene]['camera'])-1
        gold_stop_track_dict[scene] = {}
        pred_stop_track_dict[scene] = {}
        estimate_gold_stop_track_dict[scene] = {}
        start_im = data[scene]['annotations'][0]
        img = np.array(Image.open(start_im))
        obj_ids = np.unique(img)
        images_to_consider = min([images_to_consider, len(data[scene]['annotations'])-1])
        for obj in obj_ids:
            if not obj//1000 == 0:
                pred_stop_track_dict[scene][obj%1000] = []
                gold_stop_track_dict[scene][obj%1000] = []
                estimate_gold_stop_track_dict[scene][obj%1000] = []
                print("class ", obj//1000)
                collect_states=[]
                current_instance = obj%1000
                for index, im in enumerate(data[scene]['annotations'][:images_to_consider]):
                    #instances = get_instances(im)
                    im = data[scene]['camera'][index]
                    cv_im = cv2.imread(im)
                    anno_im = data[scene]['annotations'][index]
                    anno_array = np.array(Image.open(anno_im))
                    if index == start:  # init
                        mask = anno_array == obj
                        x, y, w, h = cv2.boundingRect((mask).astype(np.uint8))
                        cx, cy = x + w/2, y + h/2
                        if cx == 0 and cy == 0:
                            break
                        target_pos = np.array([cx, cy])
                        target_sz = np.array([w, h])
                        state = siamese_init(cv_im, target_pos, target_sz, model, hp, device=device)  # init tracker
                        collect_states.append(state)
                    elif end >= index > start:  # tracking
                        state = siamese_track(state, cv_im, mask_enable, refine_enable, device=device)  # track
                        collect_states.append(state)
                        current_objects = np.unique(anno_array)
                        current_instances = [current_object% 1000 for current_object in current_objects]
                        if current_instance not in current_instances:
                            gold_stop_track_dict[scene][obj%1000].append(index)
                            #print("Gold scene ", scene, " obj ", obj//1000," instance ", obj%1000,  " stop track ", index)

                        current_instance_mask = anno_array
                        current_instance_mask[current_instance_mask%1000 == current_instance] = 1
                        #print(current_instance_mask)
                        mask = state['mask']
                        check_mask = state['mask']
                        check_mask[check_mask>thrs] = 1
                        check_mask[check_mask<=thrs] = 0 
                        mask_sum = check_mask + current_instance_mask
                        intersec = np.sum(mask_sum[mask_sum==2])
                        if intersec == 0:
                            estimate_gold_stop_track_dict[scene][obj%1000].append(index)
                            #print("estimate scene ", scene, " obj ", obj//1000," instance ", obj%1000,  " stop track ", index)
                        if index > start+1:
                            current_location = state['minAreaRect']
                            current_im = cv_im 
                            prev_state = collect_states[-1]
                            prev_location = prev_state['minAreaRect']
                            prev_im = cv2.imread(data[scene]['camera'][index-1])
                            if prev_location != [] and current_location != []:
                                current_cropped_image = crop_minAreaRect(current_im, current_location)
                                prev_cropped_image = crop_minAreaRect(prev_im, prev_location)
                                common_size = (150, 150)
                                try:
                                    prev_cropped_image= cv2.resize(prev_cropped_image,common_size, interpolation=cv2.INTER_CUBIC)
                                    current_cropped_image = cv2.resize(current_cropped_image,common_size, interpolation=cv2.INTER_CUBIC)
                                    score, diff = structural_similarity(prev_cropped_image, current_cropped_image, full=True, multichannel=True)
                                except Exception as e:
                                    score=1.0
                                if score < 0.35:
                                    pred_stop_track_dict[scene][obj%1000].append(index)
                    if gold_stop_track_dict[scene][obj%1000] != []:
                        break
                pickle.dump(gold_stop_track_dict, open("pickle_files/gold_stop_track_dict2.pickle", "wb"))
                pickle.dump(pred_stop_track_dict, open("pickle_files/pred_stop_track_dict2.pickle", "wb"))
                pickle.dump(estimate_gold_stop_track_dict, open("pickle_files/estimate_gold_stop_track_dict2.pickle", "wb"))
                print(scene)
                print("Gold: obj ", obj//1000," instance ", obj%1000,  " stop track ", gold_stop_track_dict[scene][obj%1000])
                print("Estimate gold: obj ", obj//1000," instance ", obj%1000,  " stop track ", estimate_gold_stop_track_dict[scene][obj%1000])
                print("Prediction  obj ", obj//1000," instance ", obj%1000,  " stop track ", pred_stop_track_dict[scene][obj%1000])
    return gold_stop_track_dict, pred_stop_track_dict

def eval_a2d2(model, data, hp, mask_enable=True, refine_enable=True, mot_enable=False, device='cpu'): 

    with open('../../Uni/9.Semester/AP/class_list.json') as json_file: 
        lookup = json.load(json_file) 
    lookup = {ImageColor.getcolor(k, "RGB"):v for k,v in lookup.items()}
    #iou_dict = {lookup[key]:[0,0] for key in lookup.keys()}
    #print("init iou-dict ", iou_dict)
    iou_dict = {}
    #print(lookup)
    stop_track_dict = {}
    mask_dict = {}
    # change parameters here
    use_gold_crit = True
    save_output_dir = args.dataset+"_gold_results2"
    os.makedirs(save_output_dir, exist_ok=True) 
    for scene in data: 
        iou_dict[scene] = {}
        stop_track_dict[scene] = {}
        mask_dict[scene] = {}
        print("It's  time for scene ", scene)
        start = 0 
        #end = len(scene)
        end = 12
        print("Using ", end, "frames and classes.")
        len_scence = min(len(scene), end)
        if len_scence == 0:
            continue
        camera = data[scene]['camera'][:len_scence]
        annotations = data[scene]['annotations'][:len_scence]
        #anno_array = [np.array(Image.open(x).convert("L")) for x in annotations]
        color_array = [np.array(Image.open(x)) for x in annotations]
        reshaped_array = color_array[start].reshape((color_array[0].shape[0]* color_array[0].shape[1], 3))
        rgb = np.unique(reshaped_array, axis=0)
        rgb = rgb[:min(len_scence, len(rgb))]
        color_track = [annotation.astype(np.uint8) for annotation in color_array]
        pred_masks = np.zeros((len(rgb), len(camera), color_track[0].shape[0], color_track[0].shape[1]))-1
        end_tracks = []
        for object_index, code in enumerate(rgb):
            rgb_tuple = tuple(code)
            mask_dict[scene][rgb_tuple] = []
            end_track = len(camera)-1
            collect_states = []
            for camera_index, pic in enumerate(camera):
                im = cv2.imread(pic)
                if camera_index == start:  # init
                    mask = np.logical_and.reduce(color_track[0] == code, axis = -1)
                    mask_dict[scene][rgb_tuple].append(mask)
                    x, y, w, h = cv2.boundingRect((mask).astype(np.uint8))
                    cx, cy = x + w/2, y + h/2
                    if cx == 0 and cy == 0:
                        break
                    target_pos = np.array([cx, cy])
                    target_sz = np.array([w, h])
                    state = siamese_init(im, target_pos, target_sz, model, hp, device=device)  # init tracker
                elif end >= camera_index > start:  # tracking
                    state = siamese_track(state, im, mask_enable, refine_enable, device=device)  # track
                    current_anno = np.array(Image.open(annotations[camera_index]))
                    current_all_instances_mask = np.logical_and.reduce(current_anno == code, axis = -1).astype(np.uint8)
                    mask = state['mask']
                    check_mask = state['mask']
                    check_mask[check_mask>thrs] = 1
                    check_mask[check_mask<=thrs] = 0 
                    mask_sum = check_mask + current_all_instances_mask
                    intersec = np.sum(mask_sum[mask_sum==2])

                    prev_mask = mask_dict[scene][rgb_tuple][-1]
                    prev_im = np.array(Image.open(camera[camera_index-1]))
                    current_im = np.array(Image.open(camera[camera_index]))

                    #current_pixel = current_im[mask > thrs]
                    #prev_pixel = prev_im[prev_mask > thrs]
                
                    if use_gold_crit:
                        if intersec == 0:
                            relevant_region = current_anno[check_mask ==1]
                            shading_error = False
                            if not relevant_region.size == 0:
                                predicted_classes = np.unique(relevant_region, axis=0)
                                shading_error = False
                                for predicted_class in predicted_classes:
                                    predicted_colour = (int(predicted_class[0]), int(predicted_class[1]), int(predicted_class[2]))
                                    if lookup[rgb_tuple].strip("0123456789 ") in lookup[predicted_colour].strip("0123456789 "):
                                        shading_error = True
                            if not shading_error:
                                end_track = camera_index
                                break
                    else:
                        if camera_index > start+1:
                            current_location = state['minAreaRect']
                            current_im = im 
                            prev_state = collect_states[-1]
                            prev_location = prev_state['minAreaRect']
                            prev_im = cv2.imread(camera[camera_index-1])
                            if prev_location != [] and current_location != []:
                                current_cropped_image = crop_minAreaRect(current_im, current_location)
                                prev_cropped_image = crop_minAreaRect(prev_im, prev_location)
                                common_size = (150, 150)
                                try:
                                    prev_cropped_image= cv2.resize(prev_cropped_image,common_size, interpolation=cv2.INTER_CUBIC)
                                    current_cropped_image = cv2.resize(current_cropped_image,common_size, interpolation=cv2.INTER_CUBIC)

                                    score, diff = structural_similarity(prev_cropped_image, current_cropped_image, full=True, multichannel=True)
                                except Exception as e:
                                    score=1.0
                                if score < 0.35:
                                    end_track = camera_index
                                    break
                        
                    collect_states.append(state)
                if end >= camera_index >= start:
                    pred_masks[object_index, camera_index, :, :] = mask
                    #mask_dict[scene][rgb_tuple].append(mask)
                
            end_tracks.append(end_track)
            stop_track_dict[scene][rgb_tuple] = end_track
        pickle.dump(mask_dict, open(save_output_dir + "/"+"mask_dict.pickle", "wb"))
        #mask_dict = pickle.load(open(save_output_dir + "/"+"mask_dict.pickle", "rb")) 
        print("Stop track dict :")
        for sc in stop_track_dict:
            print(sc)
            print(stop_track_dict[sc])
        pickle.dump(stop_track_dict, open(save_output_dir + "/"+"stop_track_dict.pickle", "wb"))
        stop_track_dict = pickle.load(open(save_output_dir + "/"+"stop_track_dict.pickle", "rb"))
        if len(annotations) == len(camera):
            multi_mean_iou = MultiBatchIouMeter(thrs, pred_masks, color_array, rgb,end_tracks,start=None, end=None)
            for i, code in enumerate(rgb):
                for j, thr in enumerate(thrs):
                    rgb_tuple = tuple(code)
                    label = lookup[rgb_tuple]
                    iou = multi_mean_iou[i, j]
                    #logger.info('{:18s} {:18s}: IOU {:.4f}'.format(scene, label, iou))
                #iou_dict[scene][label][0]+=1
                iou_dict[scene][label] =iou
            pickle.dump(iou_dict, open(save_output_dir + "/"+"iou_dict.pickle", "wb"))
            iou_dict = pickle.load(open(save_output_dir + "/"+"iou_dict.pickle", "rb")) 
            object_ious = {lookup[key]:[0,0] for key in lookup.keys()}
            for scene in iou_dict:
                for obj in iou_dict[scene]:
                    object_ious[obj][0] +=1
                    object_ious[obj][1] +=iou_dict[scene][obj]
            print("Current mean IoU: ")
            for obj in object_ious:
                if object_ious[obj][0] != 0:
                    print("Object: ", obj, " seen ", object_ious[obj][0], " times with mean IoU: ", object_ious[obj][1]/object_ious[obj][0])
                else: 
                    print("Object: ", obj, " seen ", object_ious[obj][0], " times with mean IoU: ", 0)
def main():
    #path_to_data = "../../Uni/9.Semester/AP/dataset"
    #model_path = "experiments/siammask_sharp/SiamMask_DAVIS.pth"
    #config_path = "experiments/siammask_sharp/config_davis.json"
    global args, logger, v_id
    args = parser.parse_args()
    cfg = load_config(args)
    init_log('global', logging.INFO)
    logger = logging.getLogger('global')
    logger.info(args)

    # setup model
    if args.arch == 'Custom':
        from experiments.siammask_sharp.custom import Custom
        model = Custom(anchors=cfg['anchors'])
    else:
        parser.error('invalid architecture: {}'.format(args.arch))

    if args.resume:
        model = load_pretrain(model, args.resume)
    model.eval()
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
    model = model.to(device)

    # setup dataset
    if args.dataset == "a2d2":
        data = load_a2d2(args.datapath)
        eval_a2d2(model, data, cfg["hp"])
    elif args.dataset == "kitti":
        data = load_kitti_dataset(args.datapath)
        eval_kitti(model, data, cfg["hp"])

if __name__ == '__main__':
    main()