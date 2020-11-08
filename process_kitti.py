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
import webcolors
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import time
from scipy import ndimage
import scipy.misc
from skimage.metrics import structural_similarity
import pickle
import threading
from skimage.measure import compare_ssim
import imagehash

import sys
sys.path.append("../SiamMask")

#from SiamMask import utils

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

# own imports 
from utility import *

thrs = np.arange(0.40, 0.45, 0.05)

parser = argparse.ArgumentParser(description='Test SiamMask')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('--config', dest='config', required=True, help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--dataset', dest='dataset', default='VOT2018',
                    help='datasets')

def get_instances(im):
    img = np.array(Image.open(im))
    obj_ids = np.unique(img)
    return [obj_id % 1000 for obj_id in obj_ids]


def load_kitti_dataset(path):
        images = path + "data_tracking_image_2/training/image_02/"
        annotations = path + "instances/"
        data = OrderedDict()
        for scene in listdir(images):
            data[scene] = {}
            data[scene]['annotations'] = sorted(glob.glob(annotations+scene+ '/'+ '*.png'))
            data[scene]['camera'] = sorted(glob.glob(images+scene+'/'+'*.png'))
            assert(len(data[scene]['annotations']) == len(data[scene]['camera']))
        return data

def track_kitti(model, data, hp, mask_enable=True, refine_enable=True, mot_enable=False, device='cpu'):
    gold_stop_track_dict = {}
    estimate_gold_stop_track_dict = {}
    pred_stop_track_dict = {}
    for scene in data:
        start = 0 
        end = len(data[scene]['camera'])-1
        gold_stop_track_dict[scene] = {}
        pred_stop_track_dict[scene] = {}
        estimate_gold_stop_track_dict[scene] = {}
        start_im = data[scene]['annotations'][0]
        img = np.array(Image.open(start_im))
        obj_ids = np.unique(img)
        for obj in obj_ids:
            if not obj//1000 == 0:
                pred_stop_track_dict[scene][obj%1000] = []
                gold_stop_track_dict[scene][obj%1000] = []
                estimate_gold_stop_track_dict[scene][obj%1000] = []
                print("class ", obj//1000)
                collect_states=[]
                current_instance = obj%1000
                for index, im in enumerate(data[scene]['annotations']):
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
                    if pred_stop_track_dict[scene][obj%1000] != [] and gold_stop_track_dict[scene][obj%1000] != []:
                        break
                pickle.dump(gold_stop_track_dict, open("gold_stop_track_dict2.pickle", "wb"))
                pickle.dump(pred_stop_track_dict, open("pred_stop_track_dict2.pickle", "wb"))
                pickle.dump(estimate_gold_stop_track_dict, open("estimate_gold_stop_track_dict2.pickle", "wb"))
                print(scene)
                print("Gold: obj ", obj//1000," instance ", obj%1000,  " stop track ", gold_stop_track_dict[scene][obj%1000])
                print("Estimate gold: obj ", obj//1000," instance ", obj%1000,  " stop track ", estimate_gold_stop_track_dict[scene][obj%1000])
                print("Prediction  obj ", obj//1000," instance ", obj%1000,  " stop track ", pred_stop_track_dict[scene][obj%1000])

            

    return gold_stop_track_dict, pred_stop_track_dict                  

def main():

    path = "../../Uni/9.Semester/AP/kitti/"
    images = path + "data_tracking_image_2/"
    instances = path + "instances/"
    data = load_kitti_dataset(path)

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
    gold_stop_track_dict, pred_stop_track_dict = track_kitti(model, data, cfg["hp"])
    pickle.dump(gold_stop_track_dict, open("gold_stop_track_dict2.pickle", "wb"))
    pickle.dump(pred_stop_track_dict, open("pred_stop_track_dict2.pickle", "wb"))

if __name__ == "__main__":
    main()
