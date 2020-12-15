import os
import glob
import json
import pickle
import threading
from multiprocessing import Process, Lock
import logging
import argparse
from collections import OrderedDict
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from PIL import Image, ImageColor
import matplotlib.pylab as plt
import matplotlib.image as mpimg
from scipy import ndimage
import scipy.misc

from skimage.metrics import structural_similarity

# own imports 
from utility import *
from autoencoder import ImagenetTransferAutoencoder
from pretrained_autoencoder import init_autoencoder, pretrained_autoencoder_similarity

# SiamMask imports
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

thrs = np.arange(0.40, 0.45, 0.05)

parser = argparse.ArgumentParser(description='Evaluate Dataset')
parser.add_argument('--arch', dest='arch', default='', choices=['Custom',],
                    help='architecture of pretrained model')
parser.add_argument('--config', dest='config', help='hyper-parameter for SiamMask')
parser.add_argument('--resume', default='', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--dataset', dest='dataset', default='VOT2018',
                    help='datasets')
parser.add_argument('--datapath', dest='datapath', default='',
help='datapath')
parser.add_argument('--eval_config', dest='eval_config', required=True, help='hyper-parameter for evaluation')
parser.add_argument('--similarity', dest='similarity', default='',
                    help='similarity')
parser.add_argument('--thresholds', dest='thresholds', default=[],
                    help='thresholds')
parser.add_argument('--autoencoder_classes', dest='autoencoder_classes', default=8,
                    help='autoencoder_classes')
parser.add_argument('--seed', dest='seed', default=42,
                    help='seed')
parser.add_argument('--random_entries', dest='random_entries', default=5,
                    help='random_entries')

parser.add_argument('--frames_per_entry', dest='frames_per_entry', default=50,
                    help='frames_per_entry')


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

def autoencoder_similarity(autoencoder, current_cropped, prev_cropped_tracks, prev_cropped, average=True):
    from torchvision import transforms
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    def encoding_pipeline(cropped_im, preprocess):
        pil_image = Image.fromarray(cropped_im)
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0) 
        feature_vec=autoencoder(input_batch)
        return feature_vec
                 
    prev_feature_vec = encoding_pipeline(prev_cropped, preprocess)
    if average: 
        prev_cropped_tracks.append(prev_feature_vec)
        prev_feature_vec = torch.mean(torch.stack(prev_cropped_tracks), axis=1)

    current_feature_vec = encoding_pipeline(current_cropped, preprocess)    
    cossim = torch.nn.CosineSimilarity()
    sim = cossim(current_feature_vec, prev_feature_vec)
    #print("sim ", sim)
    return tensor_to_float(sim), prev_cropped_tracks

def ssim(prev_cropped_image, current_cropped_image):
    common_size = (150, 150)
    try:
        prev_cropped_image= cv2.resize(prev_cropped_image,common_size, interpolation=cv2.INTER_CUBIC)
        current_cropped_image = cv2.resize(current_cropped_image,common_size, interpolation=cv2.INTER_CUBIC)

        score, diff = structural_similarity(prev_cropped_image, current_cropped_image, full=True, multichannel=True)
    except Exception as e:
        print(e)
        score=1.0
    return score

def track_object(lock, autoencoder, entry_point, thr,model, hp, scene,obj, data, images_to_consider, output_dir, 
                pred_stop_track_dict, gold_stop_track_dict, estimate_gold_stop_track_dict, 
                mask_enable=True, refine_enable=True, device='cpu'):
    prev_feature = None
    current_annos = data[scene]['annotations'][entry_point:entry_point+images_to_consider]
    current_images = data[scene]['camera'][entry_point:entry_point+images_to_consider]
    start = 0 
    end = len(current_images)-1
    goldstop=False
    for index, im in enumerate(current_annos):
        cam_im = current_images[index]
        cv_im = cv2.imread(cam_im)
        anno_im = Image.open(im)
        anno_array = np.array(anno_im)
        if index == start:  # init
            mask = anno_array == obj
            x, y, w, h = cv2.boundingRect((mask).astype(np.uint8))
            cx, cy = x + w/2, y + h/2
            if cx == 0 and cy == 0:
                break
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            state = siamese_init(cv_im, target_pos, target_sz, model, hp, device=device)  # init tracker
        elif end >= index > start:  # tracking
            state = siamese_track(state, cv_im, mask_enable, refine_enable, device=device)  # track
            current_objects = np.unique(anno_array)
            #current_in = [current_object% 1000 for current_object in current_objects]
            if obj not in current_objects:
                #print("goldstop")
                gold_stop_track_dict[scene][entry_point][obj].append(index)
                #print("Gold scene ", scene, " obj ", obj//1000," instance ", obj%1000,  " stop track ", index)
                goldstop = True
            current_instance_mask = anno_array
            current_instance_mask[current_instance_mask == obj] = 1
            #print(current_instance_mask)
            mask = state['mask']
            check_mask = state['mask']
            check_mask[check_mask>thrs] = 1
            check_mask[check_mask<=thrs] = 0 
            mask_sum = check_mask + current_instance_mask
            intersec = np.sum(mask_sum[mask_sum==2])
            trackstop = False
            if intersec == 0:
                estimate_gold_stop_track_dict[scene][entry_point][obj].append(index)
                trackstop = True
            if index > start+1:
                current_poly = state["ploygon"]
                current_im = cv_im 
                prev_im = cv2.imread(current_images[index-1])
                if args.similarity in ['autoencoder', 'pretrained_autoencoder', 'ssim']: 
                    try:  
                        current_cropped = crop_rotated_rect(current_im, current_poly)
                        prev_cropped = crop_rotated_rect(prev_im, prev_poly)
                        if args.similarity == 'autoencoder':
                            score, prev_feature = autoencoder_similarity(autoencoder, 
                            current_cropped, prev_feature, prev_cropped, average=False)
                        elif args.similarity == 'pretrained_autoencoder':
                            score, prev_feature = pretrained_autoencoder_similarity(autoencoder,
                                current_cropped, prev_feature, prev_cropped)
                            #print('score ', score)
                        elif args.similarity == 'ssim':
                            score = ssim(prev_cropped, current_cropped)
                        else: 
                            print("Config error no similarity measure")
                    except Exception as e:
                        score=1.0 
                        print(e)
                if args.similarity == 'confidence_score':
                    score = state['score']
                elif args.similarity == 'constant':
                    score = 1
                if score < thr:
                    pred_stop_track_dict[scene][entry_point][obj].append(index)
            prev_poly = state["ploygon"]
        if goldstop:
            break
    lock.acquire()
    pickle.dump(gold_stop_track_dict, open(args.dataset+"_pickle_files/gold_"+output_dir+".pickle", "wb"))
    pickle.dump(pred_stop_track_dict, open(args.dataset+"_pickle_files/pred_"+output_dir+".pickle", "wb"))
    pickle.dump(estimate_gold_stop_track_dict, open(args.dataset+"_pickle_files/estimate_gold_"+output_dir+".pickle", "wb"))
    lock.release()

def eval_end_of_track(model, thr, data,hp, mask_enable=True, refine_enable=True, mot_enable=False, device='cpu'):
    gold_stop_track_dict = {}
    estimate_gold_stop_track_dict = {}
    pred_stop_track_dict = {}
    np.random.seed(args.seed)
    num_random_entries = args.random_entries
    images_to_consider = args.frames_per_entry
    output_dir = args.dataset+args.similarity+str(thr)
    print("output_dir ", output_dir)
    if args.similarity == 'autoencoder':
        autoencoder = ImagenetTransferAutoencoder(args.autoencoder_classes)
    elif args.similarity == 'pretrained_autoencoder':
        autoencoder = init_autoencoder()
    else: 
        autoencoder = ''
    #data = shuffle_data(data, images_to_consider)
    for scene in data:
        print("Scene ", scene)
        #print(len(data[scene]['camera']))
        entry_points = np.random.randint(low=0, \
            high=len(data[scene]['camera'])-5,size=num_random_entries)
        gold_stop_track_dict[scene] = {}
        pred_stop_track_dict[scene] = {}
        estimate_gold_stop_track_dict[scene] = {}
        print("entry points ", entry_points)
        for entry_point in entry_points:
            gold_stop_track_dict[scene][entry_point] = {}
            pred_stop_track_dict[scene][entry_point] = {}
            estimate_gold_stop_track_dict[scene][entry_point] = {}
            start_im = data[scene]['annotations'][entry_point]
            img = np.array(Image.open(start_im))
            obj_ids = np.unique(img)
            # TODO random entries here 
            images_to_consider = min([images_to_consider, len(data[scene]['annotations'][entry_point:])-1])
            lock = Lock()
            threads = []
            for obj in obj_ids:
                if not obj//1000 in [0,10]:
                    pred_stop_track_dict[scene][entry_point][obj] = []
                    gold_stop_track_dict[scene][entry_point][obj] = []
                    estimate_gold_stop_track_dict[scene][entry_point][obj] = []
                    t = threading.Thread(target=track_object, args=(lock,autoencoder, entry_point, thr, model, hp, scene, obj, data, images_to_consider, output_dir, \
                    pred_stop_track_dict, gold_stop_track_dict, estimate_gold_stop_track_dict))
                    threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        pred_stop_track_dict = pickle.load(open(args.dataset+"_pickle_files/pred_"+output_dir+".pickle", "rb"))
        gold_stop_track_dict = pickle.load(open(args.dataset+"_pickle_files/gold_"+output_dir+".pickle", "rb"))
        estimate_gold_stop_track_dict = pickle.load(open(args.dataset+"_pickle_files/estimate_gold_"+output_dir+".pickle", "rb"))
        for entry_point in entry_points:
            #print(gold_stop_track_dict)
            if entry_point in gold_stop_track_dict[scene]: 
                for obj in gold_stop_track_dict[scene][entry_point]:
                    if not obj//1000 in [0,10]:
                        print(scene)
                        print("Gold: obj ", obj,  " stop track ", gold_stop_track_dict[scene][entry_point][obj])
                        print("Estimate gold: obj ", obj,  " stop track ", estimate_gold_stop_track_dict[scene][entry_point][obj])
                        print("Prediction  obj ", obj,  " stop track ", pred_stop_track_dict[scene][entry_point][obj])
    return gold_stop_track_dict, pred_stop_track_dict

def eval_a2d2(model, data, hp, mask_enable=True, refine_enable=True, mot_enable=False, device='cpu'): 

    with open('../../Uni/9.Semester/Object_tracking/class_list.json') as json_file: 
        lookup = json.load(json_file) 
    lookup = {ImageColor.getcolor(k, "RGB"):v for k,v in lookup.items()}
    iou_dict = {}
    #print(lookup)
    stop_track_dict = {}
    mask_dict = {}
    #iou_dict = {lookup[key]:[0,0] for key in lookup.keys()}
    #print("init iou-dict ", iou_dict)
    
    # change parameters here
    use_gold_crit = True
    save_output_dir = args.dataset+"_gold_results2"
    os.makedirs(save_output_dir, exist_ok=True) 
    for scene in data: 
        iou_dict[scene] = {}
        stop_track_dict[scene] = {}
        mask_dict[scene] = {}
        print("It's time for scene ", scene)
        start = 0 
        #end = len(scene)
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
                            # TODO remove min area rect
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
    global args, logger
    args = parser.parse_args()
    args = load_eval_config(args)

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

    if args.dataset == 'kitti':
        data = load_kitti_dataset(args.datapath)
    elif args.dataset == 'a2d2':
        data = load_a2d2(args.datapath)
    print("hi1")
    print("mode ", args.mode)
    # setup dataset
    if args.mode == "IoU":
        eval_a2d2(model, data, cfg["hp"])
    elif args.mode == "end_of_track":
        print('hi2')
        threads = []
        for thr in args.thresholds:
            print('hi3')
            t = threading.Thread(target=eval_end_of_track, args=(model, thr, data, cfg["hp"]))
            threads.append(t)
            #eval_kitti(model, thr, data, cfg["hp"])
        for t in threads:
            t.start()
        for t in threads:
            t.join()

if __name__ == '__main__':
    main()
