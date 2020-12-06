import cv2
import numpy as np 
from collections import OrderedDict
from os.path import join, realpath, dirname, exists, isdir
from os import listdir
import yaml
import os
import glob
from PIL import Image

def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))
    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0) 
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0
    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
    pts[1][0]:pts[2][0]]
    return img_crop
    
def tensor_to_float(tensor, round_to_n=3):
    try:
        tensor.item()
        if tensor.nelement() > 0:
            return round(tensor.item(), round_to_n)
        else:
            return 0
    except ValueError as e:
        return 0 

def crop_rotated_rect(img, coord):
    rect = cv2.minAreaRect(coord)
    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    #cv2.imshow('image',warped)

    return warped

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

def shuffle_data(data, max_images, seed=42): 
    np.random.seed(seed)
    for scene in data: 
        width, height = Image.open(data[scene]['camera'][0]).size
        print(width, height)
        random_set = np.random.randint(min(len(data), max_images))
        random_scene = list(data.keys())[random_set]
        random_inserted_index =  np.random.randint(min(len(data[random_scene]['camera']), max_images))
        random_insert_index = np.random.randint(min(len(data[scene]['camera']), max_images))
        data[scene]['camera'].insert(random_insert_index, data[random_scene]['camera'][random_inserted_index]) 
        data[scene]['annotations'].insert(random_insert_index, data[random_scene]['annotations'][random_inserted_index]) 
        print(random_insert_index)
    return data

# TODO: find right size for test data
def load_a2d2(path):
    data = OrderedDict()
    for scene in listdir(path):
        if os.path.isdir(path):
            # TODO: generalize to all subfolders
            data[scene] = {}
            data[scene]['annotations'] = sorted(glob.glob(join(path,scene, 'label/cam_front_center', '*.png')))
            data[scene]['camera'] = sorted(glob.glob(join(path,scene,  'camera/cam_front_center', '*.png')))
            # assert images and annotations have same length
            assert(len(data[scene]['annotations']) == len(data[scene]['camera']))
    return data

def load_eval_config(args):
    with open(args.eval_config) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
        eval_config = yaml.load(file, Loader=yaml.FullLoader)
    # load siammask args
    args.resume = eval_config['checkpoint']
    args.arch = eval_config['arch']
    args.config = eval_config['siammask_config']

    # load custom args
    args.dataset = eval_config['dataset']
    args.datapath = eval_config['datapath']
    args.similarity = eval_config['similarity']
    args.thresholds = eval_config['thresholds']
    args.autoencoder_classes = eval_config['autoencoder_classes']
    args.seed = eval_config['seed']
    args.random_entries = eval_config['random_entries']
    args.frames_per_entry = eval_config['frames_per_entry']
    return args