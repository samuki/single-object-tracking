import cv2
import numpy as np 
from collections import OrderedDict
from os.path import join, realpath, dirname, exists, isdir
from os import listdir
import os
import glob

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
