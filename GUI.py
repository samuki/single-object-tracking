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

#pyQT imports 
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QMainWindow, QPushButton, QToolButton, QScrollArea
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject
import sys
import cv2

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
parser.add_argument('--object_lookup', dest='object_lookup', default='../../Uni/9.Semester/AP/class_list.json',
                    help='object_lookup')

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Tracker")
        # Variables for GUI creation 
        self.window_width = 1400
        self.window_height = 900

        self.display_width = self.window_width-300
        self.display_height = self.window_height-300

        self.image_label = QLabel(self)
        self.image_label.resize(self.display_width, self.display_height)

        self.setFixedSize(self.window_width, self.window_height)
        self.image_label.move(20,50)
        
        self.object_buttons = []
        self.thrs = np.arange(0.40, 0.45, 0.05)
        self.display_iou=QtWidgets.QLabel(self)
        self.display_similarity=QtWidgets.QLabel(self)
        self.display_stop_track=QtWidgets.QLabel(self)

        openFile = QtWidgets.QAction("&Open File", self)
        openFile.setShortcut("Ctrl+O")
        openFile.setStatusTip('Open File')
        openFile.triggered.connect(self.open_image)

        self.statusBar()
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(openFile)
        self.home()
        self.textLabel = QLabel('Evaluate A2D2')
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        self.setLayout(vbox)
        
        # variables needed for tracking
        args = parser.parse_args()
        with open(args.object_lookup) as json_file: 
            self.lookup = {ImageColor.getcolor(k, "RGB"):v for k,v in json.load(json_file).items()}
        self.cfg = load_config(args)
        init_log('global', logging.INFO)
        self.use_annotation = True
        logger = logging.getLogger('global')
        logger.info(args)

        # setup model
        if args.arch == 'Custom':
            from experiments.siammask_sharp.custom import Custom
            model = Custom(anchors=self.cfg['anchors'])
        else:
            parser.error('invalid architecture: {}'.format(args.arch))

        if args.resume:
            model = load_pretrain(model, args.resume)
        model.eval()
        device = torch.device('cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
        self.model = model.to(device)

        # setup dataset
        self.data = self.load_dataset(args.dataset)

    def home(self):
        btn = QtWidgets.QPushButton("Quit", self)
        btn.clicked.connect(self.close_application)
        btn.resize(btn.minimumSizeHint())
        btn.move(0,self.window_height-40)

        extractAction = QtWidgets.QAction(QtGui.QIcon('todachoppa.png'), 'Flee the Scene', self)
        extractAction.triggered.connect(self.close_application)
        self.show()

    def file_open(self):
        name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')
        file = open(name,'r')
        
    def close_application(self):
        sys.exit()

    def open_image(self):
        self.lock = threading.Lock()

        name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')
        self.path = name[0]
        self.file_name = self.path.split("/")[-1]
        # clear previous displayed scenes
        for button in self.object_buttons:
            button.deleteLater()
        self.object_buttons = []
        self.collect_masks = []
        self.collect_states = []
        self.collect_averages = []
        self.display_iou.clear()
        self.display_similarity.clear()
        self.display_stop_track.clear()

        # init image and bounding boxes
        self.split_path = self.path.split("/")
        if self.use_annotation: 
            self.anno_path = "/".join(self.split_path[:-3]+["label"]+[self.split_path[-2]]+[self.split_path[-1].replace("camera", "label")])
            bb_image, rgb_codes, mask_coordinates = self.display_object_bb()
        qt_img = self.convert_cv_qt(bb_image)
        # display it
        self.image_label.setPixmap(qt_img)
        target_obj = None
        # create buttons for objects
        for index, rgb in enumerate(rgb_codes):
            rgb_tuple = tuple(rgb)      
            label = self.lookup[rgb_tuple]
            object_btn = QtWidgets.QPushButton('{}'.format(label), self)  
            text = object_btn.text()
            width = 300
            height = 30
            object_btn.setGeometry(self.window_width - width - 20 , 50+index*height, width, height)
            object_btn.clicked.connect(lambda ch, index=index: self.init_track(mask_coordinates[index], rgb_codes[index]))
            self.object_buttons.append(object_btn)
        for button in self.object_buttons:
            button.show()

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def background_track_calculation(self, state, rgb_code, mask_enable=True, refine_enable=True, device='cpu'):
        # TODO catch error of calucaltion interrupted
        stop_track_flag = False
        end_track_flag = False
        background_iterator = 0
        while end_track_flag == False and np.array_equal(self.current_track, rgb_code):
            if background_iterator >= 1:
                #self.lock.acquire()
                im, state, qt_img, iou, stop_track_flag, score = self.single_step_object_track(state, rgb_code, background_iterator)
                #self.lock.release()
                # collect 
                self.precalc_track[background_iterator] = {}
                self.precalc_track[background_iterator]["im"] = im
                self.precalc_track[background_iterator]["state"] = state
                self.precalc_track[background_iterator]["qt_img"] = qt_img
                self.precalc_track[background_iterator]["iou"] = iou
                self.precalc_track[background_iterator]["stop_track"] = stop_track_flag
                self.precalc_track[background_iterator]["similarity"] = score
            background_iterator += 1
            if background_iterator+1 == len(self.camera):
                end_track_flag = True

    def single_step_object_track(self, state, rgb_code, index, mask_enable=True, refine_enable=True, device='cpu'):
        im = cv2.imread(self.camera[index])
        # prevent conflict in track
        state = siamese_track(state, im, mask_enable, refine_enable, device=device)
        location = state['ploygon'].flatten()
        new_mask = state['mask'] > state['p'].seg_thr
        #self.collect_averages.append(average_object_confidence)
        im[:, :, 2] = (new_mask > 0) * 255 + (new_mask == 0) * im[:, :, 2]
        cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        qt_img = self.convert_cv_qt(im)
        iou = self.single_step_eval(state["mask"], self.annotations[index], rgb_code)
        prev_mask = self.collect_masks[-1]
        prev_anno = np.array(Image.open(self.annotations[index -1]))
        current_anno = np.array(Image.open(self.annotations[index]))
        current_all_instances_mask = np.logical_and.reduce(current_anno == rgb_code, axis = -1).astype(np.uint8)
        previous_all_instances_mask = np.logical_and.reduce(prev_anno == rgb_code, axis = -1).astype(np.uint8)
        predicted_mask = state["mask"]
        predicted_mask[predicted_mask>thrs] = 1
        predicted_mask[predicted_mask<=thrs] = 0 
        mask_sum = predicted_mask + current_all_instances_mask
        intersec = np.sum(mask_sum[mask_sum==2])
        score = 1
        stop_track_flag = False
        # calculate stop track: currently broken 
        """
        if index >1:
            # compare images
            current_location = state['minAreaRect']
            prev_state = self.collect_states[-1]
            prev_location = prev_state['minAreaRect']
            if len(current_location) == len(prev_location) == 3: 
                current_im = im 
                current_cropped_image = crop_minAreaRect(current_im, current_location)
                prev_im = self.precalc_track[index-1]["im"]
                #prev_cropped_image = self.crop_minAreaRect(cv2.cvtColor(prev_im, cv2.COLOR_BGR2GRAY), prev_location)
                prev_cropped_image = crop_minAreaRect(prev_im, prev_location)
                common_size = (150, 150)
                try:
                    prev_cropped_image= cv2.resize(prev_cropped_image,common_size, interpolation=cv2.INTER_CUBIC)
                    current_cropped_image = cv2.resize(current_cropped_image,common_size, interpolation=cv2.INTER_CUBIC)

                    score, diff = structural_similarity(prev_cropped_image, current_cropped_image, full=True, multichannel=True)
                except Exception as e:
                    score=1.0
            stop_track_flag = False
            if intersec == 0:
                predicted_classes = np.unique(current_anno[predicted_mask == 1], axis=0)
                stop_track_flag = True
        """
        self.precalc_track[index] = {}
        self.precalc_track[index]["im"] = im
        self.precalc_track[index]["state"] = state
        self.precalc_track[index]["qt_img"] = qt_img
        self.precalc_track[index]["iou"] = iou
        self.precalc_track[index]["stop_track"] = stop_track_flag
        self.precalc_track[index]["similarity"] = score
        return im, state, qt_img, iou, stop_track_flag, score

    def track_object(self, state, rgb_code, mask_enable=True, refine_enable=True, device='cpu'):
        self.next_btn.clicked.disconnect()
        self.pic_index += 1
        if self.pic_index <= self.end-1:  # tracking
            if self.pic_index in self.precalc_track:
                im = self.precalc_track[self.pic_index]["im"]
                state = self.precalc_track[self.pic_index]["state"]
                qt_img = self.precalc_track[self.pic_index]["qt_img"]
                iou = self.precalc_track[self.pic_index]["iou"]
                stop_track_flag = self.precalc_track[self.pic_index]["stop_track"]
                score = self.precalc_track[self.pic_index]["similarity"]
            else:
                #self.lock.acquire()
                im, state, qt_img, iou, stop_track_flag, score = self.single_step_object_track(state, rgb_code, self.pic_index) 
                #self.lock.release()
            self.collect_states.append(state)
            if stop_track_flag:
                self.display_stop_track.setText("Stop track!")
                self.display_stop_track.setGeometry(30, self.display_height+150, 250, 50)
                self.display_stop_track.show()
            # adjust threshold
            #if score < 0.35 and self.pic_index > 1:
                #print("STOP TRACK LOW SIM")
            #current_rgbs = np.unique(current_anno, axis=0)
            self.display_iou.clear()
            self.display_iou.setText("IoU: " +str(round(iou, 3)))
            self.display_iou.setGeometry(30, self.display_height+100, 250, 50)
            self.display_iou.show()

            self.display_similarity.clear()
            self.display_similarity.setText("Similarity: " +str(round(score, 3)))
            self.display_similarity.setGeometry(30, self.display_height+50, 250, 50)
            self.display_similarity.show()

            self.image_label.setPixmap(qt_img)
        else:
            self.display_similarity.clear()
            self.display_iou.clear()
            self.display_iou.setText("End of scene")
        self.next_btn.clicked.connect(lambda: self.track_object(state, rgb_code))


    def init_track(self, mask_coordinates, rgb_code, mask_enable=True, refine_enable=True, device='cpu'):
        self.precalc_track = {}
        # clear display
        self.display_similarity.clear()
        self.display_iou.clear()
        self.display_stop_track.clear()
        self.current_track = rgb_code
        # self.collect_averages = []
        scene = self.split_path[-4]
        self.start = self.data[scene]['camera'].index(self.path)
        self.camera = self.data[scene]['camera'][self.start:]
        end = len(self.camera)
        self.end = min([end, len(self.camera)])
        self.annotations = self.data[scene]['annotations'][self.start:]
        mask = mask_coordinates
        self.pic_index = 0
        x, y, w, h = cv2.boundingRect((mask).astype(np.uint8))
        cx, cy = x + w/2, y + h/2
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])
        im = cv2.imread(self.path)
        cv2.rectangle(im,(x,y),(x+w,y+h), (0,255,0), 3)
        state = siamese_init(im, target_pos, target_sz, self.model, self.cfg["hp"], device=device)
        # display it
        self.collect_masks.append(mask)
        self.collect_states.append(state)
        qt_img = self.convert_cv_qt(im)
        self.image_label.setPixmap(qt_img)
        self.next_btn = QtWidgets.QPushButton('Next', self)  
        #btn.move(self.display_width, index)
        next_width = 250
        next_height = 50
        self.next_btn.setGeometry(self.display_width-next_width-120, self.display_height+100, next_width, next_height)

        pre_calc_thread = threading.Thread(target=self.background_track_calculation, name="pre_calc", args=[state, rgb_code])
        self.next_btn.clicked.connect(lambda: pre_calc_thread.start())
        self.next_btn.clicked.connect(lambda: self.track_object(state, rgb_code))
        self.next_btn.show()

    def single_step_eval(self, output, target, rgb):
        anno = np.array(Image.open(target))
        ious = [0]
        for k, thr in enumerate(self.thrs):
            output_thr = output > thr
            target_j = np.logical_and.reduce(anno == rgb, axis = -1)
            label_im, nb_labels = ndimage.label(target_j)
            pred = output_thr == 1
            for counter in range(nb_labels):
                compare_mask = np.full(np.shape(label_im), counter+1)
                target_mask = np.equal(label_im, compare_mask).astype(int)
                iou = 0
                mask_sum = (pred == 1).astype(np.uint8) + (target_mask > 0).astype(np.uint8)
                intxn = np.sum(mask_sum == 2)
                union = np.sum(mask_sum > 0)
                if union > 0:
                    iou = intxn / union
                elif union == 0 and intxn == 0:
                    iou = 1
                ious.append(iou)
        return max(ious)

    def display_object_bb(self,mask_enable=True, refine_enable=True, mot_enable=False, device='cpu'):
        pickle_dir = 'init_bb/' 
        os.makedirs(os.path.dirname(pickle_dir), exist_ok=True) 
        if os.path.exists(pickle_dir+self.file_name+".pickle"):
            loaded_data = pickle.load(open(pickle_dir+self.file_name+".pickle", "rb"))
            im = loaded_data["image"] 
            duplicates = loaded_data["duplicates"]
            mask_coordinates = loaded_data["masks"]
        else:
            annotations = [self.anno_path]
            color_array = [np.array(Image.open(x)) for x in annotations]
            reshaped_array = color_array[0].reshape((color_array[0].shape[0]* color_array[0].shape[1], 3))
            rgb = np.unique(reshaped_array, axis=0)
            color_track = [annotation.astype(np.uint8) for annotation in color_array]
            mask_coordinates = []
            im = cv2.imread(self.path)
            rgb_duplicates = []
            for object_index, code in enumerate(rgb): 
                mask = np.logical_and.reduce(color_track[0] == code, axis = -1)
                label_im = mask
                nb_labels = 1
                rgb_tuple = (int(code[0]), int(code[1]), int(code[2]))
                object_name = self.lookup[rgb_tuple]
                # TODO find alternative to heuristic
                relvevant_objects = ["Car", "vehicle", "Truck", "Pedestrian", "Bicycle"]
                for obj in relvevant_objects:
                    if obj in object_name:
                        label_im, nb_labels = ndimage.label(mask)
                for counter in range(nb_labels):
                    compare_mask = np.full(np.shape(label_im), counter+1)
                    mask = np.equal(label_im, compare_mask).astype(int)
                    x, y, w, h = cv2.boundingRect((mask).astype(np.uint8))
                    mask_coordinates.append(mask)
                    cv2.rectangle(im,(x,y),(x+w,y+h), rgb_tuple, 3)
                    rgb_duplicates.append(code)
            duplicates = np.array(rgb_duplicates)
            # save calculation
            save_data = {}
            save_data["image"] = im
            save_data["duplicates"] = rgb_duplicates
            save_data["masks"] = mask_coordinates
            pickle.dump(save_data, open(pickle_dir+self.file_name+".pickle", "wb"))
        return im, duplicates, mask_coordinates
        
    # TODO: find right size for test data
    def load_dataset(self, path):
        data = OrderedDict()
        for scene in listdir(path):
            # TODO: generalize to all subfolders
            data[scene] = {}
            #print("Path: ", join(path,scene, 'label/cam_front_center', '*.png'))
            data[scene]['annotations'] = sorted(glob.glob(join(os.path.abspath(path),scene, 'label/cam_front_center', '*.png')))
            data[scene]['camera'] = sorted(glob.glob(join(os.path.abspath(path),scene,  'camera/cam_front_center', '*.png')))
            # assert images and annotations have same length
            assert(len(data[scene]['annotations']) == len(data[scene]['camera']))
        return data

def main():
    #path_to_data = "../../Uni/9.Semester/AP/dataset"
    #model_path = "experiments/siammask_sharp/SiamMask_DAVIS.pth"
    #config_path = "experiments/siammask_sharp/config_davis.json"
    global args, logger, v_id
    app = QApplication([])
    a = App()
    a.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()