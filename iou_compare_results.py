import pickle
import json
from PIL import Image, ImageColor
import numpy as np


def compute_precision_recall(st1, st2):
    false_positives = 0 
    false_negatives = 0
    true_positives = 0 
    #true_negatives = 0
    #print(st2)
    for scene in st1: 
        for entry_point in st1[scene]:
            for obj in st1[scene][entry_point]:
                #if not st1[scene][entry_point][obj] == [] and not obj == 1 and not obj == 0:
                #print(st2[scene][entry_point][obj])
                if  not obj == 1 and not obj == 0 and not st1[scene][entry_point][obj] == []:
                    if entry_point in st2[scene]:
                        if st2[scene][entry_point][obj] ==[]:
                            if st1[scene][entry_point][obj] !=[]:
                                false_negatives+=1
                        elif st1[scene][entry_point][obj] == []:
                            if st2[scene][entry_point][obj] != []: 
                                false_positives+=1
                        elif st2[scene][entry_point][obj][0] < st1[scene][entry_point][obj][0]:
                            false_positives+=1
                        elif st2[scene][entry_point][obj][0] == st1[scene][entry_point][obj][0]:
                            true_positives+=1
    if true_positives == 0:
        precision = 0 
        recall = 0
    else:
        precision = true_positives/(true_positives+false_positives)
        recall = true_positives/(true_positives+false_negatives)
    print(false_positives, true_positives)
    return precision, recall

def convert_kitti_dict(st):
    for scene in st: 
        for obj in st[scene]:
            if isinstance(st[scene][obj], list):
                if st[scene][obj] == []:
                    st[scene][obj] = 9000
                else:
                    st[scene][obj] = st[scene][obj][0]
    return st


def compute_object_wise_precision_recall(st1, st2):
    instance_dict = {
        "cars": 1,
        "pedestrians": 2,
        "trucks": 3,
        "smallVehicle": 4,
        "utilityVehicle": 5,
        "bicycle": 6,
        "tractor": 7
    }
    inverted_instance_dict = {v:k for k,v in instance_dict.items()}
    object_dict = {}
    for scene in st1: 
        for entry_point in st1[scene]: 
            for obj in st1[scene][entry_point]:
                if not st1[scene][entry_point][obj] == [] and not obj == 1 and not obj == 0:
                    cl = inverted_instance_dict[int(str(obj)[0])]
                    if not cl in object_dict:
                        object_dict[cl] = {}
                        object_dict[cl]["fp"] = 0
                        object_dict[cl]["fn"] = 0
                        object_dict[cl]["tp"] = 0
                    
                    #if st2[scene][entry_point][obj] > st1[scene][entry_point][obj]:
                    if st2[scene][entry_point][obj] ==[]:
                        object_dict[cl]["fn"]+=1
                    elif st2[scene][entry_point][obj] < st1[scene][entry_point][obj]:
                        object_dict[cl]["fp"]+=1
                    else: 
                        object_dict[cl]["tp"]+=1
    return object_dict

def avg_track_length(st1, st2):
    track_length_st1= []
    track_length_st2= []
    for scene in st1: 
        for obj in st1[scene]:
            track_length_st1.append(st1[scene][obj])
            track_length_st2.append(st2[scene][obj])
    return sum(track_length_st1)/len(track_length_st1), sum(track_length_st2)/len(track_length_st2)

def compare_iou(iou_dict1, iou_dict2, lookup):
        for obj in iou_dict1:
            if iou_dict1[obj][0] != 0 and iou_dict2[obj][0] != 0:
                print(obj, ": GOLD: ", iou_dict1[obj][1]/iou_dict1[obj][0], " PRED: ", iou_dict2[obj][1]/iou_dict2[obj][0])
            else: 
                print(obj, " seen ", 0, " times with mean IoU: ", 0)

def convert_iou_object_dict(od):
    object_dict = {}
    for scene in od: 
        for obj in od[scene]:
            if obj in object_dict:
                object_dict[obj][0]+=1
                object_dict[obj][1]+=od[scene][obj]
            else: 
                object_dict[obj] = [0,0]
    return object_dict

def main():
    #with open('../../Uni/9.Semester/Object_tracking/class_list.json') as json_file: 
    #    lookup = json.load(json_file) 
    #lookup = {ImageColor.getcolor(k, "RGB"):v for k,v in lookup.items()}
    #path = "/media/samuki/Elements/camera_lidar_semantic"
    #path = "pickle_files/"
    # current tracks: 
    # dir1 = gold without exception for different shadings
    # dir3 = gold with exception for different shadings
    # dir2 = stop calculation through ssim 
    #dir1 = path+"_results/"
    #dir2 = path+"_results2/"
    #dir3 = path+"_gold_results2/"
    #st1 = pickle.load(open(dir1+"stop_track_dict.pickle", "rb"))
    #st2 = pickle.load(open(dir2+"stop_track_dict.pickle", "rb"))
    #st3 = pickle.load(open(dir3+"stop_track_dict.pickle", "rb"))
    #precision, recall = compute_precision_recall(st1, st2)
    #print("Precision: ", precision)
    #print("Recall: ", recall)
    """
    object_dict = compute_object_wise_precision_recall(st1, st2)
    for obj in object_dict:
        if object_dict[obj]["tp"] != 0:
            obj_precision = object_dict[obj]["tp"]/(object_dict[obj]["tp"]+object_dict[obj]["fp"])
            obj_recall = object_dict[obj]["tp"]/(object_dict[obj]["tp"]+object_dict[obj]["fn"])
            print(lookup[obj], " precision: ", obj_precision)
            print(lookup[obj], " recall: ", obj_recall)
        else:
            print(lookup[obj], " no true positives")
    """
    #print("len st1: ", sum(track_length_st1)/len(track_length_st1))
    #print("len st2: ", sum(track_length_st2)/len(track_length_st2))

    #tl1, tl3 = avg_track_length(st1, st3)
    #print("len st1: ", tl1)
    #print("len st3: ", tl3)

    # dict2 = ssim but wrong tracking window
    # 750 autoenc is autoenc with 8 classes + 0.5 thr
    # average autoenc is no bl 
    #mode = "_average_autoenc"
    #mode = "othertest"
    #mode = "autoencoder0.7196"
    #mode = "score09"
    mode = "ssim"
    #mode = "pretrained_autoenc0.9"
    #mode = "ssim0.5"
    #mode = "score0.8"
    #mode = 'confidence_score0.95'
    #mode = "baseline1"
    #mode = "pretrained_autoenc0.9"
    thrs= [0.05,0.1,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.98]
    #thrs= [0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.98]
    #thrs = [0.85,0.9,0.95,0.98]
    #thrs = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
    thrs = [0.25]
    #mode = 'pretrained_autoencoder'
    #mode = 'pretrained_autoencoder'
    #mode = 'autoencoder'
    mode = 'confidence_score'
    #mode = 'constant'
    dataset = 'a2d2'
    if dataset == 'a2d2':
        instance_dict = {
        "cars": 1,
        "pedestrians": 2,
        "trucks": 3,
        "smallVehicle": 4,
        "utilityVehicle": 5,
        "bicycle": 6,
        "tractor": 7
        }
    else: 
        instance_dict = {'cars': 1,
                'pedestrian': 2}
    inverted_instance_dict = {val:key for key, val in instance_dict.items()}
    object_wise_iou = {obj: [] for obj in instance_dict}
    total_iou = []
    number_of_instances = {obj: [] for obj in instance_dict}
    number_of_instances['total'] = []
    ##iou_dict1 =  pickle.load(open(dir1+"iou_dict.pickile", "rb"))
    #iou_dict2 =  pickle.load(open(dir2+"iou_dict.pickle", "rb"))
    #iou_dict3 = pickle.load(open(dir3+"iou_dict.pickle", "rb"))
    #compare_iou(iou_dict1, iou_dict2, lookup)
    experiment_string6 = dataset+'42ssim0.8_gold_iou_dict.pickle'
    experiment_string7 = dataset+'8ssim0.8_gold_iou_dict.pickle'
    experiment_string8 = dataset+'69ssim0.8_gold_iou_dict.pickle'
    #experiment_string5 = dataset+'42ssim0.8_gold_iou_dict.pickle'
    #experiment_string6 = dataset+'8confidence_score0.75_gold_iou_dict.pickle'
    #experiment_string7 = dataset+'42confidence_score0.75_gold_iou_dict.pickle'
    #experiment_string8 = dataset+'69confidence_score0.75_gold_iou_dict.pickle'
    experiment_strings = [experiment_string6, experiment_string7, experiment_string8]
    for experiment_string in experiment_strings: 
        iou_dict = pickle.load(open(dataset+"_pickle_files/"+experiment_string, 'rb'))
        #gold_iou_dict = pickle.load(open(dataset+"_pickle_files/"+experiment_string6, 'rb'))
        #print('normal  ', iou_dict)
        #print('gold ', gold_iou_dict)
        for scene in iou_dict:
            for entry_point in iou_dict[scene]:
                for obj in iou_dict[scene][entry_point]:
                    if obj != 0:
                        #print(iou_dict[scene][entry_point][obj])
                        total_iou.append(iou_dict[scene][entry_point][obj][0][0])
                        obj_string = inverted_instance_dict[int(str(obj)[0])]
                        object_wise_iou[obj_string].append(iou_dict[scene][entry_point][obj][0][0])
        
        for obj in object_wise_iou:
            number_of_instances[obj].append(len(object_wise_iou[obj]))
        number_of_instances['total'].append(len(total_iou))
    
    
    for obj in object_wise_iou:
        print(obj, ':')
        if object_wise_iou[obj] != []:
            print(np.mean(np.array(number_of_instances[obj])), np.std(np.array(number_of_instances[obj])),np.mean(np.array(object_wise_iou[obj])), np.std(np.array(object_wise_iou[obj])))
        else: 
            print(0,0,0)
    print('Total:')
    print(np.mean(np.array(number_of_instances['total'])), np.std(np.array(number_of_instances['total'])), np.mean(np.array(total_iou)), np.std(np.array(total_iou)))
    


if __name__ == "__main__":
    main()
