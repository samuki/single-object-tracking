import pickle
import json
from PIL import Image, ImageColor
import numpy as np

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

def main():
    mode = "ssim"
    #thrs= [0.05,0.1,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.98]
    thrs = [0.25]
    mode = 'confidence_score'
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
        for scene in iou_dict:
            for entry_point in iou_dict[scene]:
                for obj in iou_dict[scene][entry_point]:
                    if obj != 0:
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
