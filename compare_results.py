import pickle
import json
from PIL import Image, ImageColor
import numpy as np


def compute_precision_recall(st1, st2):
    false_positives = 0 
    false_negatives = 0
    true_positives = 0 
    for scene in st1: 
        for entry_point in st1[scene]:
            for obj in st1[scene][entry_point]:
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
        for entry_point in st1[scene]:
            for obj in st1[scene][entry_point]:
                if not st1[scene][entry_point][obj] ==[]:
                    track_length_st1.append(st1[scene][entry_point][obj][0])
                    if not st2[scene][entry_point][obj] == []: 
                        track_length_st2.append(st2[scene][entry_point][obj][0])
                    else: 
                        track_length_st2.append(st1[scene][entry_point][obj][0])
    return sum(track_length_st1)/len(track_length_st1), sum(track_length_st2)/len(track_length_st2)


def main():
    mode = "ssim"
    #thrs= [0.05,0.1,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.98]
    thrs = [0.25, 0.3,0.4,0.45,0.5,0.6,0.65]
    mode = 'confidence_score'
    #mode = 'constant'
    dataset = 'a2d2'
    for thr in thrs:
        print("Thr: ", thr)
        experiment_string5 = dataset+'8'+ mode+str(thr)
        experiment_string6 = dataset  +'_69'+mode+str(thr)
        experiment_strings = [experiment_string5, experiment_string6]
        precisions, recalls, f_measures, uprecisions, urecalls, uf_measures = [],[],[],[],[],[]
        for experiment_string in experiment_strings: 
            kitti_st1 = pickle.load(open(dataset+"_pickle_files/gold_"+experiment_string+".pickle", "rb"))
            kitti_st2 = pickle.load(open(dataset+"_pickle_files/pred_"+experiment_string+".pickle", "rb"))
            kitti_st3 = pickle.load(open(dataset+"_pickle_files/estimate_gold_"+experiment_string+".pickle", "rb"))
            kitti_st1 = convert_kitti_dict(kitti_st1)
            kitti_st2 = convert_kitti_dict(kitti_st2)
            kitti_st3 = convert_kitti_dict(kitti_st3)
            upp_b_precision, upp_b_recall = compute_precision_recall(kitti_st1, kitti_st3)
            precision, recall = compute_precision_recall(kitti_st1, kitti_st2)
            """
            for obj in object_dict:
                print("INSTANCES ", object_dict[obj]["tp"]+object_dict[obj]["fn"]+object_dict[obj]["fp"])
                if object_dict[obj]["tp"] != 0:
                    obj_precision = object_dict[obj]["tp"]/(object_dict[obj]["tp"]+object_dict[obj]["fp"])
                    obj_recall = object_dict[obj]["tp"]/(object_dict[obj]["tp"]+object_dict[obj]["fn"])
                    print(obj, " precision: ", obj_precision)
                    print(obj, " recall: ", obj_recall)
                    print("f-measure: ", 2*(obj_precision*obj_recall)/(obj_precision+obj_recall))

                else:
                    print(obj, " no true positives")
            """
            if precision == 0 or recall == 0:
                f_measure = 0 
            else:
                f_measure = 2*(precision*recall)/(precision+recall)
            if upp_b_precision == 0 or upp_b_recall == 0: 
                upp_b_f_measure=0
            else:
                upp_b_f_measure = 2*(upp_b_precision*upp_b_recall)/(upp_b_precision+upp_b_recall) 
            precisions.append(precision)
            recalls.append(recall)
            f_measures.append(f_measure)
            urecalls.append(upp_b_recall)
            uprecisions.append(upp_b_precision)
            uf_measures.append(f_measure)
            print('average track length ', avg_track_length(kitti_st1, kitti_st2))
        print("Mode: ", mode)
        print("Upper bound Precision: ", np.mean(uprecisions), ' ', np.std(uprecisions))
        print("Upper bound Recall: ", np.mean(urecalls), ' ', np.std(urecalls))
        print("Upper bound F: ", np.mean(uf_measures), ' ', np.std(uf_measures))
        print("Precision: ", np.mean(precision), ' ', np.std(precisions))
        print("Recall: ", np.mean(recalls), ' ', np.std(recalls))
        print("F: ", np.mean(f_measures), ' ', np.std(f_measures))
        
if __name__ == "__main__":
    main()
