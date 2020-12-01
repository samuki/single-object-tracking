import pickle
import json
from PIL import Image, ImageColor


def compute_precision_recall(st1, st2):
    false_positives = 0 
    false_negatives = 0 
    true_positives = 0 
    #true_negatives = 0 
    for scene in st1: 
        for obj in st1[scene]:
            if st2[scene][obj] > st1[scene][obj]:
                false_negatives+=1
            elif st2[scene][obj] < st1[scene][obj]:
                false_positives+=1
            else: 
                true_positives+=1
    if true_positives == 0:
        precision = 0 
        recall = 0
    else:
        precision = true_positives/(true_positives+false_positives)
        recall = true_positives/(true_positives+false_negatives)
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
    object_dict = {}
    for scene in st1: 
        for obj in st1[scene]:
            if not obj in object_dict:
                object_dict[obj] = {}
                object_dict[obj]["fp"] = 0
                object_dict[obj]["fn"] = 0
                object_dict[obj]["tp"] = 0
            if st2[scene][obj] > st1[scene][obj]:
                object_dict[obj]["fn"]+=1
            elif st2[scene][obj] < st1[scene][obj]:
                object_dict[obj]["fp"]+=1
            else: 
                object_dict[obj]["tp"]+=1
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
    with open('../../Uni/9.Semester/AP/class_list.json') as json_file: 
        lookup = json.load(json_file) 
    lookup = {ImageColor.getcolor(k, "RGB"):v for k,v in lookup.items()}
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
    #iou_dict1 =  pickle.load(open(dir1+"iou_dict.pickle", "rb"))
    #iou_dict2 =  pickle.load(open(dir2+"iou_dict.pickle", "rb"))
    #iou_dict3 = pickle.load(open(dir3+"iou_dict.pickle", "rb"))
    #compare_iou(iou_dict1, iou_dict2, lookup)
    #compare_iou(iou_dict1,convert_iou_object_dict(iou_dict3), lookup)
    #tl1, tl2 = avg_track_length(st1, st2)
    #print("len st1: ", sum(track_length_st1)/len(track_length_st1))
    #print("len st2: ", sum(track_length_st2)/len(track_length_st2))

    #tl1, tl3 = avg_track_length(st1, st3)
    #print("len st1: ", tl1)
    #print("len st3: ", tl3)

    # dict2 = ssim but wrong tracking window
    # 750 autoenc is autoenc with 8 classes + 0.5 thr
    # average autoenc is no bl 
    mode = "_average_autoenc"
    mode = "othertest"
    #mode = "autoenc_05"
    kitti_st1 = pickle.load(open("pickle_files/gold_"+mode+".pickle", "rb"))
    kitti_st2 = pickle.load(open("pickle_files/pred_"+mode+".pickle", "rb"))
    kitti_st3 = pickle.load(open("pickle_files/estimate_gold_"+mode+".pickle", "rb"))
    kitti_st1 = convert_kitti_dict(kitti_st1)
    kitti_st2 = convert_kitti_dict(kitti_st2)
    kitti_st3 = convert_kitti_dict(kitti_st3)
    upp_b_precision, upp_b_recall = compute_precision_recall(kitti_st1, kitti_st3)
    precision, recall = compute_precision_recall(kitti_st1, kitti_st2)
    print("Upper bound Precision: ", upp_b_precision)
    print("Upper bound Recall: ", upp_b_recall)
    print("Precision: ", precision)
    print("Recall: ", recall)
    


if __name__ == "__main__":
    main()