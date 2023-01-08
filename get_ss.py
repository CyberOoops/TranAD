import numpy as np
from argparse import ArgumentParser
import os
import csv 

def calc_p2p(predict, actual):
    tp = np.sum(predict * actual)
    tn = np.sum((1-predict) * (1-actual))
    fp = np.sum(predict * (1-actual))
    fn = np.sum((1-predict) * actual)
    
    precision = tp / (tp + fp + 0.000001)
    recall = tp / (tp + fn + 0.000001)
    f1 = 2 * precision * recall / (precision + recall + 0.000001)
    return f1, precision, recall, tp, tn, fp, fn

def point_adjust(score, label, thres):
    if len(score) != len(label):
        raise ValueError("score len is %d and label len is %d\n" %(len(score), len(label)))
    
    score = np.asarray(score)
    label = np.asarray(label)
    
    # print("thre is %f"%(thres))
    predict = score > thres
    actual = label > 0.1
    anomaly_state = False
    
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
            
    return predict, actual

def get_trad_f1(score, label):
    maxx = float(score.max())
    minn = float(score.min())
    
    actual = label > 0.1
    grain = 1000
    max_f1 = 0.0
    max_f1_thres = 0.0
    p = 0
    r = 0
    for i in range(grain):
        thres = (maxx-minn)/grain * i + minn
        # thres = i / grain 
        predict = score > thres
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_thres = thres
            p = precision
            r = recall
            
    print("max f1 score is %f and threshold is %f\n" %(max_f1, max_f1_thres))
    return max_f1, max_f1_thres, p, r
    
def get_best_f1(score, label):
    score = np.asarray(score)
    maxx = float(score.max())
    minn = float(score.min())
    
    grain = 1000
    max_f1 = 0.0
    max_f1_thres = 0.0
    p = 0
    r = 0
    for i in range(grain):
        thres = (maxx-minn)/grain * i + minn
        # thres = i / grain
        predict, actual = point_adjust(score, label, thres=thres)
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_thres = thres
            p = precision
            r = recall
            
    # print("max f1 score is %f and threshold is %f\n" %(max_f1, max_f1_thres))
    return max_f1, max_f1_thres, p, r
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', default="SMD", type=str)
    hp = parser.parse_args()
    
    base_path = os.getcwd()
    dataset = hp.dataset
    if dataset == "SMD":
        f1_list = []
        p_list = []
        r_list = []
        SMD_machine = ["machine-1-1","machine-1-2","machine-1-3","machine-1-4","machine-1-5","machine-1-6","machine-1-7","machine-1-8","machine-2-1","machine-2-2","machine-2-3","machine-2-4","machine-2-5","machine-2-6","machine-2-7","machine-2-8","machine-2-9","machine-3-1","machine-3-2","machine-3-3","machine-3-4","machine-3-5","machine-3-6","machine-3-7","machine-3-8","machine-3-9","machine-3-10","machine-3-11"]
        for machine in SMD_machine:
            gt_path = "label_result/SMD/%s_gt_label.txt" % machine
            y_path = "label_result/SMD/%s_y_label.txt" % machine
            
            label = np.genfromtxt(gt_path, delimiter='\n')
            score = np.genfromtxt(y_path, delimiter='\n')
            f1, thres, p, r = get_best_f1(score, label)
            print("%s has max f1 %10f,p  %10f, r %10f with threshold %10f" %(machine, f1, p, r, thres))
            f1_list.append(f1), p_list.append(p), r_list.append(r)
            
        print("==========AVG RESULT=========")
        print(np.mean(np.asarray(f1_list)), np.mean(np.asarray(p_list)), np.mean(np.asarray(r_list)))
        
            
    else:
        gt_path = "label_result/%s_gt_label.txt" % dataset
        y_path = "label_result/%s_y_label.txt" % dataset
        
        label = np.genfromtxt(gt_path, delimiter='\n')
        score = np.genfromtxt(y_path, delimiter='\n')
        f1, thres, p, r = get_best_f1(score, label)
        print("%s has max f1 %10f,p  %10f, r %10f with threshold %10f" %(dataset, f1, p, r, thres))
        f1, thres, p, r = get_trad_f1(score, label)
        print("%s has max f1 %10f,p  %10f, r %10f with threshold %10f" %(dataset, f1, p, r, thres))
    
