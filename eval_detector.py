import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    box format: tl_row, tl_col, br_row, br_col
    '''
    i_tl_row = max(box_1[0], box_2[0])        #max tl_row
    i_tl_col = max(box_1[1], box_2[1])        #max tl_col
    i_br_row = min(box_1[2], box_2[2])        #min br_row
    i_br_col = min(box_1[3], box_2[3])        #min br_col
    
    intersec = max(0, i_br_row-i_tl_row) * max(0, i_br_col-i_tl_col)
    union = (box_1[2]-box_1[0])*(box_1[3]-box_1[1]) + (box_2[2]-box_2[0])*(box_2[3]-box_2[1])
    iou = intersec/(union-intersec)
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr, conf_thr):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        for i in range(len(gt)):
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                if iou>=iou_thr: 
                    if pred[j][4]>=conf_thr:
                        TP+=1
                    else:
                        FN+=1
                else: 
                    if pred[j][4]>=conf_thr:
                        FP+=1


    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 

iou_thrs = [0.01, 0.1, 0.25, 0.5]
for i,iou_thr in enumerate(iou_thrs):
    confidence_thrs = [0.05, 0.1, 0.3, 0.5, 0.7, 0.95, 0.9999]
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thr, conf_thr=conf_thr)

    # Plot training set PR curves

    if done_tweaking:
        print('Code for plotting test set PR curves.')
        precision = tp_train/(tp_train + fp_train)
        recall = tp_train/(tp_train + fn_train)
        print(precision, recall)
        plt.plot(recall, precision)
plt.legend(['iou_thr:0.01', 'iou_thr:0.1', 'iou_thr:0.25', 'iou_thr:0.5'])
plt.xlabel('recall')
plt.ylabel('precision')
plt.show()