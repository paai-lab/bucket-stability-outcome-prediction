import pandas as pd
import numpy as np

import os
import sys
from sys import argv

from sklearn.metrics import roc_auc_score

result_dir = "/home/jongchan/case_stability/results"

dataset_name = argv[1]
bucket_name = argv[2]
clf_name = argv[3]
n_bucket = argv[4]

data_dir = result_dir + "/dt_results_" + dataset_name + "_" + bucket_name + "_index_" + clf_name + "_" + n_bucket + ".csv"

data = pd.read_csv(data_dir)
unique_bucket = data.bucket.unique()

ibs = [None] * int(n_bucket)
ibp = [None] * int(n_bucket)
cbs = 0

if bucket_name == "prefix":
    #Intra-bucket prediction stability (IBS)
    for i in range(int(n_bucket)):
        temp_data = data[data.bucket == i+1]
        if len(temp_data.actual.unique()) != 2:
            continue
        actual = temp_data.actual.reset_index(drop=True)
        predicted = temp_data.predicted.reset_index(drop=True)
        ibp[i] = roc_auc_score(actual, predicted)
        for j in range(len(predicted)):
            if predicted[j] < 0.5:
                predicted[j] = 1 - predicted[j]
        ibs_weight = 0.05
        for k in range(len(predicted)):
            if abs(0.5-predicted[k]) > 0.4 and abs(0.5-predicted[k]) <= 0.5:
                predicted[k] = predicted[k] * (1 + (2*ibs_weight))
            if abs(0.5-predicted[k]) > 0.3 and abs(0.5-predicted[k]) <= 0.4:
                predicted[k] = predicted[k] * (1 + (1*ibs_weight))
            if abs(0.5-predicted[k]) > 0.2 and abs(0.5-predicted[k]) <= 0.3:
                predicted[k] = predicted[k] * (1 + (0*ibs_weight))
            if abs(0.5-predicted[k]) > 0.1 and abs(0.5-predicted[k]) <= 0.2:
                predicted[k] = predicted[k] * (1 - (1*ibs_weight))
            if abs(0.5-predicted[k]) >= 0 and abs(0.5-predicted[k]) <= 0.1:
                predicted[k] = predicted[k] * (1 - (2*ibs_weight))
        if np.min(predicted) == np.max(predicted):
            for ind in range(len(predicted)):
                predicted[ind] = 1
        else:
            predicted = (predicted-np.min(predicted))/(np.max(predicted)-np.min(predicted))

        ibs[i] = np.average(predicted)
    
    ibs_original = ibs[:]
    ibp_original = ibp[:]
    
    sum = 0
    for i in range(len(ibs)):
        if ibs[i] != None:
            sum = sum + len(data[data.bucket == i+1])
    
    for i in range(len(ibs)):
        if ibs[i] != None:
            ibs[i] = ibs[i] * (len(data[data.bucket == i+1])/sum)
    
    ibs_avg = np.sum([i for i in ibs if i])
    
    
    #Intra-bucket performance (IBP)
    sum = 0
    for i in range(len(ibp)):
        if ibp[i] != None:
            sum = sum + len(data[data.bucket == i+1])
    
    for i in range(len(ibp)):
        if ibp[i] != None:
            ibp[i] = ibp[i] * (len(data[data.bucket == i+1])/sum)
    
    ibp_avg = np.sum([i for i in ibp if i])
    
    
    #Cross-bucket performance stability (CBP)
    sum = 0
    for i in range(len(ibs)):
        if ibs[i] != None:
            sum = sum + len(data[data.bucket == i+1])
    
    cbs_part = [None] * int(((len([i for i in ibs if i])*(len([i for i in ibs if i])-1))/2))
    index = 0
    for j in range(len(ibp)):
        for k in range(len(ibp)):
            if j < k:
                if (ibp[j] != None) and (ibp[k] != None):
                    cbs_part[index] = abs(ibp_original[j]-ibp_original[k])*((len(data[data.bucket == j+1])+len(data[data.bucket == k+1]))/sum)
                    index = index + 1
    
    cbs = 1 - (np.sum(cbs_part) / (len([i for i in ibp if i])-1))

elif bucket_name == "cluster":
    # Intra-bucket prediction stability (IBS)
    for i in range(int(n_bucket)):
        temp_data = data[data.bucket == i]
        if len(temp_data.actual.unique()) != 2:
            continue
        actual = temp_data.actual.reset_index(drop=True)
        predicted = temp_data.predicted.reset_index(drop=True)
        ibp[i] = roc_auc_score(actual, predicted)
        for j in range(len(predicted)):
            if predicted[j] < 0.5:
                predicted[j] = 1 - predicted[j]
        ibs_weight = 0.05
        for k in range(len(predicted)):
            if abs(0.5 - predicted[k]) > 0.4 and abs(0.5 - predicted[k]) <= 0.5:
                predicted[k] = predicted[k] * (1 + (2 * ibs_weight))
            if abs(0.5 - predicted[k]) > 0.3 and abs(0.5 - predicted[k]) <= 0.4:
                predicted[k] = predicted[k] * (1 + (1 * ibs_weight))
            if abs(0.5 - predicted[k]) > 0.2 and abs(0.5 - predicted[k]) <= 0.3:
                predicted[k] = predicted[k] * (1 + (0 * ibs_weight))
            if abs(0.5 - predicted[k]) > 0.1 and abs(0.5 - predicted[k]) <= 0.2:
                predicted[k] = predicted[k] * (1 - (1 * ibs_weight))
            if abs(0.5 - predicted[k]) >= 0 and abs(0.5 - predicted[k]) <= 0.1:
                predicted[k] = predicted[k] * (1 - (2 * ibs_weight))
        if np.min(predicted) == np.max(predicted):
            for ind in range(len(predicted)):
                predicted[ind] = 1
        else:
            predicted = (predicted-np.min(predicted))/(np.max(predicted)-np.min(predicted))

        ibs[i] = np.average(predicted)

    ibs_original = ibs[:]
    ibp_original = ibp[:]

    sum = 0
    for i in range(len(ibs)):
        if ibs[i] != None:
            sum = sum + len(data[data.bucket == i])

    for i in range(len(ibs)):
        if ibs[i] != None:
            ibs[i] = ibs[i] * (len(data[data.bucket == i]) / sum)

    ibs_avg = np.sum([i for i in ibs if i])

    # Intra-bucket performance (IBP)
    sum = 0
    for i in range(len(ibp)):
        if ibp[i] != None:
            sum = sum + len(data[data.bucket == i])

    for i in range(len(ibp)):
        if ibp[i] != None:
            ibp[i] = ibp[i] * (len(data[data.bucket == i]) / sum)

    ibp_avg = np.sum([i for i in ibp if i])

    # Cross-bucket performance stability (CBP)
    sum = 0
    for i in range(len(ibs)):
        if ibs[i] != None:
            sum = sum + len(data[data.bucket == i])

    cbs_part = [None] * int(((len([i for i in ibs if i]) * (len([i for i in ibs if i]) - 1)) / 2))
    index = 0
    for j in range(len(ibp)):
        for k in range(len(ibp)):
            if j < k:
                if (ibp[j] != None) and (ibp[k] != None):
                    cbs_part[index] = abs(ibp_original[j] - ibp_original[k]) * (
                                (len(data[data.bucket == j]) + len(data[data.bucket == k])) / sum)
                    index = index + 1

    cbs = 1 - (np.sum(cbs_part) / (len([i for i in ibp if i]) - 1))

print("ibs: " + str(ibs_original))
print("ibs average: " + str(ibs_avg))
print("ibp: " + str(ibp_original))
print("ibp average: " + str(ibp_avg))
print("cbs: " + str(cbs))
