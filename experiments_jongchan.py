import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory

from StaticTransformer import StaticTransformer
from IndexBasedTransformer import IndexBasedTransformer
from AggregateTransformer import AggregateTransformer

from PrefixLengthBucketer import PrefixLengthBucketer
from ClusterBasedBucketer import ClusterBasedBucketer

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch

import time
import os
import sys
from sys import argv
import pickle
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


dataset_ref = argv[1] # Example: bpic2011_f1
params_dir = "/home/jongchan/case_stability/predictive-monitoring-benchmark-master/experiments/optimal_params"
results_dir = "/home/jongchan/case_stability/results"
bucket_method = argv[2] # Example: prefix
cls_encoding = argv[3] # Example: index
cls_method = argv[4] # Example: xgboost
max_prefix = int(argv[5]) # Example: 10
if bucket_method == "cluster":
    n_clusters = int(argv[6])


gap = 1
n_iter = 1

if bucket_method == "state":
    bucket_encoding = "last"
else:
    bucket_encoding = "agg"

method_name = "%s_%s"%(bucket_method, cls_encoding)

dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(1,6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    "sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"]
}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]
    
train_ratio = 0.8
random_state = 22

# create results directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))
    
dataset_name = datasets[0]

# load optimal params
if bucket_method == "cluster":
    optimal_params_filename = os.path.join(params_dir, "optimal_params_%s_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name, n_clusters))
else:
    optimal_params_filename = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))

with open(optimal_params_filename, "rb") as fin:
    args = pickle.load(fin)

# read the data
dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                    'static_cat_cols': dataset_manager.static_cat_cols,
                    'static_num_cols': dataset_manager.static_num_cols,
                    'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                    'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                    'fillna': True}

# determine min and max (truncated) prefix lengths
min_prefix_length = 1 #User can define this
max_prefix_length = max_prefix #User can define this

# split into training and test
train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")

if gap > 1:
    outfile = os.path.join(results_dir,
                           "performance_results_%s_%s_%s_gap%s.csv" % (cls_method, dataset_name, method_name, gap))
else:
    outfile = os.path.join(results_dir,
                           "performance_results_%s_%s_%s_%s.csv" % (cls_method, dataset_name, method_name, str(max_prefix)))

start_test_prefix_generation = time.time()
dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
test_prefix_generation_time = time.time() - start_test_prefix_generation

offline_total_times = []
online_event_times = []
train_prefix_generation_times = []

for ii in range(n_iter):
    # create prefix logs
    start_train_prefix_generation = time.time()
    dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length, gap)
    train_prefix_generation_time = time.time() - start_train_prefix_generation
    train_prefix_generation_times.append(train_prefix_generation_time)
    # Bucketing prefixes based on control flow
    bucketer_args = {'encoding_method': bucket_encoding,
                     'case_id_col': dataset_manager.case_id_col,
                     'cat_cols': [dataset_manager.activity_col],
                     'num_cols': [],
                     'random_state': random_state}

    if bucket_method == "prefix":
        bucketer = PrefixLengthBucketer(case_id_col = bucketer_args['case_id_col'])
    elif bucket_method == "cluster":
        bucketer_encoder = AggregateTransformer(case_id_col=bucketer_args['case_id_col'], cat_cols=bucketer_args['cat_cols'], num_cols=bucketer_args['num_cols'], boolean=False, fillna=True)
        clustering = KMeans(n_clusters, random_state=bucketer_args['random_state'])
        bucketer = ClusterBasedBucketer(encoder=bucketer_encoder, clustering=clustering)

    start_offline_time_bucket = time.time()
    bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
    offline_time_bucket = time.time() - start_offline_time_bucket
    bucket_assignments_test = bucketer.predict(dt_test_prefixes)
    preds_all = []
    test_y_all = []
    nr_events_all = []
    buckets_all = []
    offline_time_fit = 0
    current_online_event_times = []
    for bucket in set(bucket_assignments_test):
        current_args = args
        relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[
            bucket_assignments_train == bucket]
        relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[
            bucket_assignments_test == bucket]
        dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket)
        nr_events_all.extend(list(dataset_manager.get_prefix_lengths(dt_test_bucket)))
        buckets_all.extend([bucket]*len(list(dataset_manager.get_prefix_lengths(dt_test_bucket))))
        if len(relevant_train_cases_bucket) == 0:
            preds = [dataset_manager.get_class_ratio(train)] * len(relevant_test_cases_bucket)
            current_online_event_times.extend([0] * len(preds))
        else:
            dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                           relevant_train_cases_bucket)  # one row per event
            train_y = dataset_manager.get_label_numeric(dt_train_bucket)
            if len(set(train_y)) < 2:
                preds = [train_y[0]] * len(relevant_test_cases_bucket)
                current_online_event_times.extend([0] * len(preds))
                test_y_all.extend(dataset_manager.get_label_numeric(dt_test_bucket))
            else:
                start_offline_time_fit = time.time()
                feature_combiner = FeatureUnion(
                    [('static', StaticTransformer(case_id_col = cls_encoder_args['case_id_col'], cat_cols=cls_encoder_args['static_cat_cols'], num_cols=cls_encoder_args['static_num_cols'], fillna=cls_encoder_args['fillna'])), ('index', IndexBasedTransformer(case_id_col = cls_encoder_args['case_id_col'], cat_cols=cls_encoder_args['dynamic_cat_cols'], num_cols=cls_encoder_args['dynamic_num_cols'], max_events=None, fillna=cls_encoder_args['fillna']))])
                if cls_method == "rf":
                    cls = RandomForestClassifier(n_estimators=500,
                                                 max_features=current_args['max_features'],
                                                 random_state=random_state)
                elif cls_method == "xgboost":
                    cls = xgb.XGBClassifier(objective='binary:logistic',
                                            n_estimators=500,
                                            learning_rate=current_args['learning_rate'],
                                            subsample=current_args['subsample'],
                                            max_depth=int(current_args['max_depth']),
                                            colsample_bytree=current_args['colsample_bytree'],
                                            min_child_weight=int(current_args['min_child_weight']),
                                            seed=random_state)
                elif cls_method == "gbm":
                    cls = GradientBoostingClassifier(n_estimators=500,
  	                                             learning_rate=current_args['learning_rate'],
        	                                     min_samples_split=int(current_args['min_samples_split']),
        	                                     max_depth=int(current_args['max_depth']),
                	                             random_state=random_state)
                elif cls_method == "logit":
                    cls = LogisticRegression(C=2 ** current_args['C'],
                                             random_state=random_state)
                elif cls_method == "svm":
                    cls = SVC(C=2 ** current_args['C'],
                              gamma=2 ** current_args['gamma'],
                              random_state=random_state)
                if cls_method == "svm" or cls_method == "logit":
                    pipeline = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
                else:
                    pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])
                pipeline.fit(dt_train_bucket, train_y)
                offline_time_fit += time.time() - start_offline_time_fit
                # predict separately for each prefix case
                preds = []
                test_all_grouped = dt_test_bucket.groupby(dataset_manager.case_id_col)
                for _, group in test_all_grouped:
                    test_y_all.extend(dataset_manager.get_label_numeric(group))
                    start = time.time()
                    _ = bucketer.predict(group)
                    if cls_method == "svm":
                        pred = pipeline.decision_function(group)
                    else:
                        preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
                        pred = pipeline.predict_proba(group)[:, preds_pos_label_idx]
                    pipeline_pred_time = time.time() - start
                    current_online_event_times.append(pipeline_pred_time / len(group))
                    preds.extend(pred)
        preds_all.extend(preds)

with open(outfile, 'w') as fout:
    fout.write("%s;%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "nr_events", "n_iter", "metric", "score"))
    dt_results = pd.DataFrame({"actual": test_y_all, "predicted": preds_all, "nr_events": nr_events_all, "bucket": buckets_all})
    dt_results.to_csv(results_dir + "/dt_results_" + dataset_name + "_" + method_name + "_" + cls_method + "_" + str(max_prefix) + ".csv", index = False)
    for nr_events, group in dt_results.groupby("nr_events"):
        if len(set(group.actual)) < 2:
            fout.write(
                "%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1, "auc", np.nan))
        else:
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1, "auc",
                                                   roc_auc_score(group.actual, group.predicted)))
    fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
    dataset_name, method_name, cls_method, -1, -1, "auc", roc_auc_score(dt_results.actual, dt_results.predicted)))
    
