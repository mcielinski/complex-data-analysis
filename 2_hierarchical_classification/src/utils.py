import os
import random
import sys
import time
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC

from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn_hierarchical_classification.metrics import h_fbeta_score, multi_labeled

from tqdm import tqdm

seed_value = 2021
random.seed(seed_value)
np.random.seed(seed_value)


def file_path(file_name, data_dir='data/imclef07d', add_csv=False):
    if add_csv:
        file_name = file_name + '.csv'
    return os.path.join(data_dir, file_name)


def get_class_hierarchy(tree_data):
    class_hierarchy = {}

    def build_class_hierarchy(tree_data):
        node_id = None
        for key, value in tree_data.items():
            if key == 'id':
                node_id = value
                class_hierarchy[node_id] = []
            if key == 'children':
                for node in value:
                    class_hierarchy[node_id].append(node['id'])
                    build_class_hierarchy(node)
        class_hierarchy_copy = class_hierarchy.copy()
        for node_id, el in class_hierarchy_copy.items():
            if len(el) == 0:
                class_hierarchy.pop(node_id)
    
    build_class_hierarchy(tree_data)

    class_hierarchy[ROOT] = class_hierarchy['19'] 
    class_hierarchy.pop('19')

    return class_hierarchy


def classify(train_ds, test_ds, class_hierarchy, graph, clf, mlb=None):
    (x_train, y_train) = train_ds
    (x_test, y_test) = test_ds

    if mlb:
        y_train = mlb.fit_transform(y_train)
    
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    if mlb:
        y_pred = mlb.inverse_transform(y_pred)
    
    with multi_labeled(y_test, y_pred, graph) as (y_test_, y_pred_, graph_):
        h_fbeta = h_fbeta_score(
            y_test_,
            y_pred_,
            graph_,
        )
    result = {'h_fbeta': h_fbeta}

    return result


def lcn_hc(train_ds, test_ds, class_hierarchy, graph, base_estimator):
    mlb = MultiLabelBinarizer()
    clf_lcn = OneVsRestClassifier(estimator=base_estimator)
    result = classify(
        train_ds=train_ds,
        test_ds=test_ds, 
        class_hierarchy=class_hierarchy, 
        graph=graph, 
        clf=clf_lcn, 
        mlb=mlb
    )

    return result


def lcpl_hc(train_ds, test_ds, class_hierarchy, graph, base_estimator):
    (x_train, y_train) = train_ds
    (x_test, y_test) = test_ds

    preds = []
    for i in range(3):
        clf_lcpl = sklearn.base.clone(base_estimator)
        clf_lcpl.fit(x_train, y_train.apply(lambda x: x[i]))
        y_pred = clf_lcpl.predict(x_test)
        preds.append(y_pred)
    
    y_pred = list(zip(preds[0], preds[1], preds[2]))

    with multi_labeled(y_test, y_pred, graph) as (y_test_, y_pred_, graph_):
        h_fbeta = h_fbeta_score(
            y_test_,
            y_pred_,
            graph_,
        )
    result = {'h_fbeta': h_fbeta}

    return result


def lcpn_hc(train_ds, test_ds, class_hierarchy, graph, base_estimator):
    clf_lcpn = HierarchicalClassifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy,
        algorithm='lcpn'
    )

    (x_train, y_train) = train_ds
    (x_test, y_test) = test_ds

    train_ds_ = (x_train, y_train.apply(lambda x: x[-1]))
    test_ds_ = (x_test, y_test.apply(lambda x: x[-1]))

    result = classify(
        train_ds=train_ds_,
        test_ds=test_ds_, 
        class_hierarchy=class_hierarchy, 
        graph=graph, 
        clf=clf_lcpn
    )

    return result


def big_bang_hc(train_ds, test_ds, class_hierarchy, graph, base_estimator):
    (x_train, y_train) = train_ds
    (x_test, y_test) = test_ds

    train_ds_ = (x_train, y_train.apply(lambda x: x[-1]))
    test_ds_ = (x_test, y_test.apply(lambda x: x[-1]))

    mlb = MultiLabelBinarizer()
    result = classify(
        train_ds=train_ds,
        test_ds=test_ds, 
        class_hierarchy=class_hierarchy, 
        graph=graph, 
        clf=base_estimator, 
        mlb=mlb
    )

    return result


def flat_hc(train_ds, test_ds, class_hierarchy, graph, base_estimator):
    (x_train, y_train) = train_ds
    (x_test, y_test) = test_ds

    train_ds_ = (x_train, y_train.apply(lambda x: x[-1]))
    test_ds_ = (x_test, y_test.apply(lambda x: x[-1]))

    result = classify(
        train_ds=train_ds_,
        test_ds=test_ds_, 
        class_hierarchy=class_hierarchy, 
        graph=graph, 
        clf=base_estimator
    )

    return result


def experiments(base_estimators_dict, clf_func_dict, train_ds, test_ds, class_hierarchy, graph):
    results_df = pd.DataFrame()
    
    for clf_func_name in tqdm(list(clf_func_dict.keys())):
        for base_est_name in list(base_estimators_dict.keys()):
            clf_func = clf_func_dict[clf_func_name]
            base_est = base_estimators_dict[base_est_name]
            try:
                result = clf_func(
                    train_ds=train_ds,
                    test_ds=test_ds,
                    class_hierarchy=class_hierarchy,
                    graph=graph,
                    base_estimator=base_est
                )
                result = result['h_fbeta']
            except ValueError:
                result = np.nan

            results_df = results_df.append({
                'clf': clf_func_name, 
                'base_estimator': base_est_name, 
                'h_fbeta': result
            }, ignore_index=True)

    return results_df


def get_filtred_metrics(results_df, base_estimator):
    f1s = results_df[
        results_df['base_estimator'] == base_estimator
    ].sort_values(by=['clf'])['h_fbeta']

    return f1s.tolist()