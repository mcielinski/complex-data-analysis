import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

seed_value = 2021
random.seed(seed_value)
np.random.seed(seed_value)


def build_communication_graph(communication_df, employee_ids_range=(1, 167)):
    G_communication = nx.Graph()

    (min_employee_id, max_employee_id) = employee_ids_range
    G_communication.add_nodes_from(
        list(range(min_employee_id, max_employee_id+1))
    )

    for _, row in communication_df.iterrows():
        G_communication.add_edge(row['Sender'], row['Recipient'], weight=row['count'])

    return G_communication


def build_reportsto_graph(reportsto_df, employee_ids_range=(1, 167)):
    G_reportsto = nx.Graph()

    (min_employee_id, max_employee_id) = employee_ids_range
    G_reportsto.add_nodes_from(
        list(range(min_employee_id, max_employee_id+1))
    )

    for _, row in reportsto_df.iterrows():
        G_reportsto.add_edge(row['ID'], row['ReportsToID'])

    return G_reportsto


def graph_info(G):
    print(f'Number of nodes: {G.number_of_nodes()}')
    print(f'Number of edges: {G.number_of_edges()}')

    conn_comp = list(nx.connected_components(G))
    print(f'Number of connected components: {len(conn_comp)}')


def draw_graph(G, high_mgmt, mid_mgmt):
    color_map = []
    for node in G:
        if node in high_mgmt:
            color_map.append('r')
        elif node in mid_mgmt:
            color_map.append('g')
        else:
            color_map.append('b')

    plt.figure(figsize=(12, 8))

    nx.draw(
        G, 
        pos = nx.spring_layout(G),
        node_size=50,
        linewidths=0,
        width=0.1,
        node_color=color_map,
        with_labels=False
    )

    plt.show()


def set_static_attrs(G, reportsto_grupby_reportsToID, communication_groupby_sender, communication_groupby_recipient):
    attrs = {}
    for node in G:
        try:
            reporters_num = int(reportsto_grupby_reportsToID[
                reportsto_grupby_reportsToID['ReportsToID'] == node
            ]['count'])
        except TypeError:
            reporters_num = 0
        try:
            emails_sent_num = int(communication_groupby_sender[
                communication_groupby_sender['Sender'] == node
            ]['count'])
        except TypeError:
            emails_sent_num = 0
        try:
            emails_received_num = int(communication_groupby_recipient[
                communication_groupby_recipient['Recipient'] == node
            ]['count'])
        except TypeError:
            emails_received_num = 0 
        attrs[node] = {
            'node': node, 
            'reporters_num': reporters_num, 
            'emails_sent_num': emails_sent_num, 
            'emails_received_num': emails_received_num
        }
    nx.set_node_attributes(G, attrs)


def set_network_attributes(G):
    # https://www.geeksforgeeks.org/network-centrality-measures-in-a-graph-using-networkx-python/
    nx.set_node_attributes(G, nx.degree_centrality(G), 'degree')
    nx.set_node_attributes(G, nx.closeness_centrality(G), 'closeness')
    nx.set_node_attributes(G, nx.betweenness_centrality(G), 'betweenness')
    nx.set_node_attributes(G, nx.pagerank(G), 'pagerank')
    nx.set_node_attributes(G, nx.eigenvector_centrality(G), 'eigenvector')


def set_labels(G, uncover_nodes, high_mgmt, mid_mgmt):
    attrs = {}
    for node in uncover_nodes:
        if node in high_mgmt:
            y_true = 2
        elif node in mid_mgmt:
            y_true = 1
        else:
            y_true = 0
        attrs[node] = {
            'y_true': y_true
        }
    nx.set_node_attributes(G, attrs)


def set_node_dynamic_attr(G, node, neighbor):
    if 'y_true' in G.nodes[neighbor]:
        label = G.nodes[neighbor]['y_true']
        new_value = G.nodes[node][label] + 1
        nx.set_node_attributes(G, {node: {label: new_value}})
    elif 'y_pred' in G.nodes[neighbor]:
        label = G.nodes[neighbor]['y_pred']
        new_value = G.nodes[node][label] + 1
        nx.set_node_attributes(G, {node: {label: new_value}})


def set_dynamic_attrs(G):
    for node in G:
        nx.set_node_attributes(G, {node: {0: 0, 1: 0, 2: 0}})
    for node in G:
        for neighbor in G.neighbors(node):
            set_node_dynamic_attr(G, node, neighbor)


def get_uncover_nodes(G, sorted_by, num=10):
    # może tak
    uncover_nodes = sorted(G.nodes, key=lambda x: G.nodes[x][sorted_by], reverse=True)
    #uncover_nodes = [node for node in sorted(G.nodes, key=lambda x: G.nodes[x][sorted_by], reverse=True)]
    return uncover_nodes[:num]


def prepare_dataset(G, uncover_nodes, random_clf=False):
    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    for node in G:
        if node in uncover_nodes:
            x_train = x_train.append(G.nodes[node], ignore_index=True)
        else:
            x_test = x_test.append(G.nodes[node], ignore_index=True)
    x_train.set_index('node', drop=True, inplace=True)
    x_test.set_index('node', drop=True, inplace=True)

    y_train = x_train['y_true']
    x_train = x_train.drop(columns='y_true')

    if random_clf:
        y_test = x_test['y_true']
        x_test = x_test.drop(columns='y_true')
        to_return = (x_train, y_train, x_test, y_test)
    else:
        to_return = (x_train, y_train, x_test)

    return to_return


def get_y(G):
    y_trues = []
    y_preds = []
    for node in G:
        if 'y_pred' in G.nodes[node]:
            y_trues.append(G.nodes[node]['y_true'])
            y_preds.append(int(G.nodes[node]['y_pred']))
    return y_trues, y_preds


def get_uncover_percentage_f1s(results_df, clf='AdaBoostClassifier', sorted_by='degree'):
    f1s = results_df[
        (results_df['clf'] == clf) &
        (results_df['sorted_by'] == sorted_by)
    ].sort_values(by=['uncover_percentage'])['f1_score']

    return f1s.tolist()


def ica(G, sorted_by, clf, high_mgmt, mid_mgmt, uncover_nodes_percentage=0.25, iterations=100):
    uncover_nodes_num = int(uncover_nodes_percentage * G.number_of_nodes())
    uncover_nodes = get_uncover_nodes(G, sorted_by, uncover_nodes_num)

    set_labels(G, uncover_nodes, high_mgmt, mid_mgmt)
    set_dynamic_attrs(G)

    x_train, y_train, x_test = prepare_dataset(G, uncover_nodes)

    clf = clf.fit(x_train, y_train)
    y_preds = clf.predict(x_test)

    for node, y_pred in zip(x_test.index, y_preds):
        nx.set_node_attributes(G, {node: {'y_pred': int(y_pred)}})
    stabilized = False

    iteration = 0
    while not stabilized and iteration < iterations:
        iteration +=1
        stop_flag = True
        for node in sorted(x_test.index, key=lambda x: x_test.loc[node, sorted_by], reverse=True):
            for neighbor in G.neighbors(node):
                set_node_dynamic_attr(G, node, neighbor)
            for label in range(3):
                x_test.loc[node, label] = G.nodes[node][label]

            y_pred = int(clf.predict([x_test.loc[node]])[0])
            if y_pred != G.nodes[node]['y_pred']:
                stop_flag = False
            nx.set_node_attributes(G, {node: {'y_pred': y_pred}})
        
        if stop_flag:
            stabilized = True

    set_labels(G, x_test.index.values.tolist(), high_mgmt, mid_mgmt)

    return get_y(G)


def random_clf(G, clf, high_mgmt, mid_mgmt, uncover_nodes_percentage=0.25):
    uncover_nodes_num = int(uncover_nodes_percentage * G.number_of_nodes())
    uncover_nodes = random.sample(G.nodes, uncover_nodes_num)

    set_labels(G, G.nodes, high_mgmt, mid_mgmt) # 2nd param to nie uncover_nodes?

    x_train, y_train, x_test, y_test = prepare_dataset(G, uncover_nodes, random_clf=True)

    clf = clf.fit(x_train, y_train)

    return y_test, clf.predict(x_test)