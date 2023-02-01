import numpy as np
import pandas as pd
import networkx as nx
from utils import load_features, modify_nodes, get_test_edge, link_predict_without_lr, link_predict_with_lr
from utils import normalize_by_row, normalize_by_column, create_D_inverse, create_A_tilde
from scipy import sparse
from ade import ADE
import csv



def main():

    cnt = 100

    path = 'random_walk'

    # data filename
    graph_type = 'direct'
    dataset = 'cornell'

    # data file path
    path_feature = '../dataset/node_level/{}/features.csv'.format(dataset)
    path_edge = '../dataset/node_level/{}/edges.csv'.format(dataset)
    path_target = '../dataset/node_level/{}/target.csv'.format(dataset)

    # link predict
    test_percent = 0.3

    # parameters
    order = 100

    #result file path
    method = '{}-{}-Order3'.format(graph_type, path)
    path_result = '../result/{}_{}_{}_link.csv'.format(method, dataset, test_percent)


    X0 = sparse.coo_matrix(load_features(path_feature))
    AP = normalize_by_column(X0).T
    PA = normalize_by_row(X0)

    X00 = PA.dot(AP)
    #X00 = X0.dot(X0.T)
    X00 = sparse.coo_matrix((X00 + X00.T)) / 2
    X00 = normalize_by_row(X00)
    X00 = X00

    result_file= open(path_result, 'a', encoding='utf-8', newline='')
    writer = csv.writer(result_file)
    writer.writerow(['method', 'dataset', 'cnt', 'test_perc', 'auc', 'ap'])
    result_file.close()

    
    for c in range(cnt):
        
        # load graph edges
        edgelist = np.array(pd.read_csv(path_edge))
        # create graph from edges
        G1 = nx.from_edgelist(edgelist, create_using=nx.DiGraph)
        G1, map_idx_nodes = modify_nodes(G1)

        G1, positive_edge, negative_edge = get_test_edge(G1, test_percent)
        for j in range(G1.number_of_nodes()):
            G1.add_edge(j,j)

        A_tilde, D_inv = create_A_tilde(G1)
        A_pool, A_hat = A_tilde.copy(), A_tilde.copy()
        P = A_tilde.todense()
        P += X00
        X00 = sparse.coo_matrix(X00)
        PPMI_list = []
        for i in range(order):
            print('order', i)
            if path == 'ppmi' or path == 'random_walk_ppmi':
                PMI =  sparse.coo_matrix(G1.number_of_edges() * (A_tilde.dot(D_inv) + D_inv.dot(A_tilde.T)))
                PMI.data[PMI.data < 1.0] = 1.0
                PPMI = sparse.coo_matrix((np.log(PMI.data), (PMI.row, PMI.col)),shape=PMI.shape,dtype=np.float32)    
                PPMI_tilde = normalize_by_row(PPMI)
                PPMI_list.append(PPMI_tilde)
                tmp = sparse.coo_matrix(PPMI_tilde.dot(X00))
                P +=  tmp
            if path == 'random_walk' or path == 'random_walk_ppmi':
                tmp2 = sparse.coo_matrix(A_tilde.dot(X00))
                P +=  tmp2
            for j in range(i+1):
                if path == 'ppmi' or path == 'random_walk_ppmi':
                    P += tmp.dot(PPMI_list[j])
                if path == 'random_walk' or path == 'random_walk_ppmi':
                    tmp2 = sparse.coo_matrix(tmp2.dot(A_hat))
                    P +=  tmp2
            A_tilde = A_tilde.dot(A_hat)
            P += A_tilde
        Y = np.array(P)
        embedding = Y

        # link predict
        _, auc, ap = link_predict_without_lr(G1, embedding, positive_edge, negative_edge, test_percent)
        print('auc', auc)
        print('ap', ap)
        
        result_file= open(path_result, 'a', encoding='utf-8', newline='')
        writer = csv.writer(result_file)
        writer.writerow([method, dataset, str(c), str(test_percent), str(auc), str(ap)])
        result_file.close()

if __name__ == '__main__':
    main()
