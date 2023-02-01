import numpy as np
import pandas as pd
import networkx as nx
from utils import normalize_by_row, normalize_by_column, create_D_inverse, create_A_tilde
from scipy import sparse
import csv


class MINE():
    def __init__(self, graph, attribute, order=1, version=1):
        self.graph = graph
        self.attribute = attribute
        self.order = order
        self.version = version

    def get_scores_matrix(self): 
        if self.version == 1:
            X = sparse.coo_matrix(self.attribute)
            XA = normalize_by_column(X).T
            AX = normalize_by_row(X)
            AXA = sparse.coo_matrix(AX.dot(XA))
            
        elif self.version == 2:
            X = self.attribute
            denum = np.linalg.norm(X, axis=-1)
            denum[np.where(denum == 0)] = 1.0
            X = X / denum[:,None]
            AXA = sparse.coo_matrix(X.dot(X.T))
            
        AXA = normalize_by_row(AXA)
        
        A_tilde, D_inv = create_A_tilde(self.graph)
        A_pool, A_hat = A_tilde.copy(), A_tilde.copy()
        
        S = A_tilde
        S += AXA
        for i in range(self.order):
            print('order', i)
            tmp1 = sparse.coo_matrix(A_tilde.dot(AXA))
            S += tmp1
            tmp2 = sparse.coo_matrix(AXA.dot(A_tilde))
            #S += tmp2
            for j in range(i+1):
                tmp1 = sparse.coo_matrix(tmp1.dot(A_hat))
                S += tmp1
                tmp2 = sparse.coo_matrix(A_hat.dot(tmp2))
                S += tmp2
            A_tilde = sparse.coo_matrix(A_tilde.dot(A_hat))
            S += A_tilde
        self.S = np.asarray(S.dot(S).todense())
        

    def get_scores(self, edges):
        X_score = []
        for edge1, edge2 in edges:
            X_score.append(self.S[edge1][edge2])
        X_score = np.asarray(X_score)
        return X_score
  
















