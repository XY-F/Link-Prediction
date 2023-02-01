import numpy as np
import networkx as nx
import scipy.sparse as sps
import pandas as pd
import random 
from utils import create_A_tilde, normalize_by_row, self_cosine_similarity, save_embedding
from karateclub import FSCNMF, TADW, TENE, BANE, MUSAE, FeatherNode, DeepWalk, SINE


class CN():
	def __init__(self, graph):
		self.graph = graph

	def get_scores_matrix(self):
		self.S = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes())).toarray()
		self.S = np.asarray(self.S.dot(self.S.T))

	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2])
		return X_score

class CNA():
	def __init__(self, graph, attribute):
		self.graph = graph
		self.attribute = attribute

	def get_scores_matrix(self):
		A = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes()))
		self.S = sps.coo_matrix(A.dot(A.T))
		self.S = normalize_by_row(self.S)
		self.S += self_cosine_similarity(self.attribute)
		self.S = np.asarray(self.S)

	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2])
		return X_score

class ATT():
	def __init__(self, graph, attribute):
		self.graph = graph
		self.attribute = attribute

	def get_scores_matrix(self):
		self.S = self_cosine_similarity(self.attribute)

	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2])
		return X_score


class JC():
	def __init__(self, graph):
		self.graph = graph

	def get_scores_matrix(self):
		self.S = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes())).toarray()

	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			num = np.sum(self.S[edge1] * self.S[edge2])
			denum = np.sum(np.maximum(1, np.minimum(self.S[edge1] + self.S[edge2], 1)))
			X_score.append(num / denum)
		return X_score


class JCA():
	def __init__(self, graph, attribute):
		self.graph = graph
		self.attribute = attribute

	def get_scores_matrix(self):
		self.S1 = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes())).toarray()
		self.S2 = self_cosine_similarity(self.attribute)

	def get_scores(self, edges):
		
		structure_score = []
		attribute_score = []
		
		for edge1, edge2 in edges:
			num = np.sum(self.S1[edge1] * self.S1[edge2])
			denum = np.sum(np.maximum(1, np.minimum(self.S1[edge1] + self.S1[edge2], 1)))
			structure_score.append(num / denum)
		structure_score = np.asarray(structure_score)
		structure_score = (structure_score - np.min(structure_score)) / np.max(structure_score)
		
		for edge1, edge2 in edges:
			attribute_score.append(self.S2[edge1][edge2])
		attribute_score = np.asarray(attribute_score)
		attribute_score = (attribute_score - np.min(attribute_score)) / np.max(attribute_score)

		X_score = structure_score + attribute_score
		return X_score


class RWR():
	def __init__(self, graph, alpha=0.5):
		self.graph = graph
		self.alpha = alpha

	def get_scores_matrix(self):
		A_tilde, D_inv  = create_A_tilde(self.graph)
		self.S = (1-self.alpha) * np.linalg.inv((np.eye(A_tilde.shape[0]) - self.alpha * A_tilde.T))
		self.S = np.asarray(self.S)
		
	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2]) 
		return X_score

class RWRA():
	def __init__(self, graph, attribute, alpha=0.5):
		self.graph = graph
		self.attribute = attribute
		self.alpha = alpha

	def get_scores_matrix(self):
		A_tilde, D_inv  = create_A_tilde(self.graph)
		self.S1 = (1-self.alpha) * np.linalg.inv((np.eye(A_tilde.shape[0]) - self.alpha * A_tilde.T))
		self.S1 = np.asarray(self.S1)
		self.S2 = self_cosine_similarity(self.attribute)
		
	def get_scores(self, edges):
		structure_score = []
		attribute_score = []

		for edge1, edge2 in edges:
			structure_score.append(self.S1[edge1][edge2]) 
		structure_score = np.asarray(structure_score)
		structure_score = (structure_score - np.min(structure_score)) / np.max(structure_score)
		
		for edge1, edge2 in edges:
			attribute_score.append(self.S2[edge1][edge2])
		attribute_score = np.asarray(attribute_score)
		attribute_score = (attribute_score - np.min(attribute_score)) / np.max(attribute_score)

		X_score = structure_score + attribute_score
		return X_score

class KIA():
	def __init__(self, graph, attribute):
		self.graph = graph
		self.attribute = attribute

	def get_scores_matrix(self):
		A = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes()))
		A = A.asfptype() 
		value, _ = sps.linalg.eigsh(A, k=1, which='LM')
		A = A.toarray()
		print('lambda', value[0])
		if value[0] > 0:
			if (1 / value[0] - 0.3) > 0:
				beta = 1 / value[0] - 0.3
			elif (1 / value[0] - 0.03) > 0:
				beta = 1 / value[0] - 0.03
			else:
				beta = 1 / value[0] - 0.003
		else:
			print('lambda < 0')
		I = np.eye(A.shape[0])
		self.S1 = np.linalg.inv(I - beta * A) - I
		self.S2 = self_cosine_similarity(self.attribute)
		
	def get_scores(self,  edges):
		structure_score = []
		attribute_score = []

		for edge1, edge2 in edges:
			structure_score.append(self.S1[edge1][edge2]) 
		structure_score = np.asarray(structure_score)
		structure_score = (structure_score - np.min(structure_score)) / np.max(structure_score)
		
		for edge1, edge2 in edges:
			attribute_score.append(self.S2[edge1][edge2])
		attribute_score = np.asarray(attribute_score)
		attribute_score = (attribute_score - np.min(attribute_score)) / np.max(attribute_score)

		X_score = structure_score + attribute_score
		return X_score

class KI():
	def __init__(self, graph):
		self.graph = graph

	def get_scores_matrix(self):
		A = nx.adjacency_matrix(self.graph, nodelist = range(self.graph.number_of_nodes()))
		A = A.asfptype() 
		value, _ = sps.linalg.eigsh(A, k=1, which='LM')
		A = A.toarray()
		print('lambda', value[0])
		if value[0] > 0:
			if (1 / value[0] - 0.1) > 0:
				beta = 1 / value[0] - 0.1
			elif (1 / value[0] - 0.01) > 0:
				beta = 1 / value[0] - 0.01
			else:
				beta = 1 / value[0] - 0.001
		else:
			print('lambda < 0')
		I = np.eye(A.shape[0])
		self.S = np.linalg.inv(I - beta * A) - I

	def get_scores(self, edges):
		X_score = []
		for edge1, edge2 in edges:
			X_score.append(self.S[edge1][edge2]) 
		return X_score


class EMB():
	def __init__(self, graph, attribute, dataset, method, edge_feature, test_percent):
		self.graph = graph
		self.attribute = attribute
		self.dataset = dataset
		self.method = method
		self.edge_feature = edge_feature
		self.test_percent = test_percent

	def get_scores_matrix(self):
		if self.method == 'FeatherNode':
			model = FeatherNode(reduction_dimensions=32, eval_points=16, order=2)
		elif self.method == 'SINE':
			model = SINE(dimensions=256)
		elif self.method == 'MUSAE':
			model = MUSAE(dimensions=128)
		elif self.method == 'TADW':
			model = TADW(dimensions=128)
		elif self.method == 'DeepWalk':
			model = DeepWalk(dimensions=256)

		path_emb = '../output/{}_{}_{}_{}.csv'.format(self.dataset, self.method, self.test_percent, random.randint(1,10000))
		if self.method == 'DeepWalk':
			model.fit(self.graph)
		elif self.method == 'SINE' or 'MUSAE':
			model.fit(self.graph, sps.coo_matrix(self.attribute))
		else:
			model.fit(self.graph, self.attribute)

		X = model.get_embedding()
		print('X.shape', X.shape)

		save_embedding(X, path_emb)
		
		if self.edge_feature == 'cosine_similarity':
			denum = np.linalg.norm(X, axis=-1)
			denum[np.where(denum == 0)] = 1.0
			X = X / denum[:,None]
			self.S = X.dot(X.T)
		elif self.edge_feature == 'hadamard':
			self.S = X

	def get_scores(self, edges):
		X_score = []
		if self.edge_feature == 'cosine_similarity':
			for edge1, edge2 in edges:
				X_score.append(self.S[edge1][edge2])
		elif self.edge_feature == 'hadamard':
			for edge1, edge2 in edges:
				X_score.append(self.S[edge1] * self.S[edge2])
		return X_score
