import networkx as nx

def mine(embedding, train_pos_edge, train_neg_edge):
	X_train = []
	for edge1, edge2 in train_pos_edge:
		X_train.append(embedding[edge1][edge2])
		#X_train.append([embedding[edge1][edge2]])
	for edge1, edge2 in train_neg_edge:
		X_train.append(embedding[edge1][edge2])
		#X_train.append([embedding[edge1][edge2]])
	
	return np.array(X_train)


def CN(embedding, train_pos_edge, train_neg_edge):
	X_train = []
	for edge1, edge2 in train_pos_edge:
		X_train.append(embedding[edge1] * embedding[edge2])
	for edge1, edge2 in train_neg_edge:
		X_train.append(embedding[edge1] * embedding[edge2])

	return np.array(X_train)




