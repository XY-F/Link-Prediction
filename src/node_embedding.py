import networkx as nx
import scipy.sparse as sps
from utils import load_graph, load_features, save_embedding, 
from karateclub import FSCNMF, TADW, TENE, BANE, MUSAE, FeatherNode


#dataset = 'citeseer'
#dataset = 'cora'
#dataset = 'pubmed'
#dataset = 'facebook'
#dataset = 'wikipedia'
dataset = 'cornell'
#graph_type = 'direct'
graph_type = 'undirect'
#method = 'FeatherNode'
method = 'MUSAE'
#method = 'BANE'
#method = 'TADW'
#method = 'FSCNMF'

test_prec = 0.1

path_graph = '../dataset/node_level/{}/edges.csv'.format(dataset)
path_feature = '../dataset/node_level/{}/features.csv'.format(dataset)
path_embedding = '../output/{}_{}_{}.csv'.format(dataset, method, str(test_perc))

graph, map_idx_nodes = load_graph(path_graph, graph_type)


feature = sps.coo_matrix(load_features(path_feature))

if method == 'FeatherNode':
	model = FeatherNode(reduction_dimensions=32, eval_points=16, order=2)
elif method == 'MUSAE':
	model = MUSAE(dimensions=256)
elif method == 'BANE':
	model = BANE(dimensions=256)
elif method == 'TENE':
	model = TENE(dimensions=128)
elif method == 'TADW':
	model = TADW(dimensions=128)
elif method =='FSCNMF':
	model = FSCNMF(dimensions=128)

model.fit(graph, feature)
X = model.get_embedding()
print('X.shape', X.shape)
print('X', X)

save_embedding(X, path_embedding, map_idx_nodes)
