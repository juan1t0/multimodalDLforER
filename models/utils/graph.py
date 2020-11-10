from typing import Tuple, List
from collections import defaultdict

from scipy import special
import numpy as np


def edge2mat(link, num_node):
	A = np.zeros((num_node, num_node))
	for i, j in link:
		A[j, i] = 1
	return A

def normalize_digraph(A):  # Divide by the sum of each column
	Dl = np.sum(A, 0)
	h, w = A.shape
	Dn = np.zeros((w, w))
	for i in range(w):
		if Dl[i] > 0:
			Dn[i, i] = Dl[i] ** (-1)
	AD = np.dot(A, Dn)
	return AD

def get_spatial_graph(num_node, self_link, inward, outward):
	I = edge2mat(self_link, num_node)
	In = normalize_digraph(edge2mat(inward, num_node))
	Out = normalize_digraph(edge2mat(outward, num_node))
	A = np.stack((I, In, Out))
	return A

##############################Open pose##############################

# Joint index:
# {0:  'Nose'}
# {1:  'Neck'},
# {2:  'RShoulder'},
# {3:  'RElbow'},
# {4:  'RWrist'},
# {5:  'LShoulder'},
# {6:  'LElbow'},
# {7:  'LWrist'},
# {8:  'Hip'},
# {9:  'RHip'},
# {10: 'RKnee'},
# {11: 'RAnkle'},
# {12: 'LHip'},
# {13: 'LKnee'},
# {14: 'LAnkle'},
# {15: 'REye'},
# {16: 'LEye'},
# {17: 'REar'},
# {18: 'LEar'}
# {19: 'LFoot1'}
# {20: 'LFoot2'}
# {21: 'LFoot3'}
# {22: 'RFoot1'}
# {23: 'RFoot2'}
# {24: 'RFoot3'}

##############################simple COCO##############################
# Joint index:
# {0:  'Nose'}
# {1:  'LShoulder'},
# {2:  'RShoulder'},
# {3:  'LElbow'},
# {4:  'RElbow'},
# {5:  'LWrist'},
# {6:  'RWrist'},
# {7:  'LHip'},
# {8:  'RHip'},
# {9:  'LKnee'},
# {10: 'RKnee'},
# {11: 'LAnkle'},
# {12: 'RAnkle'},
# {13: 'Chest'},
# {14: 'Hip'},
epsilon = 1e-6

num_nodes = 15 # 25
# directed_edges =[(0,1),(0,15),(0,16),(1,2),(1,5),
#     (1,8),(2,3),(3,4),(5,6),(6,7),
#     (8,9),(8,12),(9,10),(10,11),(12,13),
#     (13,14),(15,17),(16,18),(14,21),(14,19),
#     (19,20),(11,24),(11,22),(22,23),
# 		(1,1)]

directed_edges = [(1, 3), (3, 5), (2, 4), (4, 6),
                  (7, 9), (9, 11), (8, 10), (10, 12),
                  (13, 0), (13, 1), (13, 2), (13, 14),
                  (14, 8), (14, 7), (13, 13)]

# directed_edges = [(i-1, j-1) for i, j in [
# 	(1, 13), (1, 17), (2, 1), (3, 4), (5, 6),
# 	(6, 7), (7, 8), (8, 22), (8, 23), (9, 10),
# 	(10, 11), (11, 12), (12, 24), (12, 25), (13, 14),
# 	(14, 15), (15, 16), (17, 18), (18, 19), (19, 20),
# 	(21, 2), (21, 3), (21, 5), (21, 9),
# 	(21, 21)    # Add self loop for Node 21 (the centre) to avoid singular matrices
# ]]

'''
	change graph
	revisar eso de calculo de aristas
'''

# Note: for now, let's not add self loops since the paper didn't mention this
# self_loops = [(i, i) for i in range(num_nodes)]


def build_digraph_adj_list(edges: List[Tuple]) -> np.ndarray:
	graph = defaultdict(list)
	for source, target in edges:
		graph[source].append(target)
	return graph


def normalize_incidence_matrix(im: np.ndarray, full_im: np.ndarray) -> np.ndarray:
	# -The paper assumes that the Incidence matrix is square,
	#  so that the normalized form A @ (D ** -1) is viable.
	#  However, if the incidence matrix is non-square, then
	#  the above normalization won't work.
	#  For now, move the term (D ** -1) to the front
	# -It's not too clear whether the degree matrix of the FULL incidence matrix
	#  should be calculated, or just the target/source IMs.
	#  However, target/source IMs are SINGULAR matrices since not all nodes
	#  have incoming/outgoing edges, but the full IM as described by the paper
	#  is also singular, since ±1 is used for target/source nodes.
	#  For now, we'll stick with adding target/source IMs.
	degree_mat = full_im.sum(-1) * np.eye(len(full_im))
	# Since all nodes should have at least some edge, degree matrix is invertible
	inv_degree_mat = np.linalg.inv(degree_mat)
	return (inv_degree_mat @ im) + epsilon


def build_digraph_incidence_matrix(num_nodes: int, edges: List[Tuple]) -> np.ndarray:
	# Note: For now, we won't consider all possible edges
	# max_edges = int(special.comb(num_nodes, 2))
	max_edges = len(edges)
	source_graph = np.zeros((num_nodes, max_edges), dtype='float32')
	target_graph = np.zeros((num_nodes, max_edges), dtype='float32')
	for edge_id, (source_node, target_node) in enumerate(edges):
		source_graph[source_node, edge_id] = 1.
		target_graph[target_node, edge_id] = 1.
	full_graph = source_graph + target_graph
	source_graph = normalize_incidence_matrix(source_graph, full_graph)
	target_graph = normalize_incidence_matrix(target_graph, full_graph)
	return source_graph, target_graph


def build_digraph_adj_matrix(edges: List[Tuple]) -> np.ndarray:
	graph = np.zeros((num_nodes, num_nodes), dtype='float32')
	for edge in edges:
		graph[edge] = 1
	return graph


class Graph:
	def __init__(self):
		super().__init__()
		self.num_nodes = num_nodes
		self.edges = directed_edges
		# Incidence matrices
		self.source_M, self.target_M = \
			build_digraph_incidence_matrix(self.num_nodes, self.edges)


# Check whether self loop should be added inside the graph
# Check incidence matrix size


# if __name__ == "__main__":
# 	import matplotlib.pyplot as plt
# 	graph = Graph()
# 	source_M = graph.source_M
# 	target_M = graph.target_M
# 	plt.imshow(source_M, cmap='gray')
# 	plt.show()
# 	plt.imshow(target_M, cmap='gray')
# 	plt.show()
# 	print(source_M)