import pandas as pd
import numpy as np
import networkx as nx
from codes.utils.filewriter import write_to_file

file = '../data/mammal_closure.csv'
# root_name = "entity.n.01"
root_name = "mammal.n.01"
# name = 'mammal'
# name = 'wordnet'
G = nx.DiGraph()
f = pd.read_csv(file, header=0, sep=',')
d = pd.DataFrame()
d['parent'] = f['id2']
d['child'] = f['id1']
d['w'] = f['weight']
d = d[~(d['parent'] == d['child'])]

distinctLeaves = []
distinctNodeSet = set()
for index,row in d.iterrows():
    child, parent = row['child'], row['parent']
    if child in distinctNodeSet:
        continue
    distinctNodeSet.add(child)
    distinctLeaves.append((parent,child,1))


G.add_weighted_edges_from(distinctLeaves)
leafs = []
no_leafs = []
for node in G.node:
    # our_degree is zero, means it's a leaf
    outdegree = G.out_degree(node)
    if outdegree == 0:
        leafs.append(node)
    elif node != root_name:
        no_leafs.append(node)

i = 0
node_dict = dict()
for node in leafs:
    node_dict[node] = i
    i = i+1
for node in no_leafs:
    node_dict[node] = i
    i=i+1
node_dict[root_name] = i
no_leafs.append(root_name)

e1 = [len(leafs)+len(no_leafs)]
e2 = [len(leafs)]
for edge in G.edges():
    e1.append(node_dict[edge[0]])
    e2.append(node_dict[edge[1]])

d = pd.DataFrame()
d['p'] = e1
d['c'] = e2
# d.to_csv('./data/tree2_wordnet', header=False, index=False, sep='\t')
d.to_csv('../data/tree2_mammal', header=False, index=False, sep='\t')



tl=[]
for l in leafs:
    tl.append(node_dict[l])
tl.sort()

d = pd.DataFrame()
d['p'] = tl
d['c'] = tl
# d.to_csv('./data/edges_wordnet.txt', header=False, index=False, sep='\t')
d.to_csv('../data/edges_mammal.txt', header=False, index=False, sep='\t')


mammal_index = '../data/mammal_index_label.txt'
for (k,v) in node_dict.items():
    write_to_file(mammal_index,str(v) + "\t" + k + "\r\n")








