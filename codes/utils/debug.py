import pandas as pd
import numpy as np
import networkx as nx
# files = ['../poincare-embeddings/wordnet/noun_closure.csv', '../poincare-embeddings/wordnet/noun_label.csv']
files = ['../data/mammal_closure.csv', '../poincare-embeddings/wordnet/mammal_label.csv']
# root_name = "entity.n.01"
root_name = "mammal.n.01"
# name = 'mammal'
# name = 'wordnet'
G = nx.DiGraph()
f = pd.read_csv(files[0], header=0, sep=',')
d = pd.DataFrame()
d['parent'] = f['id2']
d['child'] = f['id1']
d['w'] = f['weight']
d = d[~(d['parent'] == d['child'])]
print(d)