import sys
import os
import networkx as nx
from codes.hierarchical_node.node import Node


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(FILE_DIR, '..'))

total_levels = 1
all_leaves = set()

def dfs(u, tree, level):
    global total_levels,all_leaves
    tree[u].level = level
    tree[u].path.append(u)
    if len(tree[u].direct_children) == 0:   # leaf nodes
        if level > total_levels:
            total_levels = level
        all_leaves.add(u)
        tree[u].all_children = set([])
        return
    for v in tree[u].direct_children:
        tree[v].path.extend(tree[u].path)
        dfs(v, tree, level + 1)
        tree[u].all_children = tree[u].all_children | tree[v].all_children

def prepare_data(file_path):
    # path = os.path.join(file_path)
    g, n, m = build_hierarchical_tree(os.path.join(file_path))

    # n:num of total nodes; m:num of leaf nodes;
    tree = [None] * n
    for each in g:
        # init each node
        tree[each] = Node(each, set(g[each].keys()), set(), [], 1)
    level = 0
    dfs(n-1,tree,level + 1)
    ret = (tree,total_levels,all_leaves)
    return ret

def build_hierarchical_tree(file_path):
    G = nx.DiGraph()
    n, m = None, None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            items = line.split()
            if len(items) != 2:
                continue
            if n is None:
                n, m = int(items[0]), int(items[1])
            else:
                G.add_edge(int(items[0]), int(items[1]))
    return G,n,m


