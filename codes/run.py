#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import math
import os
import pprint
import time

import torch

from codes.network import NetworkModel
from codes.tree import TreeModel
from codes.utils.datahandler import DatasetH, networkDataset, networkLeavesDataset
from codes.utils.datahandler import DatasetV
from codes.utils.datahandler import BidirectionalOneShotIterator
import codes.utils.data_etl as etl
from torch.utils.data import DataLoader
from codes.utils.filewriter import write_to_file
from codes.model import ASModel
import numpy as np

def parse_args(args=None):
    """
    参数入口
    :param args:
    :return:
    """
    parser = argparse.ArgumentParser(
        description='Train the Archimedean Spiral Model'
    )

    parser.add_argument('--usecuda', action='store_true', help='use GPU')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--network_path', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-scr', '--single_circle_range', default=2 * math.pi * 10, type=float)

    parser.add_argument('-n', '--negative_sample_size', default=1, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('-d', '--hidden_dim', default=32, type=int)
    parser.add_argument('--cpu_num', type=int, default=1)
    parser.add_argument('-cm','--circle_margin', type=float)    # 每圈的边界

    parser.add_argument('-b', '--batch_size', default=500, type=int)
    parser.add_argument('--res_path', type=str)

    return parser.parse_args(args)




def train_dfs(curNode, res, args, tree, leavesMatrix,device):
    # children, commonSimMatrix = etl.get_branch_common_similarity_matrix(curNode, tree, leavesMatrix)
    children, simMartrix = etl.get_nodes_sim_based_on_matrix(curNode, tree, leavesMatrix, 100)
    # commonSimMatrixNorm = commonSimMatrix
    childLeavesCnt = []
    level = tree[curNode].level
    for child in children:
        childLeavesCnt.append(len(tree[child].leaves))

    if len(children) == 1:
        res[children[0]] = res[curNode] + args.single_circle_range
    else:

        simMatrixNorm = etl.normalize_adj_matrix(simMartrix)

        # init the network model
        networkModel = NetworkModel(
            children = children,
            args=args
        )


        # load the network training dataset
        networkDataLoader = DataLoader(
            networkDataset(curNode, tree, children, simMatrixNorm, args),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            # num_workers=1,
            # collate_fn=networkDataset.collate_fn,
            collate_fn=lambda x:networkDataset.collate_fn(x,args.batch_size),
            drop_last=False
        )
        trainIterator1 = BidirectionalOneShotIterator(networkDataLoader)

        learningRate1 = args.learning_rate
        optimizer1 = torch.optim.Adam(
            filter(lambda p: p.requires_grad, networkModel.parameters()),
            lr=learningRate1
        )

        preLoss = float('inf')
        for step in range(0, args.max_steps):
            loss,embeddingOmega = NetworkModel.train_step(networkModel, optimizer1, trainIterator1)
            if step % 100 ==0 :
                lossNumeric = loss.item()
                print("Network node:%d, iterator is %d, loss is:%f" % ( curNode, step, lossNumeric))
                if abs(lossNumeric - preLoss) < 0.01:
                    break
                else:
                    preLoss = lossNumeric
        print("Start training the tree......")
        # project the embedding to the spiral
        # init the tree model
        # print(embeddingOmega.data.numpy())
        treeModel = TreeModel(
            omega=embeddingOmega.data.numpy(),
            parent=curNode,
            children= children,
            res = res,
            args=args,
            leavesCnt=childLeavesCnt,
            device=device
        )
        treeModel.to(device)
        learningRate2 = args.learning_rate
        optimizer2 = torch.optim.Adam(
            filter(lambda p: p.requires_grad, treeModel.parameters()),
            lr=0.005
        )

        treePreLoss = float('inf')
        for step in range(0, args.max_steps):
            treeLossNumeric,embedding = treeModel.train_step(treeModel, optimizer2,step)
            embTmpRes = embedding.data


            # if torch.isnan(treeLossNumeric):


            if step % 100 == 0 :

                print("Tree node:%d, iterator is %d, loss is:%f" % ( curNode, step, treePreLoss))
                # pprint.pprint(level)
                if abs(treeLossNumeric - treePreLoss) < (level**4) * 0.0005:

                    for k in children:
                        pprint.pprint(str(k) + '    ' + str(len(tree[k].leaves)))
                        # pprint.pprint(len(tree[k].leaves))
                    # pprint.pprint(children)

                    pprint.pprint(embTmpRes.numpy())
                    l, h = torch.split(embTmpRes, args.single_dim, dim=1)
                    pprint.pprint(np.around((h-l).numpy(),decimals=2))

                    for indexer in range(len(children)):
                        child = children[indexer]
                        res[child] = embTmpRes[indexer]
                    break
                else:
                    treePreLoss = treeLossNumeric

            if step == args.max_steps -1 :
                print('final')
                pprint.pprint(embTmpRes.numpy())
                l, h = torch.split(embTmpRes, args.single_dim, dim=1)
                pprint.pprint(np.around((h - l).numpy(), decimals=2))
                for indexer in range(len(children)):
                    child = children[indexer]
                    res[child] = embTmpRes[indexer]
    # exit(1)
    for child in children:
        if tree[child].direct_children:
            train_dfs(child, res, args, tree, leavesMatrix)

def main(args):
    """
    主函数
    :param args:
    :return:
    """
    if (not args.do_train):
        raise ValueError('error.')
    if (args.hidden_dim % 2 != 0):
        raise ValueError('hidden_error')
    args.single_dim = args.hidden_dim // 2

    if args.usecuda:
        device = "cuda:2" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    '''
    prepare the dataset:
    tree: all of the nodes in tree including leaves and none-leaf node, each node object is in format of :
        [
            all_children: a set of all children of the current node,
            direct_children: a set of one-hop children of the current node,
            id: the id of the current node,
            level: the depth of the current node in the tree,
            path: a list of nodes in the order of from the root to the current node
        ]
    total_level: the height of the tree,
    all_leaves: all the leaves in the tree    
    '''

    tree, total_level,all_leaves = etl.prepare_tree(args.data_path)

    graph = etl.prepare_graph(args.network_path)

    # specify the root node
    root = len(tree) - 1
    root_embedding_lower = torch.zeros(1, args.single_dim)
    root_embedding_upper = args.single_circle_range * torch.ones(1, args.single_dim)
    root_embedding = torch.cat((root_embedding_lower, root_embedding_upper), 1)[0]

    # subnodes_embedding =

    res = torch.zeros(len(tree),args.hidden_dim)

    res[root] = root_embedding

    # calc the graph similarity
    leaves_similarity = etl.get_leaves_similarity(graph)

    train_dfs(root,res, args, tree, leaves_similarity,device)

    res_output = os.path.join(args.res_path, "res_"+str(int(time.time())))


    write_to_file(res_output, json.dumps(res.numpy().tolist()))



    # whole_tree_similarity = etl.get_tree_based_similarity(tree, leaves_similarity )
    #
    # # init the model
    # model = ASModel(
    #     tree = tree,
    #     graph = graph,
    #     tree_height=total_level,
    #     all_leaves = all_leaves,
    #     args = args
    # )
    #
    # if args.cuda:
    #     model = model.cuda()
    #
    # if args.do_train:
    #     # Set training dataloader iterator of the horizontal(global) direction.
    #     train_dataloader_h = DataLoader(
    #         DatasetH(all_leaves,tree,graph, args,total_level),
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         num_workers=max(1, args.cpu_num // 2),
    #         # num_workers=1,
    #         collate_fn=DatasetH.collate_fn
    #     )
    #     train_iterator_h = BidirectionalOneShotIterator(train_dataloader_h)
    #
    #     # Set training dataloader iterator of the vertical(local) direction.
    #     train_dataloader_v = DataLoader(
    #         DatasetV(all_leaves, tree, args, total_level),
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         num_workers=max(1, args.cpu_num // 2),
    #         # num_workers=1,
    #         collate_fn=DatasetV.collate_fn
    #     )
    #     train_iterator_v = BidirectionalOneShotIterator(train_dataloader_v)
    #
    #     # Define the optimizer
    #     current_learning_rate = args.learning_rate
    #     optimizer = torch.optim.Adam(
    #         filter(lambda p: p.requires_grad, model.parameters()),
    #         lr=current_learning_rate
    #     )
    #
    #     if args.warm_up_steps:
    #         warm_up_steps = args.warm_up_steps
    #     else:
    #         warm_up_steps = args.max_steps // 2
    #
    #     init_step = 0
    #
    #     logging.info('learning_rate = %d' % current_learning_rate)
    #
    #     training_logs = []
    #
    #     # Training Loop
    #     for step in range(init_step, args.max_steps):
    #
    #         log = model.train_step(model, optimizer,train_iterator_v, train_iterator_h)
    #         print(log)
    #         training_logs.append(log)
    #
    #         if step >= warm_up_steps:
    #             current_learning_rate = current_learning_rate / 10
    #             logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
    #             optimizer = torch.optim.Adam(
    #                 filter(lambda p: p.requires_grad, model.parameters()),
    #                 lr=current_learning_rate
    #             )
    #             warm_up_steps = warm_up_steps * 3


if __name__ == '__main__':
    main(parse_args())