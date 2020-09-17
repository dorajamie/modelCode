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
import sys
import torch
from tqdm import tqdm

ROOT_DIR='/home/lduan/modelCode'
sys.path.append(ROOT_DIR)
from codes.network import NetworkModel
from codes.tree import TreeModel
from codes.hierarchy import HierarchyModel
from codes.utils.datahandler import networkDataset,treeDataset

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
    3-32
    4-16
    5-64
    6-128
    """
    parser = argparse.ArgumentParser(
        description='Train the Archimedean Spiral Model'
    )

    parser.add_argument('--usecuda', action='store_true', help='use GPU')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--network_path', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--max_epoch', type=int, default=50000)
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-scr', '--single_circle_range', default=2 * math.pi * 10, type=float)


    parser.add_argument('-dn', '--hidden_dim_n', default=16, type=int)
    parser.add_argument('-dt', '--hidden_dim_t', default=2, type=int)

    parser.add_argument('--cpu_num', type=int, default=1)
    parser.add_argument('-cm','--circle_margin', type=float)    # 每圈的边界

    parser.add_argument('-b', '--batch_size', default=500, type=int)
    parser.add_argument('--res_path', type=str)

    parser.add_argument('--loss_distance', default=0.0001, type=float)
    parser.add_argument('--loss_shape', default=0.0001, type=float)
    parser.add_argument('--loss_overlap', default=0.0001, type=float)
    parser.add_argument('--loss_exceed', default=0.0001, type=float)
    parser.add_argument('--loss_positive', default=0.0001, type=float)



    return parser.parse_args(args)



def layerWiseTraining(curLayer, res, args, tree, leavesMatrix, device, layerCounter, parentDict, layerBasedDict):
    """
    Layer by layer training function.
    :param curLayer:  current layer that to be trained
    :param res:         the final results
    :param args:        the arguments in script
    :param tree:        the tree structure
    :param leavesMatrix:    the similarity matrix among leaves
    :param device:      the training device
    :param layerCounter:    a dict that store the list of nodes of each layer
    :return:
    """
    # Return the leaf level and needn't process.
    if curLayer > len(layerCounter) - 2 :
        return
    print("Start training the network:"+str(curLayer)+"......")
    parentList = layerCounter[curLayer]
    childrenList = layerCounter[curLayer + 1]

    # Calc the ground-truth of this layer, the simMatrix is in the order of childrenList.
    simMatrix = etl.getLayerNodesSimBasedOnLeavesSim(leavesMatrix, childrenList, tree, 100)
    simMatrixNorm = etl.normalizeMatrix(simMatrix)
    # print('init network')
    # Initialize the network model.
    networkModel = NetworkModel(
        children=childrenList,
        args=args
    )
    # print('init dataset')
    # Load the network training dataset.
    networkDataLoader = DataLoader(
        networkDataset(simMatrixNorm, args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=lambda x: networkDataset.collate_fn(x, args.batch_size),
        drop_last=False
    )
    # The network iterator.
    networkTrainingIterator = BidirectionalOneShotIterator(networkDataLoader)

    networkLearningRate = args.learning_rate
    networkOptimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, networkModel.parameters()),
        lr=networkLearningRate
    )
    # Start training the network.
    preLoss = float('inf')
    for step in range(0, args.max_steps):
        loss, embeddingOmega = NetworkModel.train_step(networkModel, networkOptimizer, networkTrainingIterator)
        if step % 100 == 0:
            lossNumeric = loss.item()
            print("Network layer:%d, iterator is %d, loss is:%f" % (curLayer, step, lossNumeric))
            if abs(lossNumeric - preLoss) < 1:
                break
            else:
                preLoss = lossNumeric

    # Start training the tree.
    print("Start training the tree:" + str(curLayer) + "......")
    # The medium omega is in the order of children list.
    omega = embeddingOmega.data.numpy()

    treeModel = HierarchyModel(
        layer = curLayer,
        omega=omega,
        res=res,
        args=args,
        childrenList=childrenList,
        parentsList=parentList,
        device=device,
        parentDict = parentDict,
        layerBasedRes = layerBasedDict,
        tree = tree
    )

    treeModel.to(device)
    treeModel.train()

    # Load the tree training dataset.
    treeDataLoader = DataLoader(
        treeDataset(omega, args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=lambda x: treeDataset.collate_fn(x, args.batch_size),
        drop_last=False
    )

    treeLearningRate = args.learning_rate
    treeOptimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, treeModel.parameters()),
        lr=treeLearningRate
    )

    # Start training the tree in epochs.
    treePreLoss = float('inf')

    for epoch in range(0,  args.max_epoch):
        for i, data in enumerate(treeDataLoader):
            idx = data[0].to(device)
            omega = data[1].to(device)
            # data = data.to(device)
            # treeLossNumeric, embedding = treeModel.trainStep(treeModel, treeOptimizer, treeTrainingIterator, step)
            treeOptimizer.zero_grad()
            treeLoss = treeModel(idx, omega, epoch)
            treeLoss.backward()
            treeOptimizer.step()
            # for name, parms in treeModel.named_parameters():
            #     print('-->name:', name, '-->grad_requires:', parms.requires_grad, \
            #       ' -->grad_value:', parms.grad)

        if epoch % 100 == 0:
            loss = treeLoss.item()

            print("Tree layer:%d, epoch is %d, loss is:%f" % (curLayer, epoch, loss))
            if abs(loss - treePreLoss) < 0.001:
                embTmpRes = treeModel.childrenEmbedding.data.cpu()
                for k in childrenList:
                    pprint.pprint(str(k) + '    ' + str(len(tree[k].leaves)))
                pprint.pprint(embTmpRes.numpy())
                l, h = torch.split(embTmpRes, args.single_dim_t, dim=1)
                pprint.pprint(np.around((h - l).numpy(), decimals=6))
                for indexer in range(len(childrenList)):
                    child = childrenList[indexer]
                    res[child] = embTmpRes[indexer]
                break
            else:
                # if abs(loss - treePreLoss) < 0.01:
                # for name, parms in t reeModel.named_parameters():
                #     # print('-->name:', name, '-->grad_requires:', parms.requires_grad, \
                #     #       ' -->grad_value:', parms.grad)
                #     print('grad:')
                #     print(parms.grad)
                treePreLoss = loss




        if epoch == args.max_epoch - 1:
            # embTmpRes = treeModel.childrenEmbedding.data.cpu()
            embTmpRes = treeModel.childrenEmbedding.data.cpu()
            for k in childrenList:
                pprint.pprint(str(k) + '    ' + str(len(tree[k].leaves)))
            pprint.pprint(embTmpRes.numpy())
            l, h = torch.split(embTmpRes, args.single_dim_t, dim=1)
            pprint.pprint(np.around((h - l).numpy(), decimals=6))
            for indexer in range(len(childrenList)):
                child = childrenList[indexer]
                res[child] = embTmpRes[indexer]

    # Start training the tree.
    # treePreLoss = float('inf')
    # for step in range(0, args.max_steps):
    #
    #     treeLossNumeric, embedding = treeModel.trainStep(treeModel, treeOptimizer, treeTrainingIterator, step)
    #
    #
    #     if step % 100 == 0:
    #         print("Tree layer:%d, iterator is %d, loss is:%f" % (curLayer, step, treePreLoss))
    #         embTmpRes = embedding.data.cpu()
    #         if abs(treeLossNumeric - treePreLoss) < ((curLayer + 1)) * 0.00003   :
    #
    #             for k in childrenList:
    #                 pprint.pprint(str(k) + '    ' + str(len(tree[k].leaves)))
    #
    #             pprint.pprint(embTmpRes.numpy())
    #             l, h = torch.split(embTmpRes, args.single_dim_t, dim=1)
    #             pprint.pprint(np.around((h - l).numpy(), decimals=4))
    #
    #             for indexer in range(len(childrenList)):
    #                 child = childrenList[indexer]
    #                 res[child] = embTmpRes[indexer]
    #             break
    #         else:
    #             treePreLoss = treeLossNumeric
    #     if step == args.max_steps - 1:
    #         embTmpRes = embedding.data.cpu()
    #         for indexer in range(len(childrenList)):
    #             child = childrenList[indexer]
    #             res[child] = embTmpRes[indexer]



    layerWiseTraining(curLayer+1, res, args, tree, leavesMatrix, device, layerCounter, parentDict, layerBasedDict)

def main(args):
    """
    training entrance
    :param args:
    :return:
    """
    # Parameters verifying.
    if (not args.do_train):
        raise ValueError('error.')
    if (args.hidden_dim_t % 2 != 0):
        raise ValueError('hidden_error')
    args.single_dim_t = args.hidden_dim_t // 2

    # Select a device, cpu or gpu.
    if args.usecuda:
        devicePU = "cuda:3" if torch.cuda.is_available() else "cpu"
    else:
        devicePU = "cpu"
    device = torch.device(devicePU)

    # Load the tree and some properties of the tree.
    tree, total_level, all_leaves = etl.prepare_tree(args.data_path)
    # load the graph
    graph = etl.prepare_graph(args.network_path)
    # Define the root node
    root = len(tree) - 1
    # Calc the graph similarity, i.e. the matrix \capA in paper.
    leavesSimilarity = etl.get_leaves_similarity(graph)

    # Initialize the result and fix the root node's embedding.
    root_embedding_lower = torch.zeros(1, args.single_dim_t)
    root_embedding_upper = args.single_circle_range * torch.ones(1, args.single_dim_t)
    root_embedding = torch.cat((root_embedding_lower, root_embedding_upper), 1)[0]
    # print(root_embedding)
    res = torch.zeros(len(tree), args.hidden_dim_t)
    res[root] = root_embedding

    # Initialize the layer dict containing lists of nodes of each layer.
    layerCounter = [[] for i in range(total_level)]
    parentDict = {}
    for node in tree:
        if node.id != root:
            parentDict[node.id] = node.path[-2]
        layerCounter[node.level - 1].append(node.id)

    # Initialize the layerBasedDistance dict.
    layerBasedDict = {}

    # Train HASNE layer by layer, start from the 0(which the root locate in) layer.
    layerWiseTraining(0, res, args, tree, leavesSimilarity, device, layerCounter, parentDict, layerBasedDict)


    res_output = os.path.join(args.res_path, "res_"+str(int(time.time())))

    final_res = {
        'learningRate':args.learning_rate,
        'batchSize':args.batch_size,
        'networkDims':args.hidden_dim_n,
        'treeDims':args.hidden_dim_t,
        'circleRange':args.single_circle_range,
        'epoch':args.max_epoch,
        'loss_distance':args.loss_distance,
        'loss_exceed':args.loss_exceed,
        'loss_overlap':args.loss_overlap,
        'loss_shape':args.loss_shape,
        'loss_positive':args.loss_positive,
        'embedding':res.numpy().tolist()
    }
    write_to_file(res_output, json.dumps(final_res))






if __name__ == '__main__':
    main(parse_args())
