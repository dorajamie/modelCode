#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math
import os
import sys
import time
import networkx as nx
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd

import torch
from matplotlib import colors
from matplotlib.patches import Ellipse, Circle
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from codes.utils.filewriter import write_to_file
import codes.utils.data_etl as etl



class Metric(object):


    @staticmethod
    def drawAS():

        pdf_output = '../res/as.pdf'

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        xx = []
        yy = []
        r = 2
        stop = 3.3 * 2 * math.pi
        tmp = 0
        while tmp < stop:

            x = tmp * math.cos(tmp )
            y = tmp * math.sin(tmp )
            x = float(x)
            y = float(y)
            xx.append(x)
            yy.append(y)
            tmp = tmp + 0.001 * math.pi
        # print(xx)
        ax.plot(xx, yy, label='debug', linewidth=2)




        plt.show()

        pp = PdfPages(pdf_output)
        pp.savefig(fig)
        pp.close()


    @staticmethod
    def poincareDraw():
        res = torch.load("../res/mammals_2_best_poincare.pth")
        print(res)




    @staticmethod
    def classification(X, params):
        """
        分类任务
        :param X:所有叶子节点的坐标
        :param params:度量参数
        :return:
        """
        # X_scaled = scale(X)     # 均值化
        # y = dh.load_ground_truth(os.path.join(DATA_PATH, params["ground_truth"]))
        # acc = 0.0
        # for _ in xrange(params["times"]):
        #      X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = params["test_size"], stratify = y)
        #      clf = getattr(mll, params["classification_func"])(X_train, y_train)
        #      acc += mll.infer(clf, X_test, y_test)[1]
        # acc /= float(params["times"])
        # return acc
        pass

    @staticmethod
    def display(embedding):
        embedding = np.around(embedding, decimals=5)
        lower = embedding[:,:8]
        higher = embedding[:,8:]
        # print(lower)
        # print(higher)
        # exit(1)
        diff = higher - lower

        displayfile = '../res/metric_display.txt'

        tree, total_level, all_leaves = etl.prepare_tree('../data/tree2_hamilton')

        tmp = {}
        for i in range(len(diff)):
            tmpl = []
            l = diff[i].tolist()
            for each in l:
                tmpl.append('%.03f'%each)
            s = str(i) + ' ' + str(json.dumps(tmpl)) + "\r\n"
            tmp[i] = s
            # write_to_file(displayfile, s)

        write_to_file(displayfile, tmp[2618])
        write_to_file(displayfile, "\r\n\r\n")
        secondLevel = tree[2618].direct_children
        for i in secondLevel:
            write_to_file(displayfile, tmp[i])
        write_to_file(displayfile, "\r\n\r\n")
        thirdLevel = tree[secondLevel[0]].direct_children
        for j in thirdLevel:
            write_to_file(displayfile, tmp[j])
        write_to_file(displayfile, "\r\n\r\n")
        fourthLevel = tree[thirdLevel[0]].direct_children
        for k in fourthLevel:
            write_to_file(displayfile, tmp[k])

    @staticmethod
    def parseData():
        target = 'georgetown'

        flagFile = '../data/flag_'+target+'.txt'

        tree, total_level, all_leaves = etl.prepare_tree('../data/tree2_'+target)

        flag = {}
        with open(flagFile, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                if len(items) != 2:
                    continue
                if items[1] in flag.keys():
                    flag[items[1]].append(items[0])
                else:
                    flag[items[1]] = [items[0]]
        print(flag['3'])
        flagCnt = {}
        for k,v in flag.items():
            # print(k)
            # print(v)
            flagCnt[k] = len(v)
        print(flagCnt)

    @staticmethod
    def displayTrainRes(embedding):
        embedding = np.around(embedding, decimals=5)
        total = len(embedding)
        for i in range(total):
            index = total - i - 1
            print(index)
            print(embedding[index])
            # print("\r\n")

    @staticmethod
    def visualization(embedding):
        pdf_output = '../res/vis_mammal.pdf'

        fig = plt.figure(figsize=(10, 9.2))
        ax = fig.add_subplot(1, 1, 1)

        for each in embedding:

            start = each[0]
            end = each[1]

            tmp = start
            xx = []
            yy = []
            while tmp < end:
                x = tmp * math.cos(tmp/10)
                y = tmp * math.sin(tmp/10)
                x = float(x)
                y = float(y)
                xx.append(x)
                yy.append(y)
                tmp += math.pi * 0.0001
            ax.plot(xx, yy, label='debug', linewidth=8)

        plt.show()

        # pp = PdfPages(pdf_output)
        # pp.savefig(fig)
        # pp.close()

    @staticmethod
    def drawG():
        graph = etl.prepare_graph('../data/tree2_debug')
        pdf_output = '../res/vis_debug_default.pdf'
        fig = plt.figure(figsize=(30, 30))



        nx.draw(
            graph,
            with_labels=False,
            # pos = nx.random_layout(graph),
            node_size=100,
            width=1,
            node_color='b',
            edge_color='r'
        )



        plt.show()
        # fig = plt.figure(figsize=(30, 30))
        # G = nx.Graph()
        #
        # f = pd.read_csv('../data/mammal_closure.csv', header=0, sep=',')
        # d = pd.DataFrame()
        # d['parent'] = f['id2']
        # d['child'] = f['id1']
        # d['w'] = f['weight']
        # d = d[~(d['parent'] == d['child'])]
        # G.add_weighted_edges_from(d.values)
        #
        # nx.draw(
        #     G,
        #     node_size=100,
        #     width=0.3,
        #     node_color='b',
        #     edge_color='r',
        #     # pos=nx.random_layout(G),
        # )
        # plt.show()
        # pp = PdfPages(pdf_output)
        # pp.savefig(fig)
        # pp.close()

    @staticmethod
    def drawPoincare():
        model = torch.load("../res/mammals_2_best.pth")
        embeddings = model['embeddings']
        pdf_output = '../res/vis_mammal_poincare.pdf'

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        xx = []
        yy = []

        for each in embeddings:
            x = each[0] * 1000
            y = each[1] * 1000
            x = float(x)
            y = float(y)
            xx.append(x)
            yy.append(y)


        ax.plot(xx, yy, label='debug', linewidth=1,color='k')

        plt.show()

        pp = PdfPages(pdf_output)
        pp.savefig(fig)
        pp.close()


    @staticmethod
    def drawGNE():

        pdf_output = '../res/vis_mammal_GNE.pdf'



        f = open('../res/new_train_res', encoding='utf-8')
        content = f.read()  # 使用loads（）方法需要先读文件
        dict = json.loads(content)
        embeddings = dict['coordinates']


        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)

        xx = []
        yy = []

        for each in embeddings:
            x = each[0] * 1
            y = each[1] * 1
            x = float(x)
            y = float(y)
            xx.append(x)
            yy.append(y)
        print(xx)
        print(yy)
        ax.plot(xx, yy, label='debug', linewidth=1, color='k')

        plt.show()

        pp = PdfPages(pdf_output)
        pp.savefig(fig)
        pp.close()