#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import torch
from codes.utils.datahandler import DatasetH
from codes.utils.datahandler import DatasetV
from codes.utils.datahandler import BidirectionalOneShotIterator
import codes.utils.data_etl as etl
from torch.utils.data import DataLoader
from codes.model import ASModel

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Train the Archimedean Spiral Model'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)

    parser.add_argument('-n', '--negative_sample_size', default=1, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    parser.add_argument('-d', '--hidden_dim', default=2, type=int)
    parser.add_argument('--cpu_num', type=int, default=1)
    parser.add_argument('-cm','--circle_margin', type=float)    # 每圈的边界

    parser.add_argument('-b', '--batch_size', default=16, type=int)

    return parser.parse_args(args)



def main(args):
    if (not args.do_train):
        raise ValueError('error.')
    if (args.hidden_dim % 2 != 0):
        raise ValueError('hidden_error')

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
    tree, total_level,all_leaves = etl.prepare_data(args.data_path)

    # init the model
    model = ASModel(
        tree = tree,
        tree_height=total_level,
        all_leaves = all_leaves,
        args = args
    )

    if args.cuda:
        model = model.cuda()

    if args.do_train:
        # Set training dataloader iterator of the horizontal(global) direction.
        train_dataloader_h = DataLoader(
            DatasetH(all_leaves,tree, args,total_level),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            # num_workers=1,
            collate_fn=DatasetH.collate_fn
        )
        train_iterator_h = BidirectionalOneShotIterator(train_dataloader_h)

        # Set training dataloader iterator of the vertical(local) direction.
        train_dataloader_v = DataLoader(
            DatasetV(all_leaves, tree, args, total_level),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            # num_workers=1,
            collate_fn=DatasetV.collate_fn
        )
        train_iterator_v = BidirectionalOneShotIterator(train_dataloader_v)

        # Define the optimizer
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )

        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

        init_step = 0

        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []

        # Training Loop
        for step in range(init_step, args.max_steps):

            log = model.train_step(model, optimizer,train_iterator_v, train_iterator_h)
            print(log)
            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

# def log_metrics(mode, step, metrics):
#     '''
#     Print the evaluation logs
#     '''
#     for metric in metrics:
#         logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


if __name__ == '__main__':
    main(parse_args())