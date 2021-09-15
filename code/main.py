import os
import sys
import pickle
import signal
import argparse
import traceback, copy
from termcolor import colored

import torch
import numpy as np

import code.embedding.factory as ebd
import code.classifier.factory as clf
from code.dataset import loader
import code.train.factory as train_utils

def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if args.embedding != "cnn" and attr[:4] == "cnn_": continue
        if args.classifier != "proto" and attr[:6] == "proto_": continue
        if args.embedding != "cnn" and attr[:4] == "cnn_": continue
        if args.classifier != "mlp" and attr[:4] == "mlp_": continue
        if args.classifier != "proto" and attr[:6] == "proto_": continue
        if args.classifier != "transfo" and attr[:4] == "transfo_": continue
        print("\t{}={}".format(attr.upper(), value))

def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def main(args):

    print(args)
    print_args(args)
    set_seed(args.seed)

    # load data
    train_data, val_data, test_data, vocab = loader.load_dataset(args)
    if args.mode == "test_from_other": 
        args_temp = copy.deepcopy(args)
        args_temp['dataset'] = args_temp['finetuned_dataset']
        args_temp['data_path'] = args_temp['finetuned_data_path']
        args_temp['n_train_class'] = args_temp['finetuned_n_train_class']
        args_temp['n_val_class'] = args_temp['finetuned_n_val_class']
        args_temp['n_test_class'] = args_temp['finetuned_n_test_class']
        train_data, val_data, test_data, vocab = loader.load_dataset(args_temp)

    # initialize model
    model = {}
    model["ebd"] = ebd.get_embedding(vocab, args)
    print('embedding built, now starting clf.get_classifier')
    model["clf"] = clf.get_classifier(model["ebd"].ebd_dim, args)

    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        train_utils.train(train_data, val_data, model, args)

    elif args.mode in ['supervised']:
        print( colored('supervised', 'yellow'), len(train_data['text']) )
        args.query = 1
        args.shot= 1
        args.way = args.n_train_class
        args.train_episodes = len(train_data['text']) #// 2
        # args.train_epochs = len(train_data)
        train_utils.train(train_data, val_data, model, args)

    val_acc, val_std = 0, 0

    if args.mode in ['test_from_other']:
        print(colored('test_from_other', 'green'), colored(args['dataset'], 'red'), colored(args['data_path'], 'red') )
        train_data, val_data, test_data, vocab = loader.load_dataset(args)
        print( colored(args['dataset'], 'green') )

    print( colored('test_data', 'green') )
    # print(test_data['label'], set(test_data['label']))

    test_acc, test_std, _, _, _, _, _, _ = train_utils.test(test_data, model, args, args.test_episodes, target='test')

    if args.result_path:
            directory = args.result_path[:args.result_path.rfind("/")]
            if not os.path.exists(directory):
                os.makedirs(directory)

            result = {
                "test_acc": test_acc,
                "test_std": test_std,
                "val_acc": val_acc,
                "val_std": val_std
            }

            for attr, value in sorted(args.__dict__.items()):
                result[attr] = value

            with open(args.result_path, "wb") as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        os.killpg(0, signal.SIGKILL)

    exit(0)
