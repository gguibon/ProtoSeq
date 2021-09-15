import os, logging, pickle
import sys
import pickle
import signal
import argparse
import traceback, copy
from collections import Counter
from os.path import join
from termcolor import colored
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import code.embedding.factory as ebd
import code.classifier.factory as clf
from code.dataset import loader
import code.train.factory as train_utils
from code.dataset.supervised_dataset import SupervisedDataset
import code.dataset.utils as dataset_utils
import code.train.supervised as supervised

from transformers import BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, BertForTokenClassification, AutoModelForTokenClassification
from torch.utils.tensorboard import SummaryWriter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


logger = logging.getLogger(__name__)

writer = SummaryWriter(join("runs", "TEMP"))
logger.info("tf writer initialized")

def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def main(args):
    """
        Main class dedicated to supervised learning baseline (CESTa) for clarity
    """

    print(args)
    set_seed(args.seed)

    train_data, val_data, test_data, vocab = loader.load_dataset(args)

    model = {}
    model["ebd"] = ebd.get_embedding(vocab, args)
    model['ebd'].train()
    
    ## Uncomment this if you want to merge both validation and train sets
    # for k in ['ids', 'text', 'text_len', 'label', 'raw']:#, 'vocab_size']:
    #     train_data[k] = np.array(train_data[k].tolist() + val_data[k].tolist())

    train_loader = DataLoader(SupervisedDataset(train_data, args), batch_size=args.batch_size, num_workers=2, shuffle=False)
    val_loader   = DataLoader(SupervisedDataset(val_data, args), batch_size=args.batch_size, num_workers=2, shuffle=False)
    test_loader  = DataLoader(SupervisedDataset(test_data, args), batch_size=args.batch_size, num_workers=2, shuffle=False)

    model["clf"] = clf.get_classifier(model["ebd"].ebd_dim, args)

    args.way = args.n_train_class

    supervised.train(train_data, val_data, model, args, loader=train_loader)

    val_acc, val_std, _, _, _, _, _, _ = supervised.test(val_data, model, args, args.val_episodes, target='val', loader=val_loader)

    print( colored('test_data', 'green') )

    if args.classifier == 'cesta':
        test_acc, test_std, _, _, _, _, _, _ = supervised.test(test_data, model, args, target='test', loader=test_loader)
    test_acc, test_std, _, _, _, _ = supervised.test(test_data, model, args, target='test', loader=test_loader)

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
