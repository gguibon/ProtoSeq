import os
import time
import datetime
from numpy.lib.arraysetops import unique

from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from tqdm import tqdm
from termcolor import colored

from sklearn.metrics import classification_report, confusion_matrix

from code.dataset.parallel_sampler_seq import ParallelSampler
# from code.dataset.parallel_sampler import ParallelSampler
from code.train.utils import named_grad_param, grad_param, get_norm
from code.dataset.supervised_dataset import SupervisedDataset

DEVICE = "CPU"

def train(train_data, val_data, model, args, loader=None):
    '''
        Train the model
        Use val_data to do early stopping
    '''

    global DEVICE
    if args.cuda != -1: DEVICE = torch.device("cuda:%s" % (args.cuda))
    else: DEVICE = "cpu"

    # creating a tmp directory to save the models
    out_dir = os.path.abspath(os.path.join(
                                  os.path.curdir,
                                  "tmp-runs",
                                  str(int(time.time() * 1e7))))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    best_acc = 0
    best_score = 0
    sub_cycle = 0
    best_path = None

    # opt = torch.optim.Adam(grad_param(model, ['ebd', 'clf']), lr=args.lr)
    opt = torch.optim.Adam(grad_param(model, ['ebd', 'clf']), lr=args.lr, betas=(0.9, 0.98), eps=pow(10, -9)) # CESTa optimizer parameters

    if args.negative_lr:
        for opt_i, opt_param in enumerate(opt.param_groups): opt.param_groups[opt_i]['lr'] *= -1  ## negative learning rate

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', patience=2, factor=0.1, verbose=True) # args.patience//2

    print("{}, Start training supervised".format(datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    if loader is None:
        loader = DataLoader(SupervisedDataset(train_data, args), batch_size=args.batch_size, num_workers=2, shuffle=False)

    for ep in range(args.train_epochs):

        grad = {'clf': [], 'ebd': []}
        
        # train on training set
        for batch in tqdm(loader, ncols=80, leave=False, desc=colored('Training on train', 'yellow')):
            train_one(batch, model, opt, args, grad)

        # Evaluate validation accuracy
        cur_acc, cur_std, cur_f1, cur_f1_std, cur_mcc, cur_mcc_std, cur_f1_micro, cur_f1_micro_std = test(val_data, model, args, False)
        print(("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, {:s}{:>7.4f} ± {:>6.4f}, {:s}{:>7.4f} ± {:>6.4f}, {:s}{:>7.4f} ± {:>6.4f},"
               "{:s} {:s}{:>7.4f}, {:s}{:>7.4f}").format(
               datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
               "ep", ep,
               colored("val  ", "cyan"),
               colored("acc:", "blue"), cur_acc, cur_std,
               colored("f1:", "blue"), cur_f1, cur_f1_std,
               colored("mcc:", "blue"), cur_mcc, cur_mcc_std,
               colored("f1 micro:", "blue"), cur_f1_micro, cur_f1_micro_std,
               colored("train stats", "cyan"),
               colored("ebd_grad:", "blue"), np.mean(np.array(grad['ebd'])),
               colored("clf_grad:", "blue"), np.mean(np.array(grad['clf'])),
               ), flush=True)
        scores = {'acc':cur_acc, 'f1': cur_f1, 'mcc': cur_mcc, 'f1_micro': cur_f1_micro}

        # Update the current best model if val acc is better
        # if cur_acc > best_acc:
        #     best_acc = cur_acc
        if scores[args.patience_metric] > best_score:
            best_score = scores[args.patience_metric]
            best_path = os.path.join(out_dir, str(ep))

            print( colored( "{}, Attempt to save cur best model to {}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                best_path) , 'magenta' ))

            while True:
                try:
                    torch.save(model['ebd'].state_dict(), best_path + '.ebd')
                    torch.save(model['clf'].state_dict(), best_path + '.clf')
                    break
                except (FileNotFoundError):
                    continue

            # save current model
            print( colored( "{}, Saved cur best model to {}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                best_path) , 'magenta' ))

            sub_cycle = 0
        else:
            sub_cycle += 1

        if args.scheduler: scheduler.step(cur_acc)

        # Break if the val acc hasn't improved in the past patience epochs
        if sub_cycle == args.patience:
            break

    print("{}, End of training. Restore the best weights".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')),
            flush=True)

    # restore the best saved model
    while True:
        try:
            model['ebd'].load_state_dict(torch.load(best_path + '.ebd'))
            model['clf'].load_state_dict(torch.load(best_path + '.clf'))
            break
        except (FileNotFoundError):
            continue
    

    if args.save:
        # save the current model
        out_dir = os.path.abspath(os.path.join(
                                      os.path.curdir,
                                      "saved-runs",
                                      str(int(time.time() * 1e7))))
        if args.result_path != '':
            dir_path = os.path.split(args.result_path)[0]
            out_dir = os.path.abspath(os.path.join(
                                      os.path.curdir, dir_path) )

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        best_path = os.path.join(out_dir, 'best')

        print(colored("{}, Save best model to {}".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
            best_path), "green"), flush=True)

        torch.save(model['ebd'].state_dict(), best_path + '.ebd')
        torch.save(model['clf'].state_dict(), best_path + '.clf')

        with open(best_path + '_args.txt', 'w') as f:
            for attr, value in sorted(args.__dict__.items()):
                f.write("{}={}\n".format(attr, value))

    return


def train_one(batch, model, opt, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['ebd'].train()
    model['clf'].train()
    opt.zero_grad()
    
    batch['text'] = batch['text'].to(DEVICE)
    batch['label'] = batch['label'].to(DEVICE)

    XS = model['ebd'](batch)
    YS = batch['label']

    if args.classifier == "cesta":
        acc, loss, f1, mcc, f1_micro = model['clf'](XS, YS, None, None, authors=batch['authors'])
    else:
        # Apply the classifier (need to be MLP classifier)
        acc, loss, f1, mcc = model['clf'](XS, YS, None, None)

    if loss is not None:
        loss.backward()

    if torch.isnan(loss):
        return

    if args.clip_grad is not None:
        nn.utils.clip_grad_value_(grad_param(model, ['ebd', 'clf']), args.clip_grad)
        # nn.utils.clip_grad_norm_(grad_param(model, ['ebd', 'clf']), args.clip_grad) #0.5

    grad['clf'].append(get_norm(model['clf']))
    grad['ebd'].append(get_norm(model['ebd']))

    opt.step()


def test(test_data, model, args, verbose=True, target='val', loader=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy, 
        the weighted f1 score and the matthew correlation coeficient and their
        associated std. (ensure the model used is modified to return the values)
    '''
    model['ebd'].eval()
    model['clf'].eval()

    acc, f1, mcc, f1_micro, trues, preds = [], [], [], [], [], []
    if loader is None:
        loader = DataLoader(SupervisedDataset(test_data, args), batch_size=args.batch_size, num_workers=2, shuffle=False)

    for batch in tqdm(loader, desc=colored('Testing regular on %s' % (target), 'yellow'), total=loader.__len__()):
        res_acc, res_f1, res_mcc, res_f1_micro, res_pred, res_true = test_one(batch, model, args, out=(target=='test'))
        acc.append(res_acc)
        f1.append(res_f1)
        mcc.append(res_mcc)
        f1_micro.append(res_f1_micro)
        trues.extend(res_true.cpu().detach().tolist())
        preds.extend(res_pred.cpu().detach().tolist())

    acc, f1, mcc, f1_micro = np.array(acc), np.array(f1), np.array(mcc), np.array(f1_micro)

    if target == 'test' and args.dataset == 'ouitchat_seq':
        target_names = ['no emotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
        labels = [0, 1, 2, 3, 4, 5, 6]
        print( confusion_matrix(np.array(trues), np.array(preds), labels=labels))
        print(classification_report(np.array(trues), np.array(preds), labels=labels, target_names=target_names ) )

    if verbose:
        print("{}, {:s} {:>7.4f} ({:s} {:>7.4f}), {:s} {:>7.4f} ({:s} {:>7.4f}), {:s} {:>7.4f} ({:s} {:>7.4f}), {:s} {:>7.4f} ({:s} {:>7.4f})".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                colored("acc mean", "blue"),
                np.mean(acc),
                colored("std", "blue"),
                np.std(acc),
                colored("f1 mean", "blue"),
                np.mean(f1),
                colored("std", "blue"),
                np.std(f1),
                colored("mcc mean", "blue"),
                np.mean(mcc),
                colored("std", "blue"),
                np.std(mcc),
                colored("f1 micro mean", "blue"),
                np.mean(f1_micro),
                colored("std", "blue"),
                np.std(f1_micro),
                ), flush=True)

        # latex table
        print("{:s} & {:s} & {:>7.4f} \\tiny $\\pm {:>7.4f}$ & {:>7.4f} \\tiny $\\pm {:>7.4f}$ & {:>7.4f} \\tiny $\\pm {:>7.4f}$ & {:>7.4f} \\tiny $\\pm {:>7.4f}$".format(
                args.embedding.replace('_', '\\_'),
                args.classifier.replace('_', '\\_'),
                np.mean(acc),
                np.std(acc),
                np.mean(f1),
                np.std(f1),
                np.mean(mcc),
                np.std(mcc),
                np.mean(f1_micro),
                np.std(f1_micro),
                ), flush=True)
    if args.classifier == 'cesta':
        return np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(mcc), np.std(mcc), np.mean(f1_micro), np.std(f1_micro)
    return np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(mcc), np.std(mcc)


def test_one(batch, model, args, out=False):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''

    batch['text'] = batch['text'].to(DEVICE)
    batch['label'] = batch['label'].to(DEVICE)

    # Embedding the document
    XS = model['ebd'](batch)
    YS = batch['label']

    # Apply the classifier
    if args.dump and out:
        acc, loss, f1, mcc = model['clf'](XS, YS=YS, out=out, XS_ids=batch['ids'])
    elif out and args.classifier != 'cesta':
        acc, loss, f1, mcc, y_pred, y_true = model['clf'](XS, YS=YS, return_preds=True)
        return acc, f1, mcc, y_pred, y_true
    else:
        if args.classifier == 'cesta': 
            acc, loss, f1, mcc, f1_micro, y_pred, y_true = model['clf'](XS, YS=YS, authors=batch['authors'], return_preds=True)
            return acc, f1, mcc, f1_micro, y_pred, y_true
        else: acc, loss, f1, mcc = model['clf'](XS, YS=YS)

    return acc, f1, mcc
