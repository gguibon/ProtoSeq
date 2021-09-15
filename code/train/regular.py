import os
import time
import datetime
from numpy.lib.arraysetops import unique

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from termcolor import colored

from code.dataset.parallel_sampler_seq import ParallelSampler
# from code.dataset.parallel_sampler import ParallelSampler
from code.train.utils import named_grad_param, grad_param, get_norm


def train(train_data, val_data, model, args):
    '''
        Train the model
        Use val_data to do early stopping
    '''
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

    opt = torch.optim.Adam(grad_param(model, ['ebd', 'clf']), lr=args.lr)
    # opt = torch.optim.AdamW(grad_param(model, ['ebd', 'clf']), lr=args.lr)
    # opt = torch.optim.ASGD(grad_param(model, ['ebd', 'clf']), lr=args.lr)
    # opt = torch.optim.SGD(grad_param(model, ['ebd', 'clf']), lr=0.1, momentum=0.9)
    # opt = torch.optim.Adadelta(grad_param(model, ['ebd', 'clf']), lr=args.lr)
    # opt = torch.optim.Adamax(grad_param(model, ['ebd', 'clf']), lr=args.lr)

    if args.negative_lr:
        for opt_i, opt_param in enumerate(opt.param_groups): opt.param_groups[opt_i]['lr'] *= -1  ## negative learning rate
        # opt.param_groups[0]['lr'] *= -1

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, 'max', patience=2, factor=0.1, verbose=True) # args.patience//2

    print("{}, Start training".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    train_gen = ParallelSampler(train_data, args, args.train_episodes)
    train_gen_val = ParallelSampler(train_data, args, args.val_episodes)
    val_gen = ParallelSampler(val_data, args, args.val_episodes)

    for ep in range(args.train_epochs):
        sampled_tasks = train_gen.get_epoch()

        grad = {'clf': [], 'ebd': []}

        if not args.notqdm:
            sampled_tasks = tqdm(sampled_tasks, total=train_gen.num_episodes,
                    ncols=80, leave=False, desc=colored('Training on train',
                        'yellow'))

        for task in sampled_tasks:
            if task is None:
                break
            train_one(task, model, opt, args, grad)

        if ep % 10 == 0:
            acc, std, _, _, _, _, _, _ = test(train_data, model, args, args.val_episodes, False,
                            train_gen_val.get_epoch())
            print("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f} ".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                "ep", ep,
                colored("train", "red"),
                colored("acc:", "blue"), acc, std,
                ), flush=True)

        # Evaluate validation accuracy
        cur_acc, cur_std, cur_f1, cur_f1_std, cur_mcc, cur_mcc_std, cur_f1_micro, cur_f1_micro_std = test(val_data, model, args, args.val_episodes, False,
                                val_gen.get_epoch())
        print(("{}, {:s} {:2d}, {:s} {:s}{:>7.4f} ± {:>6.4f}, {:s}{:>7.4f} ± {:>6.4f}, {:s}{:>7.4f} ± {:>6.4f}, {:s}{:>7.4f} ± {:>6.4f}, "
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
        scores = {'acc':cur_acc, 'f1': cur_f1, 'mcc': cur_mcc, 'f1_micro':cur_f1_micro}

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


def train_one(task, model, opt, args, grad):
    '''
        Train the model on one sampled task.
    '''
    model['ebd'].train()
    model['clf'].train()
    opt.zero_grad()

    support, query = task

    # Embedding the document
    XS = model['ebd'](support)
    YS = support['label']

    XQ = model['ebd'](query)
    YQ = query['label']

    # Apply the classifier
    _, loss, _, _, _ = model['clf'](XS, YS, XQ, YQ)

    if loss is not None:
        loss.backward()

    if torch.isnan(loss):
        # do not update the parameters if the gradient is nan
        # print("NAN detected")
        # print(model['clf'].lam, model['clf'].alpha, model['clf'].beta)
        return

    if args.clip_grad is not None:
        nn.utils.clip_grad_value_(grad_param(model, ['ebd', 'clf']), args.clip_grad)
        # nn.utils.clip_grad_norm_(grad_param(model, ['ebd', 'clf']), args.clip_grad) #0.5

    grad['clf'].append(get_norm(model['clf']))
    grad['ebd'].append(get_norm(model['ebd']))

    opt.step()


def test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None, target='val'):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy, 
        the weighted f1 score and the matthew correlation coeficient and their
        associated std. (ensure the model used is modified to return the values)
    '''
    model['ebd'].eval()
    model['clf'].eval()

    if sampled_tasks is None:
        sampled_tasks = ParallelSampler(test_data, args,
                                        num_episodes).get_epoch()

    acc, f1, mcc, f1_micro = [], [], [], []
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                             leave=False,
                             desc=colored('Testing regular on {}'.format(target), 'yellow'))

    for task in sampled_tasks:
        res_acc, res_f1, res_mcc, res_f1_micro = test_one(task, model, args, out=(target=='test'))
        acc.append(res_acc)
        f1.append(res_f1)
        mcc.append(res_mcc)
        f1_micro.append(res_f1_micro)


    acc, f1, mcc, f1_micro = np.array(acc), np.array(f1), np.array(mcc), np.array(f1_micro)

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
                args.embedding,
                args.classifier,
                np.mean(acc),
                np.std(acc),
                np.mean(f1),
                np.std(f1),
                np.mean(mcc),
                np.std(mcc),
                np.mean(f1_micro),
                np.std(f1_micro),
                ), flush=True)

    return np.mean(acc), np.std(acc), np.mean(f1), np.std(f1), np.mean(mcc), np.std(mcc), np.mean(f1_micro), np.std(f1_micro)


def test_one(task, model, args, out=False):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''
    support, query = task

    # Embedding the document
    XS = model['ebd'](support)
    YS = support['label']

    XQ = model['ebd'](query)
    YQ = query['label']

    # Apply the classifier
    if args.dump and out:
        acc, _, f1, mcc, f1_micro = model['clf'](XS, YS, XQ, YQ, out=out, XS_ids=support['ids'], XQ_ids=query['ids'])
    else:
        acc, _, f1, mcc, f1_micro = model['clf'](XS, YS, XQ, YQ)

    return acc, f1, mcc, f1_micro
