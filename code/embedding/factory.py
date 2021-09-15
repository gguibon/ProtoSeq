import torch
import datetime

from termcolor import colored

from code.embedding.wordebd import WORDEBD
from code.embedding.cxtebd import CXTEBD

from code.embedding.avg_seq import AVGseq
from code.embedding.cnn_seq import CNNseq
from code.embedding.cnnlstm_seq import CNNLSTMseq
from code.embedding.transfo_seq import TransformerSeqModel as TRANSFOseq
from code.embedding.rnn_seq import RNNseq


def get_embedding(vocab, args):
    print("{}, Building embedding".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    # check if loading pre-trained embeddings
    if args.bert:
        ebd = CXTEBD(args.pretrained_bert,
                     cache_dir=args.bert_cache_dir,
                     finetune_ebd=args.finetune_ebd,
                     return_seq=(args.embedding!='ebd'))
    else:
        ebd = WORDEBD(vocab, args.finetune_ebd)
    
    if args.embedding == 'avg_seq':
        model = AVGseq(ebd, args)
    elif args.embedding == 'cnn_seq':
        model = CNNseq(ebd, args)
    elif args.embedding == 'cnnlstm_seq':
        model = CNNLSTMseq(ebd, args) # ProtoSeq
    elif args.embedding == 'transfo_seq':
        model = TRANSFOseq(ebd, args)
    elif args.embedding == 'rnn_seq':
        model = RNNseq(ebd, args)
    elif args.embedding == 'ebd' and args.bert:
        model = ebd  # using bert representation directly
    elif args.embedding == 'ebd':
        model = ebd

    print("{}, Building embedding".format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

    if args.snapshot != '':
        if args.multitask:
            
            print("{}, Loading pretrained embedding from {}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                '%s_%s.ebd' % (args.snapshot, args.task)
                ))
            model.load_state_dict(  torch.load( '%s_%s.ebd' % (args.snapshot, args.task) ), strict=False  )

        else:    
            # load pretrained models
            print("{}, Loading pretrained embedding from {}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'),
                '{}.ebd'.format(args.snapshot)
                ))
            model.load_state_dict(  torch.load( '{}.ebd'.format(args.snapshot) ), strict=False  )

    if args.cuda != -1: return model.cuda(args.cuda)
    else: return model
