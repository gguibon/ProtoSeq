import torch
from code.classifier.proto_seq import PROTOseq
from code.classifier.cesta import CESTa
from code.dataset.utils import tprint


def get_classifier(ebd_dim, args):
    tprint("Building classifier")

    if args.classifier == 'proto_seq':
        model = PROTOseq(ebd_dim, args)
    elif args.classifier == 'mlp_seq':
        # detach top layer from rest of MLP
        if args.mode == 'finetune':
            top_layer = MLP.get_top_layer(args, args.n_train_class)
            model = MLP(ebd_dim, args, top_layer=top_layer)
        elif args.mode == 'finetune_emotweet':
            top_layer = MLP.get_top_layer(args, args.n_test_class)
            model = MLP(ebd_dim, args, top_layer=top_layer)
        # if not finetune, train MLP as a whole
        else:
            model = MLPseq(ebd_dim, args)
    elif args.classifier == 'cesta':
        model = CESTa(ebd_dim, args)
    else:
        raise ValueError('Invalid classifier. '
                         'classifier can only be: proto_seq, mlp_seq, cesta.')

    if args.snapshot != '':
        if args.multitask:
            print("Loading pretrained embedding from {}".format('%s_%s.clf' % (args.snapshot, args.task) ))
            model.load_state_dict(  torch.load( '%s_%s.clf' % (args.snapshot, args.task) ), strict=False  )
        else:
            # load pretrained models
            tprint("Loading pretrained classifier from {}".format( args.snapshot + '.clf' ))
            model.load_state_dict(torch.load(args.snapshot + '.clf'))

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model
