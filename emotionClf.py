# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import logging
from threading import local

from numpy.core.numeric import indices
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import pdb, json, sys, os, time, pickle, math, re, argparse, logging, glob, operator, random, string, io, h5py, pickle, csv, gc
from pprint import pprint
import subprocess as sp
from collections import OrderedDict, Counter, defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm
from termcolor import colored
from munch import Munch, DefaultMunch, DefaultFactoryMunch, munchify, unmunchify

from nltk.tokenize import TweetTokenizer

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelWithLMHead, BertModel, BertConfig, BertTokenizer
from transformers import XLMRobertaForMaskedLM, XLMRobertaTokenizer
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
from torch.utils.tensorboard import SummaryWriter

from torchtext.vocab import Vocab, Vectors

import torchnet as tnt
from torchnet.transform import compose
from torchnet.dataset import ListDataset, TransformDataset, TensorDataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sn
import plotly
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
plt.figure(figsize=(15, 7))

from functools import partial

from gensim.models.fasttext import FastText
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import preprocessor as twp


from code.embedding.wordebd import WORDEBD
import code.train.factory as bao_train_utils
import code.train.regular as bao_regular
import code.embedding.factory as ebd
import code.classifier.factory as clf
from code import main as code_main
from code import main_full_supervised as sl_main


## Global Variables
CODE_ARGS = Munch({
	# task configuration
	'way': 5,
	'shot': 5,
	'query': 25,
	'dataset': "dailydialog",
	'n_train_class': 7, 
	'n_val_class': 7, 
	'n_test_class': 7, 
	'data_path': 'data/filename.json',
	'wv_path':"data/", 
	'word_vector':"wiki-news-300d-1M.vec", # "cc.fr.300.vec" for French
	'finetune_ebd': False,
	'mode':'train',
	'maxtokens': 30, # totally related to the amount of RAM of you GPU
	'taskmode': 'episodic', # 'supervised
	# conversational depth options
	'convmode': None, # define the conversation mode. 'seq' is the one we need
	'context_size': 18, # context size from the input dataset (made in batch)
	'protoconv_lstm_layers': 2,
	'crf': True, # if True CRF is used to refine prototype seq predictions
	'authors': False,
	# load bert embeddings for sent-level datasets (optional)
	'bert':False,
	'n_workers': 10,
	'bert_cache_dir': '~/.cache/torch/transformers/',
	'pretrained_bert': 'prajjwal1/bert-tiny',
	# model options
	'auxiliary':[],
	'embedding':'avg_seq',
	'classifier': 'proto_seq',

	# 'meta_w_target':False,
	
	# cnn config
	'cnn_filter_sizes':[3,4,5],
	'cnn_num_filters': 100,
	# proto config
	'proto_hidden': [300,300],
	'warmproto': False,
	# training options 
	'lr': 1e-3,
	'negative_lr': False,
	'clip_grad': None,
	'save': False,
	'dump': False, # to dump outputs during test for later exploration
	'suffix_output': '', # output name for dump (for qualitative evaluation)
	'snapshot': '',
	'notqdm': False,
	'result_path': '',
	'seed': 330,
	'dropout': 0.1,
	'patience': 100,
	'patience_metric': 'f1_micro',
	'cuda': 0, # -1 for cpu,
	'scheduler': False,
	'batch_size': 32, # only used for supervised learning obviously
	# train/test configuration
	'train_epochs': 10000, # epochs or iterations over sets of episodes
	'train_episodes': 100,
	'val_episodes': 100,
	'test_episodes': 1000,
	
	# transformer encoder configurations
	'transfo_emsize': 300,
	'transfo_nhid': 300,
	'transfo_nhead': 2,
	'transfo_nlayers': 1,
	'transfo_dropout': 0.2,
	'transfo_pe_dropout': 0.1,

	# runs
	'scope': 'tiny',
	'targetdata':'dailydialog',
	'dump':False,
	# warmproto RNN configuration
	"warmproto": False,
	"warmprotoconfig": {
		"type": "lstm",
		"input_size": 1202,
		"hidden_size": 150,
		"num_layers": 2,
		"dropout": 0.5,
		"bidirectional": True
		},
})

def creaDailyDialogSeq():
	print(colored('CREATING DAILYDIALOG UNIFIED PREPROCESSED FILE FROM data/ijcnlp_dailydialog/  first make sure the per split dailydialog json files are there. Otherwise, please download dailydialog and run the formatting script as follows:', 'yellow'))
	print("""cd data/ijcnlp_dailydialog
	python3 parser_gg.py -i data/ijcnlp_dailydialog/train -o data/train 
	python3 parser_gg.py -i data/ijcnlp_dailydialog/validation -o data/validation
	python3 parser_gg.py -i data/ijcnlp_dailydialog/test -o data/test""")
	splits = {'train':'data/ijcnlp_dailydialog/train/dailydialog_train.json', 'test':'data/ijcnlp_dailydialog/test/dailydialog_test.json', 'val': 'data/ijcnlp_dailydialog/validation/dailydialog_validation.json'}
	splits_emotionflows_fp = { k: os.path.join( os.path.dirname(v), 'dailydialog_{}_emotionflow.json'.format(k) ) for k,v in splits.items() }
	txt2l = { 'no emotion': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6 }
	l2txt = { v:k for k, v in txt2l.items() } 
	emotionSet = list(set(list(txt2l.keys())))

	emotionFlows = {'train':[],'test':[],'val':[]}
	def getEmotionFlows(row, split):
		row = json.loads(row)
		emotionFlow = [ l2txt[r['label']] for r in row ]
		emotionFlows[split].append(emotionFlow)

	for k,v in tqdm(splits.items()):
		for row in tqdm( open(v, 'r').read().split('\n'), desc=colored(v, 'cyan') ) :
			getEmotionFlows(row, k)

	seq_lengths = np.array([ len(x) for x in emotionFlows['train'] ] )
	seq_size = np.percentile( seq_lengths, 90)
	# seq_size = max(seq_lengths)
	print(colored('seq_size', 'yellow'), seq_size, '75%:', np.percentile(seq_lengths, 75), 'max:', max(seq_lengths))

	def tokenize_seq(chat):
		tokenizer = TweetTokenizer()
		text =  [ ' '.join( list( map(lambda x: x.lower(), tokenizer.tokenize(m)) ) ) for m in chat['text'] ]
		chat['text'] = text
		return chat

	def _trimpad(size, row, pad=True, trim=True):
		''' trim and padding (with <pad>) '''
		if len(row['texts']) > size:
			row['texts'] = row['texts'][int(-size):]
			row['labels'] = row['labels'][int(-size):]
		else:
			row['texts'] = [ ['<pad>' for j in range(5)] for k in range( int(size) - len(row['texts'])  ) ]   +   row['texts']
			row['labels'] = [ 0 for k in range( int(size) - len(row['labels'])  ) ]   +   row['labels']
		
		assert len(row['texts']) == size
		assert len(row['labels']) == size
		return row

	dataSplits = {'train':[], 'test': [], 'val':[]}
	for split in dataSplits.keys():
		for i, data in tqdm( enumerate( [ json.loads(line) for line in open(splits[split], 'r').read().split('\n') ]  ), desc=colored('formatting sequence '+split, 'cyan'), total=len(emotionFlows[split]) ):
			# data = [ tokenize_seq(chat) for chat in tqdm(data, total=len(data), desc='tokenizing') ]
			data = [ tokenize_seq(chat) for chat in data ]
			entry = {'texts': [ x['text'] for x in data], 'labels': [ x['label'] for x in data], 'split': split}
			entry = _trimpad(seq_size, entry)
			dataSplits[split].append(entry)
	
	
	labels_train = Counter([ label for line in dataSplits['train'] for label in line['labels'] ])
	labels_val = Counter([ label for line in dataSplits['val'] for label in line['labels'] ])
	labels_test = Counter([ label for line in dataSplits['test'] for label in line['labels'] ])
	print(colored('labels balance', 'yellow'), labels_train, len(labels_train.keys()), labels_val, len(labels_val.keys()), labels_test, len(labels_test.keys()))

	records = dataSplits['train'] + dataSplits['val'] + dataSplits['test']
	jsonLines = [json.dumps(line) for line  in tqdm(records)]
	with open('data/dailydialog_conv{}seq_splits.json'.format(str(int(seq_size))), 'w') as f: f.write('\n'.join(jsonLines))
	print( colored('data/dailydialog_conv%sseq_splits.json created!' % (str(int(seq_size))), 'green') )
	print(colored('You can now run the labelling tasks.', 'green'))

def runConvSeqFSL(encoder='transfo', train=True, test=True, save=False, dataset='dailydialog', scope='tiny'):
	'''
	FSL for conversation as sequence of messages
	'''

	print('='*10)
	print(dataset)
	print('='*10)
	
	if dataset in ['dailydialog']:
		labels = ['no emotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
		labels = [1, 2, 3, 4, 5, 6]
		CODE_ARGS.update({'n_train_class':7, 'n_val_class':7, 'n_test_class':7, 'way':7, 'shot': 5, 'query':10, 'labels': labels})

	CODE_ARGS['runname'] = '{}_{}_fsl'.format(dataset, encoder)
	CODE_ARGS['dataset'] = "{}_seq".format(dataset)
	CODE_ARGS['convmode'] = 'seq'
	print( colored(CODE_ARGS['runname'],'green') )
	CODE_ARGS['embedding'] = encoder
	CODE_ARGS['save'] = save
	CODE_ARGS['data_path'] = 'data/{}_conv{}seq_splits.json'.format(dataset, CODE_ARGS['context_size'])

	if scope == 'tiny': # just an option to do tiny training/val/test
		CODE_ARGS['train_episodes'] = 10
		CODE_ARGS['val_episodes'] = 10
		CODE_ARGS['test_episodes'] = 100 

	def trainEpisodes():
		CODE_ARGS['result_path'] = 'saved-runs/{}_{}_{}/best_results.pkl'.format(CODE_ARGS['runname'], CODE_ARGS['mode'], CODE_ARGS['embedding'])
		CODE_ARGS['scheduler'] = False

		code_main.main(CODE_ARGS)

	def testEpisodes():
		print('='*10)
		print(colored('TEST', 'yellow'))
		print('='*10)
	
		CODE_ARGS['mode'] = 'test'
		CODE_ARGS['snapshot'] = 'saved-runs/{}_train_{}/best'.format(CODE_ARGS['runname'], CODE_ARGS['embedding'])

		code_main.main(CODE_ARGS)

	if train: trainEpisodes()
	if test: testEpisodes()

def runSupervisedConvSeq(encoder='transfo', train=True, test=True, save=False, dataset='ouitchat', scope='tiny'):
	'''
	run fully supervised emotion sequence labelling
	'''

	labels = ['no emotion', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
	labels = [1, 2, 3, 4, 5, 6]
	CODE_ARGS.update({'n_train_class':7, 'n_val_class':7, 'n_test_class':7, 'way':7, 'shot': 5, 'query':10, 'labels': labels})
	CODE_ARGS.update({'taskmode': 'supervised', 'finetune_ebd': False, 'embedding': 'cnn_seq', 'classifier': 'cesta', 'cnn_num_filters':100, 'authors': True})

	CODE_ARGS['runname'] = '{}_{}_supervised_emotion_seq'.format(dataset, encoder)
	CODE_ARGS['dataset'] = "{}_seq".format(dataset)
	CODE_ARGS['convmode'] = 'seq'
	print( colored(CODE_ARGS['runname'],'green') )
	CODE_ARGS['save'] = save
	CODE_ARGS['data_path'] = 'data/{}_convfull{}seq_splits.json'.format(dataset, CODE_ARGS['context_size']) 
	CODE_ARGS['mode'] = 'supervised'

	CODE_ARGS['result_path'] = 'saved-runs/{}_{}_{}/best_results.pkl'.format(CODE_ARGS['runname'], CODE_ARGS['mode'], CODE_ARGS['embedding'])
	CODE_ARGS['scheduler'] = False
	sl_main.main(CODE_ARGS)

def parse_args():
	parser = argparse.ArgumentParser(description="emotion")

	parser.add_argument("--task", type=str, default="fsl_emoseq",
						help="Classification task"
							  "Options: [prepa_dataset, fsl_emoseq, supervised_emoseq]"
							  "[Default: metalearning]")
	parser.add_argument("--target_data", type=str, default="dailydialog",
						help="target dataset. Only publicly support dailydialog (customerlivechat described in the paper is proprietary)"
							  "Options: [dailydialog, customerlivechat]"
							  "[Default: dailydialog]")
	parser.add_argument("--pipeline", type=str, default="train_test",
						help="pipeline of runs. train_test = train then test"
							  "Options: [train, test, train_test]"
							  "[Default: train_test]")
	parser.add_argument("--encoder", type=str, default="transfo",
						help="Encoder when applied. 'avg' is the faster, 'transfo' yields best results "
							  "Options: [transfo, cnn, avg, avgatt, cnnatt, bert, distilbert, xlnet, distilroberta, roberta]"
							  "[Default: transfo]")
	parser.add_argument("--classifier", type=str, default="proto_seq",
						help="Classifier. proto_seq for few shot and cesta for supervised CESTa baseline "
							  "Options: [cesta]"
							  "[Default: cesta]")
	parser.add_argument("--crf", action="store_true", default=False, help="whether to use CRF to refine prototypes or not")
	parser.add_argument("--warmproto", action="store_true", default=False, help="specify warmproto to apply their config")
	parser.add_argument("--train_epochs", type=int, default=10000,
						help="Number of epochs to train (or maximum number of epochs given the patience is never triggered)"
							  "[Default: 10000]")
	parser.add_argument("--patience", type=int, default=100,
						help="Number of epochs without improvement in the specific metric (on validation set)."
							  "[Default: 100]")
	parser.add_argument("--patience_metric", type=str, default="f1_micro",
						help="Metric to consider for the early stopping."
							  "Options: [acc, f1, mcc, f1_micro]"
							  "[Default: acc]")
	parser.add_argument("--lr", type=float, default=1e-3, 
						help="Learning rate."
							  "Default : 1e-3")
	parser.add_argument("--batch_size", type=int, default=32,
						help="Batch size to use for supervised learning if applicable. Use batch size 64 for CESTa"
							  "[Default: 32]")
	parser.add_argument("--context_size", type=int, default=35,
						help="Size of the conversational context."
							  "[Default: 35]")
	parser.add_argument("--clip_grad", type=float, default=None, help="gradient clipping")
	parser.add_argument("--nosave", action="store_true", default=False, help="do not save the model")
	parser.add_argument("--tiny", action="store_true", default=False, help="trim the training for testing purposes")
	parser.add_argument("--cuda", type=int, default=-1, help="cuda device, -1 for cpu")

	return parser.parse_args()

if __name__ == "__main__":

	args = parse_args()
	CODE_ARGS.update({
			'save': not args.nosave, 'cuda': args.cuda, 'embedding': args.encoder, 'classifier': args.classifier, 'lr': args.lr, 'patience': args.patience, 'patience_metric':args.patience_metric, 'task':args.task, 'scope': args.tiny, 'clip_grad': args.clip_grad, 'batch_size': args.batch_size, 'targetdata': args.target_data, 'crf': args.crf, 'context_size': args.context_size, 'warmproto': args.warmproto
		})
	if args.tiny: CODE_ARGS['scope'] = 'tiny'
	else: CODE_ARGS['scope'] = 'full'
	trainModel = 'train' in args.pipeline
	testModel = 'test' in args.pipeline

	if args.task == 'prepa_dataset':
		creaDailyDialogSeq()
	elif args.task == 'supervised_emoseq':
		runSupervisedConvSeq(encoder=CODE_ARGS.embedding, train=trainModel, test=testModel, save=(not args.nosave), dataset=CODE_ARGS.targetdata, scope=CODE_ARGS.scope)
	else:
		runConvSeqFSL(encoder=CODE_ARGS.embedding, train=trainModel, test=testModel, save=(not args.nosave), dataset=CODE_ARGS.targetdata, scope=CODE_ARGS.scope)