import torch, os
from torch.utils.data import Dataset
import code.dataset.utils as utils
from transformers import AutoTokenizer
import numpy as np

class SupervisedDataset(Dataset):
    def __init__(self, data, args, berttokenizer=False):
        '''
            data : dict_keys(['ids', 'text', 'text_len', 'label', 'raw', 'vocab_size', 'is_train']) 'authors'
        '''
        self.berttokenizer = berttokenizer
        if self.berttokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.pretrained_bert))
        self.args = args
        self.ids = data['ids']
        self.text = data['text']
        self.text_len = data['text_len']
        self.label = data['label']
        self.raw = data['raw']
        self.authors = data['authors']
        self.vocab_size = data['vocab_size']
        self.train = False
        if 'is_train' in data:
            self.is_train = data['is_train']
            self.train = True

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        
        item = {
            'ids': self.ids[idx], 
            'text': self.text[idx], 
            'text_len': self.text_len[idx], 
            # 'label': np.expand_dims(self.label[idx],0),  # .expand_dims(x, axis=0) unsqueeze(0) for seq labelling (bert)
            'label': self.label[idx],
            # 'raw': self.raw[idx].tolist(), 
            'vocab_size': self.vocab_size,
            'authors': self.authors[idx]
        }

        # item = utils.to_tensor(item, self.args.cuda, exclude_keys=[])

        # # if self.args.cuda != -1:
        # #     utils.batch_to_cuda(item, self.args.cuda, exclude_keys=[])
        # print('raw dataset', self.raw[idx], idx)
        # b_sequence = ' '.join( self.raw[idx].tolist() ).replace("<pad>", "", 30)
        # print('self.raw[idx].tolist()',  )
        # print(self.raw[idx], b_sequence)
        # print('tokenized', self.tokenizer(b_sequence, padding='max_length', truncation=True, max_length=30, return_tensors="pt") )
        # print('text idx', self.text[idx].shape, self.text[idx])
        # print('label', self.label[idx], self.label[idx].shape)
        
        


        if self.berttokenizer: 
            item_bert = {'input_ids': torch.zeros(18, 30), 'token_type_ids':  torch.zeros(18, 30), 'attention_mask': torch.zeros(18, 30)}
            for i, (utt, y_true) in enumerate(zip(self.raw[idx], self.label[idx])):
                if y_true == 0: continue
                # print('y_true', y_true)
                # print('utt', utt, len(utt.split(' ')) )
                tokenized = self.tokenizer(utt, padding='max_length', truncation=True, max_length=30,return_tensors="pt")
                # print(tokenized)
                # print(tokenized.size())
                item_bert['input_ids'][i] = tokenized['input_ids'][0]
                item_bert['token_type_ids'][i] = tokenized['token_type_ids'][0]
                item_bert['attention_mask'][i] = tokenized['attention_mask'][0]
            
            item.update(item_bert)

            # print('item', item)
            # print(item['input_ids'].size())
            # exit()
        #     n_messages = 18
        #     n_tokens = 30
        #     b_sequence = self.raw[idx].tolist()
        #     # b_sequence =  ' '.join(b_sequence).replace("<pad>", "", 30) #[ t for m in self.raw[idx].tolist() for t in m ] # flatten the whole conversation
        #     # item.update( self.tokenizer(b_sequence, padding='max_length', truncation=True, max_length=n_tokens*n_messages-30, return_tensors="pt") )
        #     b_sequence =  ' '.join(b_sequence).replace("<pad>", "", n_tokens) #[ t for m in self.raw[idx].tolist() for t in m ] # flatten the whole conversation
        #     item.update( self.tokenizer(b_sequence, padding='max_length', truncation=True, max_length=n_tokens, return_tensors="pt") )
        if self.train: item.update({'is_train': self.is_train})
        
        return item