import sys
sys.path.append('/home/bohu/github repos/CS598_PROJECT-20240428/CS598_PROJECT')

from common.common import create_folder
from common.pytorch import load_model
import pytorch_pretrained_bert as Bert
from model.utils import age_vocab
from common.common import load_obj
from dataLoader.MLM import MLMLoader
from torch.utils.data import DataLoader
import pandas as pd
from model.MLM import BertForMaskedLM
from model.optimiser import adam
import sklearn.metrics as skm
import numpy as np
import torch
import time
import torch.nn as nn
import os
import pickle


class BertConfig(Bert.modeling.BertConfig):
    def __init__(self, config):
        super(BertConfig, self).__init__(
            vocab_size_or_config_json_file=config.get('vocab_size'),
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('num_hidden_layers'),
            num_attention_heads=config.get('num_attention_heads'),
            intermediate_size=config.get('intermediate_size'),
            hidden_act=config.get('hidden_act'),
            hidden_dropout_prob=config.get('hidden_dropout_prob'),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
            max_position_embeddings = config.get('max_position_embedding'),
            initializer_range=config.get('initializer_range'),
        )
        self.seg_vocab_size = config.get('seg_vocab_size')
        self.age_vocab_size = config.get('age_vocab_size')

class TrainConfig(object):
    def __init__(self, config):
        self.batch_size = config.get('batch_size')
        self.use_cuda = config.get('use_cuda')
        self.max_len_seq = config.get('max_len_seq')
        self.train_loader_workers = config.get('train_loader_workers')
        self.test_loader_workers = config.get('test_loader_workers')
        self.device = config.get('device')
        self.output_dir = config.get('output_dir')
        self.output_name = config.get('output_name')
        self.best_name = config.get('best_name')


file_config = {
    'vocab':'./CS598_PROJECT/output/token2idx',  # vocabulary idx2token, token2idx
    'data': './CS598_PROJECT/output/dataset.pkl',  # formated data
    'model_path': './CS598_PROJECT/modeloutput', # where to save model
    'model_name': 'MLM_MODEL', # model name
    'file_name': 'MLM_LOG',  # log path
}
create_folder(file_config['model_path'])


global_params = {
    'max_seq_len': 64,
    'max_age': 110,
    'month': 1,
    'age_symbol': None,
    'min_visit': 2,
    'gradient_accumulation_steps': 1
}

optim_param = {
    'lr': 3e-5,
    'warmup_proportion': 0.1,
    'weight_decay': 0.01
}

train_params = {
    'batch_size': 256,
    'use_cuda': True,
    'max_len_seq': global_params['max_seq_len'],
    'device': 'cuda:0'
    #'device': 'cpu'
}

BertVocab = load_obj(file_config['vocab'])
ageVocab, _ = age_vocab(max_age=global_params['max_age'], mon=global_params['month'], symbol=global_params['age_symbol'])

data = pd.read_pickle(file_config['data'])
print(data.shape)

# data = pd.read_parquet(file_config['data'])
# remove patients with visits less than min visit
data['length'] = data['ICD9_CODE'].apply(lambda x: len([i for i in range(len(x)) if x[i] == 'SEP']))
data = data[data['length'] >= global_params['min_visit']]
data = data.reset_index(drop=True)
print(data.shape)


# data = pd.read_parquet(file_config['data'])
# remove patients with visits less than min visit
data['length'] = data['ICD9_CODE'].apply(lambda x: len([i for i in range(len(x)) if x[i] == 'SEP']))
data = data[data['length'] >= global_params['min_visit']]
data = data.reset_index(drop=True)
print(data.shape)

Dset = MLMLoader(data, BertVocab['token2idx'], ageVocab, max_len=train_params['max_len_seq'], code='ICD9_CODE', age = 'AGE')
trainload = DataLoader(dataset=Dset, batch_size=train_params['batch_size'], shuffle=True, num_workers=3)

model_config = {
    'vocab_size': len(BertVocab['token2idx'].keys()), # number of disease + symbols for word embedding
    'hidden_size': 288, # word embedding and seg embedding hidden size
    'seg_vocab_size': 2, # number of vocab for seg embedding
    'age_vocab_size': len(ageVocab.keys()), # number of vocab for age embedding
    'max_position_embedding': train_params['max_len_seq'], # maximum number of tokens
    'hidden_dropout_prob': 0.1, # dropout rate
    'num_hidden_layers': 6, # number of multi-head attention layers required
    'num_attention_heads': 12, # number of attention heads
    'attention_probs_dropout_prob': 0.1, # multi-head attention dropout rate
    'intermediate_size': 512, # the size of the "intermediate" layer in the transformer encoder
    'hidden_act': 'gelu', # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
    'initializer_range': 0.02, # parameter weight initializer range
}

conf = BertConfig(model_config)
model = BertForMaskedLM(conf)

model = model.to(train_params['device'])
optim = adam(params=list(model.named_parameters()), config=optim_param)

def cal_acc(label, pred):
    logs = nn.LogSoftmax()
    label=label.cpu().numpy()
    ind = np.where(label!=-1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    truepred = logs(torch.tensor(truepred))
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = skm.precision_score(truelabel, outs, average='micro')
    return precision


def train(e, loader, f):
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    cnt= 0
    start = time.time()

    for step, batch in enumerate(loader):
        cnt +=1
        batch = tuple(t.to(train_params['device']) for t in batch)
        age_ids, input_ids, posi_ids, segment_ids, attMask, masked_label = batch
        # print('masked_label', masked_label.shape)
        # print(masked_label)
        loss, pred, label = model(input_ids, age_ids, segment_ids, posi_ids,attention_mask=attMask, masked_lm_labels=masked_label)
        if global_params['gradient_accumulation_steps'] >1:
            loss = loss/global_params['gradient_accumulation_steps']
        loss.backward()

        temp_loss += loss.item()
        tr_loss += loss.item()

        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        if step % 200==0:
            print("epoch: {}\t| cnt: {}\t|Loss: {}\t| precision: {:.4f}\t| time: {:.2f}".format(e, cnt, temp_loss/2000, cal_acc(label, pred), time.time()-start))
            
            f.write("epoch: {}\t| cnt: {}\t|Loss: {}\t| precision: {:.4f}\t| time: {:.2f}\n".format(e, cnt, temp_loss/2000, cal_acc(label, pred), time.time()-start))

            temp_loss = 0

            # Save the NumPy array to a pickle file
            numpy_array = label.cpu().detach().numpy()
            with open('./CS598_PROJECT/output/label.pickle', 'wb') as f:
                pickle.dump(numpy_array, f)
            numpy_array = pred.cpu().detach().numpy()
            with open('./CS598_PROJECT/output/pred.pickle', 'wb') as f:
                pickle.dump(numpy_array, f)
            start = time.time()

        if (step + 1) % global_params['gradient_accumulation_steps'] == 0:
            optim.step()
            optim.zero_grad()

    print("** ** * Saving fine - tuned model ** ** * ")
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    create_folder(file_config['model_path'])
    output_model_file = os.path.join(file_config['model_path'], file_config['model_name'])

    torch.save(model_to_save.state_dict(), output_model_file)

    cost = time.time() - start
    return tr_loss, cost

f = open(os.path.join(file_config['model_path'], file_config['file_name']), "w")
f.write('epoch: {}\t| cnt: {}\t|Loss: {}\t| precision: {}\t| time: {}\n'.format('epoch', 'cnt', 'loss', 'precision', 'time'))
for e in range(50):
    loss, time_cost = train(e, trainload, f)
    loss = loss/data.shape[0]
    f.write('epoch: {}\t | Loss: {}\t| time: {}\n'.format(e, loss, time_cost))
f.close()