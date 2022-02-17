import tqdm
import time
import cProfile
import numpy as np
from apex import amp

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, rnn
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

from tokenizers import BertWordPieceTokenizer
from transformers import RobertaTokenizerFast

def get_bert(bert_name):
    if 'roberta' in bert_name:
        print('load roberta-base')
        model_config = RobertaConfig.from_pretrained('roberta-base')
        model_config.output_hidden_states = True
        bert = RobertaModel.from_pretrained('roberta-base', config=model_config)
    elif 'xlnet' in bert_name:
        print('load xlnet-base-cased')
        model_config = XLNetConfig.from_pretrained('xlnet-base-cased')
        model_config.output_hidden_states = True
        bert = XLNetModel.from_pretrained('xlnet-base-cased', config=model_config)
    else:
        print('load bert-base-uncased')
        model_config = BertConfig.from_pretrained('bert-base-uncased')
        model_config.output_hidden_states = True
        bert = BertModel.from_pretrained('bert-base-uncased', config=model_config)
    return bert

class LightIncXML(nn.Module):
    def __init__(self, params, group_y=None, feature_layers=3):
        super(LightIncXML, self).__init__()

        self.swa_state = {}
        self.use_swa = params.swa
        self.swa_update_step = params.swa_step
        self.swa_warmup_epoch = params.swa_warmup
        self.update_count = params.update_count
        print('swa', self.use_swa, self.swa_warmup_epoch, self.swa_update_step, self.swa_state)
        print('update_count', self.update_count)

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.candidates_topk = params.topk[0]
        self.num_labels = params.num_labels

        self.bert_name, self.bert, self.feature_layers = params.bert, get_bert(params.bert), feature_layers

        self.group_y = group_y

        model_outsize = self.feature_layers*self.bert.config.hidden_size
        
        if self.group_y is not None:
            if params.dataset == 'amazon670k':
                max_group = {8192:82, 16384:41, 32768:21, 65536:11}
            elif params.dataset == 'amazon3M':
                max_group = {131072:22}
            elif params.dataset == 'wiki500k':
                max_group = {65536:8}

            num_clusters = group_y.shape[0]
            
            for i in range(num_clusters):
                if len(group_y[i]) < max_group[num_clusters]:
                    group_y[i] = np.pad(group_y[i], (0, 1), constant_values=self.num_labels).astype(np.int32)
                else:
                    group_y[i] = np.array(group_y[i]).astype(np.int32)
            group_y = np.stack(group_y)
            self.group_y = torch.LongTensor(group_y).cuda()

            print(f'Number of Meta-labels: {num_clusters}; top_k: {self.candidates_topk}')#; meta-epochs: {self.meta_epoch}')

            drop = 0.3

            self.meta_classifier = nn.Sequential(
                                    nn.Dropout(0.2),
                                    nn.Linear(self.bert.config.hidden_size, params.hidden_dim),
                                    nn.GELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(params.hidden_dim, num_clusters)
            )

            # self.meta_classifier = nn.Linear(params.hidden_dim, num_clusters)

            self.ext_hidden = nn.Sequential(
                                    nn.Dropout(drop),
                                    nn.Linear(model_outsize, params.hidden_dim),
                                    nn.GELU(),
                                    nn.Dropout(0.2)
            )
            
            self.ext_classif_embed = nn.Embedding(params.num_labels+1, params.hidden_dim, padding_idx=self.num_labels)
            nn.init.xavier_uniform_(self.ext_classif_embed.weight[:-1])
            self.ext_classif_embed.weight[-1].data.fill_(0)
        else:
            self.meta_classifier = nn.Sequential(
                                    nn.Dropout(drop),
                                    spectral_norm(nn.Linear(model_outsize, params.hidden_dim)),
                                    nn.GELU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(params.hidden_dim, self.num_labels)
            )


    def get_candidates(self, group_logits, group_gd=None):
        group_scores = torch.sigmoid(group_logits.detach())
        TF_scores = group_scores.clone()
        if group_gd is not None:
            TF_scores += group_gd
        scores, indices = torch.topk(TF_scores, k=self.candidates_topk)
        if self.is_training:
            scores = group_scores[torch.arange(group_scores.shape[0]).view(-1,1).cuda(), indices]
        candidates = self.group_y[indices]
        candidates_scores = torch.ones_like(candidates) * scores[...,None] 
        
        return indices, candidates.flatten(1), candidates_scores.flatten(1)

    def forward(self, input_ids, attention_mask, extreme_labels=None, group_labels=None):
        self.is_training = extreme_labels is not None
        token_type_ids = torch.zeros_like(attention_mask).cuda()
        outs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[-1]

        # takes first word embedding of each hidden state
        hid_last5 = torch.cat([outs[-i][:, 0] for i in range(1, self.feature_layers+1)], dim=-1)
        hid_first = outs[6][:, 0]
        # hid_first = torch.cat([outs[4][:, 0], outs[5][:, 0], outs[6][:, 0]], dim=1)
        # hid_second = torch.cat([outs[10][:, 0], outs[11][:, 0], outs[12][:, 0]], dim=1)

        # meta_logits = self.meta_classifier(hid_last5)
        meta_logits = self.meta_classifier(hid_first)

        if self.group_y is None:
            labels = torch.stack([torch.zeros(self.num_labels).scatter(0, l, 1.0) for l in extreme_labels[0]]).cuda()
            if self.is_training:
                loss = self.loss_fn(meta_logits, extreme_labels.float())
                return loss, meta_logits.sigmoid(), None, None
            else:
                return None, meta_logits.sigmoid(), None
        
        groups, candidates, group_candidates_scores = self.get_candidates(meta_logits, group_gd=group_labels)
        
        new_labels, new_cands, new_group_cands = [], [], []
        for i in range(input_ids.shape[0]):
            new_cands.append(candidates[i][torch.where(candidates[i] != self.num_labels)[0]])
            new_group_cands.append(group_candidates_scores[i][torch.where(candidates[i] != self.num_labels)[0]])
            if self.is_training:
                ext = extreme_labels[i].cuda()
                lab_bin = (new_cands[-1][..., None] == ext).any(-1).float()
                new_labels.append(lab_bin)
        
        if self.is_training:
            labels = rnn.pad_sequence(new_labels, True, 0).cuda()
        candidates = rnn.pad_sequence(new_cands, True, self.num_labels)
        group_candidates_scores = rnn.pad_sequence(new_group_cands, True, 0.)

        h_ext = self.ext_hidden(hid_last5).unsqueeze(-1)
        cand_weights = self.ext_classif_embed(candidates) # N, sampled_size, H
        classif_logits = torch.bmm(cand_weights, h_ext).squeeze(-1)

        candidates_scores = torch.where(classif_logits == 0., -np.inf, classif_logits.double()).float().sigmoid()

        if self.is_training:
            loss = self.loss_fn(classif_logits, labels) + self.loss_fn(meta_logits, group_labels)
            comb_scores = candidates_scores * group_candidates_scores
            return loss, comb_scores, meta_logits.sigmoid(), candidates
        else:
            return candidates, candidates_scores, candidates_scores * group_candidates_scores

    def swa_init(self):
        self.swa_state = {'models_num': 1}
        for n, p in self.named_parameters():
            self.swa_state[n] = p.data.cpu().clone().detach()

    def swa_step(self):
        if 'models_num' not in self.swa_state:
            return
        self.swa_state['models_num'] += 1
        beta = 1.0 / self.swa_state['models_num']
        with torch.no_grad():
            for n, p in self.named_parameters():
                self.swa_state[n].mul_(1.0 - beta).add_(beta, p.data.cpu())

    def swa_swap_params(self):
        if 'models_num' not in self.swa_state:
            return
        for n, p in self.named_parameters():
            self.swa_state[n], p.data =  self.swa_state[n].cpu(), p.data.cpu()
            self.swa_state[n], p.data =  p.data.cpu(), self.swa_state[n].cuda()