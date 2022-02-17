import tqdm
import time
import cProfile
import numpy as np
from apex import amp

import torch
from torch import nn

from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

from bert_encoder import DetachedBertModel

from tokenizers import BertWordPieceTokenizer
from transformers import RobertaTokenizerFast

from torch.nn.utils import rnn


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
        # bert = DetachedBertModel.from_pretrained('bert-base-uncased', config=model_config)
        model_config.output_hidden_states = True
        bert = BertModel.from_pretrained('bert-base-uncased', config=model_config)
    return bert



class Detached_LightXML(nn.Module):
    def __init__(self, params, train_ds, feature_layers=5, dropout=0.5):
        super(Detached_LightXML, self).__init__()

        self.use_swa = params.swa
        self.swa_warmup_epoch = params.swa_warmup
        self.swa_update_step = params.swa_step
        self.swa_state = {}
        self.update_count = params.update_count
        print('swa', self.use_swa, self.swa_warmup_epoch, self.swa_update_step, self.swa_state)
        print('update_count', self.update_count)

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.candidates_topk = params.topk
        self.candidates_topk = self.candidates_topk.split(',') if isinstance(self.candidates_topk, str) else self.candidates_topk
        assert isinstance(self.candidates_topk, list), "topK should be a list with 2 integers"

        groups = train_ds.groups
        if params.dataset == 'amazon670k':
            max_group = {8192:82, 16384:41, 32768:21, 65536:11}
        self.num_ele = [len(g) for g in groups] + [params.num_labels]
        
        for i in range(self.num_ele[-2]):
            if len(groups[-1][i]) < max_group[len(groups[-1])]:
                groups[-1][i] = np.pad(groups[-1][i], (0, 1), constant_values=self.num_ele[-1]).astype(np.int32)
            else:
                groups[-1][i] = np.array(groups[-1][i]).astype(np.int32)

        groups = [np.stack(g) for g in groups]
        self.groups = [torch.LongTensor(g).cuda() for g in groups]

        num_meta_labels = [g.shape[0] for g in self.groups]
        print(f'Number of Meta-labels: {num_meta_labels}; top_k: {self.candidates_topk}')

        print('hidden dim:',  params.hidden_dim)
        print('label goup numbers:',  self.num_ele)

        self.bert_name, self.bert = params.bert, get_bert(params.bert)
        self.feature_layers = feature_layers

        hidden_sizes = [self.bert.config.hidden_size, self.bert.config.hidden_size, 3*self.bert.config.hidden_size]
        hidden_drops = [0.2, 0.2, 0.4]

        self.Cn_hidden = nn.ModuleList([nn.Sequential(
                              nn.Dropout(hidden_drops[i]),
                              nn.Linear(hidden_sizes[i], params.hidden_dim),
                              nn.GELU(),
                              nn.Dropout(0.2)
                            ) 
            for i in range(len(self.num_ele))])
        
        out_sizes = self.num_ele[:-1]
        self.Cn = nn.ModuleList([nn.Embedding(out_s, params.hidden_dim) for out_s in out_sizes])
        self.Cn.append(nn.Embedding(self.num_ele[-1]+1, params.hidden_dim, padding_idx=-1))
        
        self.init_classifier_weights()

    def init_classifier_weights(self):
        for C_hid in self.Cn_hidden:
            nn.init.xavier_uniform_(C_hid[1].weight)
        
        for C in self.Cn:
            nn.init.xavier_uniform_(C.weight)
        self.Cn[-1].weight[-1].data.fill_(0)


    def get_candidates(self, group_scores, prev_cands, level, group_gd=None):
        TF_scores = group_scores.clone()
        if group_gd is not None:
            TF_scores += group_gd
        scores, indices = torch.topk(TF_scores, k=self.candidates_topk[level - 1])
        if self.is_training:
            scores = group_scores[torch.arange(group_scores.shape[0]).view(-1,1).cuda(), indices]
        indices = prev_cands[torch.arange(indices.shape[0]).view(-1,1).cuda(), indices]
        candidates = self.groups[level - 1][indices] 
        candidates_scores = torch.ones_like(candidates) * scores[...,None] 
        
        return indices, candidates.flatten(1), candidates_scores.flatten(1)

    def forward(self, input_ids, attention_mask, all_labels = None):
        self.is_training = all_labels is not None
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        token_type_ids = torch.zeros_like(attention_mask).cuda()
        bert_outs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[-1]
        
        # outs = [bert_outs[4][:, 0], bert_outs[7][:, 0], torch.cat([bert_outs[10][:, 0], bert_outs[11][:, 0], bert_outs[12][:, 0]], dim=1)] #256->4K
        outs = [bert_outs[5][:, 0], bert_outs[8][:, 0], torch.cat([bert_outs[10][:, 0], bert_outs[11][:, 0], bert_outs[12][:, 0]], dim=1)] #This is better 
        # outs = [bert_outs[4][:, 0], bert_outs[8][:, 0], torch.cat([bert_outs[10][:, 0], bert_outs[11][:, 0], bert_outs[12][:, 0]], dim=1)]

        prev_logits, prev_labels, all_losses, all_probs, all_probs_weighted, all_candidates = None, None, [], [], [], []
        for i, (hidden, embed, feat) in enumerate(zip(self.Cn_hidden, self.Cn, outs)):
            feat = hidden(feat).unsqueeze(-1)
            if self.is_training:
                labels = all_labels[i]
            
            if i==0:
                candidates = torch.arange(embed.num_embeddings)[None].expand(feat.shape[0], -1).to(embed.weight.device)
                embed_weights = embed(candidates)
                logits = torch.bmm(embed_weights, feat).squeeze(-1)
                candidates_scores = torch.sigmoid(logits)
                all_probs.append(candidates_scores)
                all_probs_weighted.append(all_probs[-1])
                all_candidates.append(candidates)

                prev_logits = candidates_scores.detach()

                if self.is_training:
                    # first level has binarized labels
                    all_losses.append(loss_fn(logits, labels))
                    prev_labels = labels

            else:
                groups, candidates, group_candidates_scores = self.get_candidates(prev_logits, prev_cands, level=i, group_gd=prev_labels)
                new_labels, new_cands, new_group_cands = [], [], []
                
                for j in range(input_ids.shape[0]):
                    if i == len(self.Cn) - 1:
                        new_cands.append(candidates[j][torch.where(candidates[j] != self.num_ele[i])[0]])
                        new_group_cands.append(group_candidates_scores[j][torch.where(candidates[j] != self.num_ele[i])[0]])
                    else:
                        new_cands.append(candidates[j])
                    if self.is_training:
                        ext = labels[j].cuda()
                        lab_bin = (new_cands[-1][..., None] == ext).any(-1).float()
                        new_labels.append(lab_bin)
                
                if self.is_training:
                    labels = rnn.pad_sequence(new_labels, True, 0).cuda()
                if i == len(self.Cn) - 1:
                    candidates = rnn.pad_sequence(new_cands, True, self.num_ele[i])
                    group_candidates_scores = rnn.pad_sequence(new_group_cands, True, 0.)

                embed_weights = embed(candidates) # N, sampled_size, H
                logits = torch.bmm(embed_weights, feat).squeeze(-1)
                
                if i == len(self.Cn) - 1:
                    candidates_scores = torch.where(logits == 0., -np.inf, logits.double()).float().sigmoid()
                else:
                    candidates_scores = torch.sigmoid(logits)     
                
                weighted_scores = candidates_scores * group_candidates_scores
                all_probs.append(candidates_scores)
                all_probs_weighted.append(weighted_scores)
                all_candidates.append(candidates)

                # prev_logits = weighted_scores.detach()
                prev_logits = candidates_scores.detach()

                if self.is_training:
                    all_losses.append(loss_fn(logits, labels))
                    prev_labels = labels
            
            prev_cands = candidates
        
        if self.is_training:
            return all_probs, all_candidates, sum(all_losses)
        else:
            return all_probs, all_candidates, all_probs_weighted

    def save_model(self, path):
        self.swa_swap_params()
        torch.save(self.state_dict(), path)
        self.swa_swap_params()

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

    def get_accuracy(self, candidates, logits, labels):
        if candidates is not None:
            candidates = candidates.detach().cpu()
        scores, indices = torch.topk(logits.detach().cpu(), k=10)

        acc1, acc3, acc5, total = 0, 0, 0, 0
        for i, l in enumerate(labels):
            l = set(np.nonzero(l)[0])

            if candidates is not None:
                labels = candidates[i][indices[i]].numpy()
            else:
                labels = indices[i, :5].numpy()

            acc1 += len(set([labels[0]]) & l)
            acc3 += len(set(labels[:3]) & l)
            acc5 += len(set(labels[:5]) & l)
            total += 1

        return total, acc1, acc3, acc5