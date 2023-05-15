import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import scipy
import numpy as np

from transformers import BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig
from transformers import XLNetModel, XLNetConfig
from torch.utils.checkpoint import checkpoint

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
        # model_config.output_attentions = True
        # model_config.gradient_checkpointing = True
        bert = BertModel.from_pretrained('bert-base-uncased', config=model_config)
    return bert

def feat_maker(recipe, bert_outs):
    feats = [None] * len(recipe)
    for i, idx_list in enumerate(recipe):
        if isinstance(idx_list, int):
            feats[i] = bert_outs[idx_list][:, 0]
        
        elif isinstance(idx_list, tuple):
            feats[i] = torch.cat([bert_outs[idx][:, 0] for idx in idx_list], dim=1)

        else:
            raise ValueError("Invalid feat recipe")
    return feats
    

class CascadeXML(nn.Module):
    def __init__(self, params, train_ds, device):
        super(CascadeXML, self).__init__()

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.candidates_topk = params.topk
        self.candidates_topk = self.candidates_topk.split(',') if isinstance(self.candidates_topk, str) else self.candidates_topk
        assert isinstance(self.candidates_topk, list), "topK should be a list with at least 2 integers"
        self.return_shortlist = params.return_shortlist
        self.device = device
        self.rw_loss = params.rw_loss


        clusters = train_ds.groups 
        max_cluster = max([len(c) for c in clusters[-1]])
        self.num_ele = [len(g) for g in clusters] + [params.num_labels]
                
        for i in range(self.num_ele[-2]):
            clusters[-1][i] = np.pad(clusters[-1][i], (0, max_cluster-len(clusters[-1][i])), 
                                        constant_values=self.num_ele[-1]).astype(np.int32)

        clusters = [np.stack(c) for c in clusters]
        self.clusters = [torch.LongTensor(c).to(device) for c in clusters]

        num_meta_labels = [c.shape[0] for c in self.clusters]
        
        print(f'label goup numbers: {self.num_ele}; top_k: {self.candidates_topk}')
        
        if len(self.num_ele) == 4:
            self.layers = [(5, 6), 8, 10, 12]
            # self.layers = [(8, 9), 10, 11, 12] #other option
        elif len(self.num_ele) == 3:
            self.layers = [(7, 8), 10, 12]
            # self.layers = [(9, 10), 11, 12] #other option

        embed_drops = params.embed_drops
        
        print(f'Layers used: {self.layers}; Dropouts: {embed_drops}')
        assert len(self.layers) == len(self.num_ele)
        
        self.bert_name, self.bert = params.bert, get_bert(params.bert)
        
        embed_size = self.bert.config.hidden_size
        concat_size = len(self.layers[0])*embed_size
        self.Cn_hidden = nn.Sequential(
                              nn.Dropout(0.2),
                              nn.Linear(concat_size, embed_size)
                        )
        
        self.embed_drops = nn.ModuleList([nn.Dropout(p) for p in embed_drops])
        
        self.Cn = nn.ModuleList([nn.Embedding(out_s, embed_size, sparse=params.sparse) for i, out_s in enumerate(self.num_ele[:-1])])
        self.Cn.append(nn.Embedding(self.num_ele[-1]+1, embed_size, padding_idx=-1, sparse=params.sparse))

        self.Cn_bias = nn.ModuleList([nn.Embedding(out_s, 1, sparse=params.sparse) for out_s in self.num_ele[:-1]])
        self.Cn_bias.append(nn.Embedding(self.num_ele[-1]+1, 1, padding_idx=-1, sparse=params.sparse))

        self.init_classifier_weights()


    def init_classifier_weights(self):
        nn.init.xavier_uniform_(self.Cn_hidden[1].weight)

        for C in self.Cn:
            nn.init.xavier_uniform_(C.weight)
        self.Cn[-1].weight[-1].data.fill_(0)

        for bias in self.Cn_bias:
            bias.weight.data.fill_(0)

    def reinit_weights(self):
        for C in self.Cn[:-1]:
            nn.init.xavier_uniform_(C.weight)

    def get_candidates(self, group_scores, prev_cands, level, group_gd=None):
        TF_scores = group_scores.clone()
        if group_gd is not None:
            TF_scores += group_gd
        scores, indices = torch.topk(TF_scores, k=self.candidates_topk[level - 1])
        if self.is_training:
            scores = group_scores[torch.arange(group_scores.shape[0]).view(-1,1).to(self.device), indices]
        indices = prev_cands[torch.arange(indices.shape[0]).view(-1,1).to(self.device), indices]
        candidates = self.clusters[level - 1][indices] 
        candidates_scores = torch.ones_like(candidates) * scores[...,None] 

        return indices, candidates.flatten(1), candidates_scores.flatten(1)

    def forward(self, input_ids, attention_mask, epoch=None, all_labels = None, return_out=False):
        self.is_training = all_labels is not None
        
        token_type_ids = torch.zeros_like(attention_mask).to(self.device)
        bert_outs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[-1]

        if return_out:
            # out = torch.stack(bert_outs).detach().clone().cpu() #Attention Maps
            # out = [bert_outs[idx][:, 0].detach().clone().cpu() for idx in range(13)] #CLS Token
            out = bert_outs[-1][:, 0].detach().clone().cpu() #last embedding
            del bert_outs
            return out

        outs = feat_maker(self.layers, bert_outs)

        prev_logits, prev_labels, all_losses, all_probs, all_probs_weighted, all_candidates = None, None, [], [], [], []
        for i, (embed, feat) in enumerate(zip(self.Cn, outs)):
            if i == 0:
                feat = self.Cn_hidden(feat)
            feat = self.embed_drops[i](feat).unsqueeze(-1)

            if self.is_training:
                labels = all_labels[i]
            
            if i == 0:
                candidates = torch.arange(embed.num_embeddings)[None].expand(feat.shape[0], -1)
            else:
                shortlisted_clusters, candidates, group_candidates_scores = self.get_candidates(prev_logits, prev_cands, level=i, group_gd=prev_labels)
            
            candidates = candidates.to(input_ids.device)
        
            new_labels, new_cands, new_group_cands = [], [], []
            for j in range(input_ids.shape[0]):
                if i == len(self.Cn) - 1:
                    new_cands.append(candidates[j][torch.where(candidates[j] != self.num_ele[i])[0]])
                    new_group_cands.append(group_candidates_scores[j][torch.where(candidates[j] != self.num_ele[i])[0]])
                else:
                    new_cands.append(candidates[j])
                
                if self.is_training:
                    ext = labels[j].to(candidates.device)
                    lab_bin = (new_cands[-1][..., None] == ext).any(-1).float()
                    new_labels.append(lab_bin)
            
            if self.is_training:
                labels = pad_sequence(new_labels, True, 0).to(input_ids.device)
            if i == len(self.Cn) - 1:
                candidates = pad_sequence(new_cands, True, self.num_ele[i])
                group_candidates_scores = pad_sequence(new_group_cands, True, 0.)
                
                if self.return_shortlist:
                    return candidates

            candidates = candidates.to(embed.weight.device)
            embed_weights = embed(candidates)
            logits = torch.bmm(embed_weights, feat.to(embed.weight.device)).squeeze(-1)
            logits = (logits + self.Cn_bias[i].to(embed.weight.device)(candidates).squeeze(-1)).to(input_ids.device) 

            if i == len(self.Cn) - 1:
                candidates_scores = torch.where(logits == 0., -np.inf, logits.double()).float().sigmoid() #Handling padding
            else:
                candidates_scores = torch.sigmoid(logits) 

            weighted_scores = candidates_scores * group_candidates_scores if i != 0 else candidates_scores
            
            all_probs.append(candidates_scores)
            all_probs_weighted.append(weighted_scores)
            all_candidates.append(candidates)
            
            prev_logits = candidates_scores.detach()
            prev_cands = candidates

            if self.is_training:
                all_losses.append(self.loss_fn(logits, labels))
                prev_labels = labels
        
        if self.is_training:
            sum_loss = 0.
            for i, l in enumerate(all_losses):
                sum_loss += l * self.rw_loss[i]
            return all_probs, all_candidates, sum_loss
        else:
            return all_probs, all_candidates, all_probs_weighted