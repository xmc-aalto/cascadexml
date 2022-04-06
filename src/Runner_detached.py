import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp

from transformers import AdamW
from tqdm import tqdm
from adahessian import Adahessian

from torch.cuda.amp import GradScaler
import torch.distributed as dist
from training_schedule import ThreePhaseOneCycleLR
import os
from sklearn.preprocessing import normalize
from torch.optim.swa_utils import SWALR, AveragedModel
import copy

class Runner:
    def __init__(self, params, train_dl, test_dl, inv_prop, top_k=5):
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.num_train, self.num_test = len(train_dl.dataset), len(test_dl.dataset)
        self.top_k = top_k
        self.use_swa = params.swa
        self.swa_warmup_epoch = params.swa_warmup
        self.swa_update_step = params.swa_step
        self.update_count = params.update_count
        self.inv_prop = torch.from_numpy(inv_prop)

    def save_model(self, model, epoch, name):
        # model.swa_swap_params()
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'scaler': self.scaler.state_dict()
        }
        torch.save(checkpoint, name)
        # model.swa_swap_params()

    def load_model(self, model, name, finetune):
        # model_name = name.split('/')[-2]
        # print("Loading model: " + model_name)
        
        checkpoint = torch.load(name)
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        except RuntimeError as E:
            print(E)

        if not finetune:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint["scaler"])
        init = checkpoint['epoch']
        return model, init
    
    # def predict_binarized(self, preds, y_true, ext=None):
    #     for pred, yt in zip(preds, y_true):
    #         tr = torch.nonzero(yt, as_tuple=True)[0]
    #         match = (pred[..., None].to(tr.device) == tr).any(-1)
    #         self.counts[0] += torch.cumsum(match, dim=0)

    def get_recall(self, cands, y_true, ext=None):
        for pred, yt in zip(preds, y_true):
            tr = torch.nonzero(yt, as_tuple=True)[0]
            match = (pred[..., None].cuda() == tr).any(-1)
            self.counts[i] += torch.sum(match)/len(tr)

    def predict(self, preds, y_true, i, ext = None):
        ext = self.counts if ext is None else ext
        for pred, tr in zip(preds, y_true):
            match = (pred[..., None] == tr).any(-1)
            ext[i] += torch.cumsum(match, dim=0).to(ext[i].device)
        
    def psp(self, preds, y_true, num=None, den=None):
        num = self.num if num is None else num
        den = self.den if den is None else den
        for pred, tr in zip(preds, y_true):
            match = (pred[..., None] == tr).any(-1).double()
            match[match > 0] = self.inv_prop[pred[match > 0]]
            num += torch.cumsum(match, dim=0).to(num.device)

            inv_prop_sample = torch.sort(self.inv_prop[tr], descending=True)[0]

            match = torch.zeros(self.top_k)
            match_size = min(tr.shape[0], self.top_k)
            match[:match_size] = inv_prop_sample[:match_size]
            den += torch.cumsum(match, dim=0).to(den.device)

    def fit_one_epoch(self, model, params, device, epoch):
        trainLoss = torch.tensor(0.0).to(device)
        self.counts = [torch.zeros(self.top_k, dtype=np.int).to(device) for _ in range(len(model.clusters)+1)]
        use_r = params.use_r
        # self.recall = [torch.zeros(100, dtype=np.int).to(device) for _ in range(2)]

        model.train()
        len_dl = len(self.train_dl)

        if self.use_swa and epoch == self.swa_warmup_epoch:
            model.swa_init()
        
        self.optimizer.zero_grad()

        if params.local_rank==0:
            print(f'\nStarting Epoch: {epoch}\n')
            pbar = tqdm(self.train_dl, desc=f"Epoch {epoch}")
        else:
            pbar = self.train_dl

        for step, sample in enumerate(pbar):
            x_batch, attention_mask, labels = sample[0].to(device), sample[1].to(device), sample[2]
            
            with torch.cuda.amp.autocast():
                all_probs, all_candidates, loss = model(x_batch, attention_mask, epoch, labels, use_precomputed=params.train_W, use_r=use_r)
            self.scaler.scale(loss).backward()
            
            if not params.distributed:
                if (step + 1) % 4 == 0:
                    # self.scaler.unscale_(self.optimizer)
                    # nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.cycle_scheduler.step()

            trainLoss += loss.item()

            if self.use_swa and step % self.swa_update_step == 0:
                model.swa_step()

            all_preds = [torch.topk(probs, self.top_k)[1].cpu() for probs in all_probs]
            # all_preds = [torch.topk(probs, 100)[1].cpu() for probs in all_probs[:-1]]
            # all_preds.append(torch.topk(all_probs[-1], self.top_k)[1].cpu())

            if all_candidates is not None:
                all_preds = [
                    candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds].cpu()
                    for candidates, preds in zip(all_candidates, all_preds)
                ]
            
            for i, (preds, label) in enumerate(zip(all_preds, labels)):
                # if use_r:
                #     label, _ = zip(*label)
                self.predict(preds, label, i)
            
            # pbar.set_postfix({'group_counts': self.group_count.tolist(), 'extreme_counts': self.extreme_count.tolist()})
        
        if params.local_rank == 0:
            if params.distributed:
                dist.reduce(trainLoss, dst=0, op=dist.ReduceOp.SUM)
                for c in self.counts:
                    dist.reduce(c, dst=0, op=dist.ReduceOp.SUM)

            trainLoss /= len_dl
            precs = [count.detach().cpu().numpy() * 100.0 / (self.num_train * np.arange(1, self.top_k+1)) for count in self.counts]

            print(f"Epoch: {epoch},  Train Loss: {trainLoss.item()}")
            print("Grouped LRs: ", [param_group['lr'] for param_group in self.optimizer.param_groups])
            for i in range(len(precs)):
                print(f'Level-{i} Training Scores: P@1: {precs[i][0]:.2f}, P@3: {precs[i][2]:.2f}, P@5: {precs[i][4]:.2f}')
            print()
        
        if params.distributed:
            dist.barrier()

        # if trainLoss < self.best_train_Loss:
            # self.best_train_Loss = trainLoss
        # if epoch in [8, 9, 10, 11, 15]:
        #     self.save_model(model, epoch, params.model_name + f"/model_{epoch}.pth")
        
        self.test(model, params, device, epoch)

    def train(self, model, params, device):
        # test only on one process
        self.best_train_Loss = float('Inf')
        self.best_test_acc = 0
        lr = params.lr
        steps_per_epoch = len(self.train_dl)

        if params.distributed:
            module = model.module
        else:
            module = model

        no_decay = ['bias', 'LayerNorm.weight']
        
        if params.train_W:
            optimizer_grouped_parameters = [
                {'params': module.bert.parameters(), 'weight_decay': 0.05, 'lr': lr},
                {'params': module.Cn[:-1].parameters(), 'weight_decay': 0.005, 'lr': lr*10},
                {'params': module.Cn[-1].parameters(), 'weight_decay': 0.005, 'lr': lr}
            ]

        else:
            wd = params.weight_decay 
            optimizer_grouped_parameters = [
                {'params': [p for n, p in module.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': wd, 'lr': lr},
                {'params': [p for n, p in module.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
                {'params': [p for n, p in [*module.Cn_hidden.named_parameters(), *module.Cn.named_parameters()] 
                                                if not any(nd in n for nd in no_decay)], 'weight_decay': wd/10, 'lr': lr*10},
                {'params': [p for n, p in [*module.Cn_hidden.named_parameters()] if any(nd in n for nd in no_decay)] + 
                                                    [p for n, p in [*module.Cn_bias.named_parameters()]], 'weight_decay': 0.0, 'lr': lr*10}
            ]
        
        self.scaler = GradScaler()
        self.optimizer = AdamW(optimizer_grouped_parameters, lr = lr)
        
        init = 0
        last_batch = -1

        if len(params.load_model):
            print("Loading model from ", params.load_model)
            model, init = self.load_model(model, params.load_model, params.train_W)
            last_batch = init*(steps_per_epoch//4)
            
            if params.test:
                self.test(model, params, device, init)
                exit()

            if params.return_embeddings:
                self.return_embeddings(model, params)

            if params.return_shortlist:
                self.return_shortlist(model, params)

            # init = 0
            # last_batch = -1
            # print("Reinitializing Cn weights")
            # model.reinit_weights()

            # if params.train_W:
            #     for param in model.bert.parameters():
            #         param.requires_grad = False

                # for param in model.bert.embeddings.parameters():
                #     param.requires_grad = False

                # print(f"Freezing first {params.freeze_layer_count} layers.")
                # for layer in model.bert.encoder.layer[:params.freeze_layer_count]:
                #     for param in layer.parameters():
                #         param.requires_grad = False

            # lrs = [lr, lr, lr*10, lr*10, lr*10]
            # for i, param_group in enumerate(self.optimizer.param_groups):
            #     param_group['lr'] = lrs[i]

        if params.test:
            self.test(model, params, device, init)
            return

        if params.train_W:
            self.cycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer, max_lr=[lr, lr, lr, lr*10, lr], epochs=params.num_epochs, 
                steps_per_epoch=steps_per_epoch//4 + 1, pct_start=0.4,
                div_factor=20, final_div_factor=200, last_epoch=last_batch)
        else:
            self.cycle_scheduler = ThreePhaseOneCycleLR(
                optimizer=self.optimizer, max_lr=[lr, lr, lr*10, lr*10], epochs=params.num_epochs, 
                steps_per_epoch=steps_per_epoch//4 + 1, pct_epoch=[params.warmup, params.num_epochs-3],
                div_factor=10, final_div_factor=100, last_epoch=last_batch, three_phase=True) 

        print("Starting LRs per group: ", [param_group['lr'] for param_group in self.optimizer.param_groups])
        
        # self.collate_swa_weights(model, params)

        for epoch in range(init, params.num_epochs):
            # if epoch >= 15:
            #     for param in model.bert.embeddings.parameters():
            #         param.requires_grad = False

            #     params.freeze_layer_count = [-1]*15 + [6, 9, 12, 12]
            #     if params.freeze_layer_count[epoch] != -1:
            #         print(f"Freezing first {params.freeze_layer_count[epoch]} layers.")
            #         # if freeze_layer_count == -1, we only freeze the embedding layer
            #         # otherwise we freeze the first `freeze_layer_count` encoder layers
            #         for layer in model.bert.encoder.layer[:params.freeze_layer_count[epoch]]:
            #             for param in layer.parameters():
            #                 param.requires_grad = False
            self.fit_one_epoch(model, params, device, epoch+1)

    @torch.no_grad()
    def test(self, model, params, device, epoch=0):
        if not params.dist_eval and params.local_rank != 0:
            dist.barrier()  # params.distributed will always be true here
            return
        
        model.eval()
        self.counts = [torch.zeros(self.top_k, dtype=np.int).to(device) for _ in range(len(model.clusters)+1)]
        self.weighted_counts = [torch.zeros(self.top_k, dtype=np.int).to(device) for _ in range(len(model.clusters)+1)]
        
        self.num = torch.zeros(self.top_k).cuda()
        self.den = torch.zeros(self.top_k).cuda()

        if params.local_rank==0:
            pbar = tqdm(self.test_dl, desc=f"Epoch {epoch}")
            if self.use_swa:
                model.swa_swap_params()
        else:
            pbar = self.test_dl

        for step, sample in enumerate(pbar): 
            x_batch, attention_mask, labels = sample[0].to(device), sample[1].to(device), sample[2]

            with torch.cuda.amp.autocast():
                all_probs, all_candidates, all_probs_weighted = model(x_batch, attention_mask, epoch, use_precomputed=params.train_W)

            all_preds = [torch.topk(probs, self.top_k)[1].cpu() for probs in all_probs]
            all_weighted_preds = [torch.topk(probs, self.top_k)[1].cpu() for probs in all_probs_weighted]

            if all_candidates is not None:
                all_preds = [candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds].cpu()
                    for candidates, preds in zip(all_candidates, all_preds)]

                all_weighted_preds = [candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds].cpu()
                    for candidates, preds in zip(all_candidates, all_weighted_preds)]
            
            for i, (preds, w_preds, label) in enumerate(zip(all_preds, all_weighted_preds, labels)):
                self.predict(preds, label, i)
                if i > 0:
                    self.predict(w_preds, label, i, self.weighted_counts)
            
            # self.psp(all_preds[-1], labels[-1])
            # pbar.set_postfix({'group_counts': self.group_count.tolist(), 'extreme_counts': self.extreme_count.tolist()})

        if params.local_rank == 0:
            if params.distributed:
                for c, wc in zip(self.counts, self.weighted_counts):
                    dist.reduce(c, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(wc, dst=0, op=dist.ReduceOp.SUM)

            precs = [count.detach().cpu().numpy() * 100.0 / (self.num_test * np.arange(1, self.top_k+1)) for count in self.counts]
            weighted_precs = [count.detach().cpu().numpy() * 100.0 / (self.num_test * np.arange(1, self.top_k+1)) for count in self.weighted_counts]
            # psp = (self.num * 100 / self.den).detach().cpu().numpy()

            for i in range(len(precs)):
                print(f'Level-{i} Test Scores: P@1: {precs[i][0]:.2f}, P@3: {precs[i][2]:.2f}, P@5: {precs[i][4]:.2f}')
                if i != 0:
                    print(f'Level-{i} Weighted Test Scores: P@1: {weighted_precs[i][0]:.2f}, P@3: {weighted_precs[i][2]:.2f}, P@5: {weighted_precs[i][4]:.2f}')
            # print(f"Level-{i} Weighted PSP Score: PSP@1: {psp[0]:.2f}, PSP@3: {psp[2]:.2f}, PSP@5: {psp[4]:.2f}")

            if self.use_swa:
                model.swa_swap_params()

            if(precs[-1][0]+precs[-1][2]+precs[-1][4] > self.best_test_acc and not params.test):
                self.best_test_acc = precs[-1][0]+precs[-1][2]+precs[-1][4]
                self.save_model(model, epoch, params.model_name + "/model_best_test.pth")
            print()
            
        if params.distributed:
            dist.barrier()


    @torch.no_grad()
    def return_embeddings(self, model, params):
        model.eval()
        train_feats, test_feats = [], []
        print(f'\Creating learnt feature embeddings')
        pbar = tqdm(self.train_dl, desc=f"Creating train data embeddings")
        
        for step, sample in enumerate(pbar): 
            x_batch, attention_mask = sample[0].cuda(), sample[1].cuda()
            with torch.cuda.amp.autocast():
                bert_feats = model(x_batch, attention_mask, 0, return_out=True)
            train_feats.append(bert_feats)
        
        train_feats = torch.cat(train_feats, dim=0).numpy()
        np.save('./bert_trn_68_75.npy', train_feats)

        pbar = tqdm(self.test_dl, desc=f"Creating test data embeddings")
        
        for step, sample in enumerate(pbar): 
            x_batch, attention_mask = sample[0].cuda(), sample[1].cuda()
            with torch.cuda.amp.autocast():
                bert_feats = model(x_batch, attention_mask, 0, return_out=True)
            test_feats.append(bert_feats)
        
        test_feats = torch.cat(test_feats, dim=0).numpy()
        np.save('./bert_tst_68_75.npy', test_feats)

        exit()

    @torch.no_grad()
    def return_shortlist(self, model, params):
        model.eval()

        pbar = tqdm(self.train_dl, desc=f"Creating train data embeddings")
        row_idx, col_idx = [], []
        for step, sample in enumerate(pbar): 
            x_batch, attention_mask = sample[0].cuda(), sample[1].cuda()
            with torch.cuda.amp.autocast():
                candidates = model(x_batch, attention_mask, 0)
            row_idx.extend((step*params.batch_size + torch.arange(candidates.shape[0])).repeat_interleave(candidates.shape[1]).tolist())
            col_idx.extend(torch.flatten(candidates).tolist())
        
        value = [1]*len(col_idx)

        n = len(self.train_dl.dataset)
        m = params.num_labels + 1

        shortlist = sp.csr_matrix((value, (row_idx, col_idx)), shape=(n, m))
        shortlist = shortlist[:, :-1]
        sp.save_npz('W10_Shortlist.npz', shortlist)

        # pbar = tqdm(self.test_dl, desc=f"Creating test data embeddings")
        
        # for step, sample in enumerate(pbar): 
        #     x_batch, attention_mask = sample[0].cuda(), sample[1].cuda()
        #     with torch.cuda.amp.autocast():
        #         bert_feats = model(x_batch, attention_mask, 0)
        #     test_feats.append(bert_feats)
        
        # test_feats = torch.cat(test_feats, dim=0).numpy()
        # np.save('./bert_tst_68_23.npy', test_feats)

        exit()