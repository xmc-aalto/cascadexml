import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self, params, train_dl, test_dl, inv_prop = None, top_k=5):
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.num_train, self.num_test = len(train_dl.dataset), len(test_dl.dataset)
        self.top_k = top_k
        self.use_swa = params.swa
        self.swa_warmup_epoch = params.swa_warmup
        self.swa_update_step = params.swa_step
        self.update_count = params.update_count
        # self.inv_prop = torch.from_numpy(inv_prop).cuda()

    def save_model(self, model, epoch, name):
        # model.swa_swap_params()
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, name)
        # model.swa_swap_params()

    def load_model(self, model, name):
        # model_name = name.split('/')[-2]
        # print("Loading model: " + model_name)

        checkpoint = torch.load(name)
        model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
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
            num += torch.cumsum(match, dim=0)

            inv_prop_sample = torch.sort(self.inv_prop[tr], descending=True)[0]

            match = torch.zeros(self.top_k)
            match_size = min(tr.shape[0], self.top_k)
            match[:match_size] = inv_prop_sample[:match_size]
            den += torch.cumsum(match, dim=0)


    def fit_one_epoch(self, model, params, device, epoch):
        trainLoss = torch.tensor(0.0).to(device)
        self.counts = [torch.zeros(self.top_k, dtype=np.int).to(device) for _ in range(3)]
        use_r = False
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
            
            if params.adahess:
                # no amp with adahess
                all_probs, all_candidates, loss = model(x_batch, attention_mask, labels, use_r=use_r)
                loss.backward(create_graph=True)
                if not params.distributed:
                    if (step + 1) % 4 == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.cycle_scheduler.step()   
            else:
                with torch.cuda.amp.autocast():
                    all_probs, all_candidates, loss = model(x_batch, attention_mask, labels, use_precomputed=params.train_W, use_r=use_r)
                self.scaler.scale(loss).backward()
                
                if not params.distributed:
                    if (step + 1) % 4 == 0:
                        # self.scaler.unscale_(self.optimizer)
                        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        self.cycle_scheduler.step()
                        # self.cos_scheduler.step()

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
                if use_r:
                    label, _ = zip(*label)
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
        if epoch in [12]:
            self.save_model(model, epoch, params.model_name + f"/model_{epoch}.pth")
        
        self.test(model, params, device, epoch)

    def fit_one_swa_epoch(self, model, params, device, epoch):
        trainLoss = torch.tensor(0.0).to(device)
        self.counts = [torch.zeros(self.top_k, dtype=np.int).to(device) for _ in range(3)]
        use_r = False
        # self.recall = [torch.zeros(100, dtype=np.int).to(device) for _ in range(2)]

        model.train()
        len_dl = len(self.train_dl)

        self.optimizer.zero_grad()

        if params.local_rank==0:
            print(f'\nStarting Epoch: {epoch}\n')
            pbar = tqdm(self.train_dl, desc=f"Epoch {epoch}")
        else:
            pbar = self.train_dl

        for step, sample in enumerate(pbar):
            x_batch, attention_mask, labels = sample[0].to(device), sample[1].to(device), sample[2]
 
            with torch.cuda.amp.autocast():
                all_probs, all_candidates, loss = model(x_batch, attention_mask, labels, use_r=use_r)
            self.scaler.scale(loss).backward()
            
            if not params.distributed:
                if (step + 1) % 4 == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    # self.cos_scheduler.step()

            trainLoss += loss.item()

            if step % (len_dl//4) == (len_dl//4)-1:
                os.makedirs(f'swa_model/{params.model_name}', exist_ok=True)
                print(f'Saving SWA model at epoch-{epoch} and step-{step}')
                torch.save(model.state_dict(), f'swa_model/{params.model_name}/model_{epoch}_{step}.pth')

            all_preds = [torch.topk(probs, self.top_k)[1].cpu() for probs in all_probs]
            # all_preds = [torch.topk(probs, 100)[1].cpu() for probs in all_probs[:-1]]
            # all_preds.append(torch.topk(all_probs[-1], self.top_k)[1].cpu())

            if all_candidates is not None:
                all_preds = [
                    candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds].cpu()
                    for candidates, preds in zip(all_candidates, all_preds)
                ]
            
            for i, (preds, label) in enumerate(zip(all_preds, labels)):
                if use_r:
                    label, _ = zip(*label)
                self.predict(preds, label, i)
            
            # pbar.set_postfix({'group_counts': self.group_count.tolist(), 'extreme_counts': self.extreme_count.tolist()})
        self.cycle_scheduler.step()
        
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
        
        self.test(model, params, device, epoch)  # maybe not needed

    def collate_swa_weights(self, model, params):
        import glob
        swa_models = sorted(glob.glob(f'swa_model/{params.model_name}/*.pth'))
        for i in range(0, 3):
            avg_model = AveragedModel(copy.deepcopy(model), device=torch.device('cuda:0'))
            for j in range((2**i)-1, len(swa_models), 2**i):
                wts = swa_models[j]
                model.load_state_dict(torch.load(wts))
                print(f'adding wt {wts}')
                avg_model.update_parameters(model)
            self.save_model(avg_model, 20, params.model_name + f"/swa_model_{i}.pth")
            print()
            self.test(avg_model, params, torch.device('cuda:0'), 21 + i)
            print()
    
    @torch.no_grad()
    def recompute_clusters(self, model, params):
        features = []
        print(f'\nRecomputing Clusters & Label Graph')
        # pbar = tqdm(self.train_dl, desc=f"Recomputing Clusters")
        pbar = tqdm(self.test_dl, desc=f"Computing Test features")
        model.eval()
        
        for step, sample in enumerate(pbar): 
            x_batch, attention_mask = sample[0].cuda(), sample[1].cuda()
            with torch.cuda.amp.autocast():
                bert_feats = model(x_batch, attention_mask, return_out=True)
            features.append(bert_feats)
        
        features = torch.cat(features, dim=0).numpy()
        np.save('./bert_test_768.npy', features)

        exit()


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
        
        if params.adahess:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in module.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-4, 'lr': lr},
                {'params': [p for n, p in module.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
                {'params': module.Cn_hidden.parameters(), 'weight_decay': 1e-4, 'lr': lr},
                {'params': module.Cn.parameters(), 'weight_decay': 1e-4, 'lr': lr}
            ]
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in module.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.05, 'lr': lr},
                {'params': [p for n, p in module.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
                {'params': [p for n, p in module.Cn_hidden.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.005, 'lr': lr*10},
                {'params': [p for n, p in module.Cn_hidden.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr*10},
                {'params': module.Cn.parameters(), 'weight_decay': 0.005, 'lr': lr*10}
            ]
        
        self.scaler = GradScaler()
        print("initializing optimizer")
        if params.adahess:
            print('Using AdaHessian optimizer')
            # self.optimizer = Adahessian(optimizer_grouped_parameters, betas=(0.9, 0.98))
            self.optimizer = Adahessian(optimizer_grouped_parameters, betas=(0.9, 0.999))    
        else:
            self.optimizer = AdamW(optimizer_grouped_parameters, lr = lr)
        
        init = 0
        last_batch = -1

        if len(params.load_model):
            print("Loading model from ", params.load_model)
            model, init = self.load_model(model, params.load_model)
            if params.test:
                self.test(model, params, device, init)

            last_batch = init*steps_per_epoch//4 

            init = 0
            last_batch = -1
            # self.recompute_clusters(model, params)
            print("Reinitializing Cn weights")
            model.reinit_weights()

            if params.train_W:
                for param in model.bert.parameters():
                    param.requires_grad = False

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
        
        # self.cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
        #                                     T_max = (steps_per_epoch//4 + 1)*(params.num_epochs-init), eta_min=1e-5)

        if not params.swa:
            if params.train_W:
                self.cycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=self.optimizer, max_lr=[lr, lr, lr*10, lr*10, lr*100], epochs=params.num_epochs, 
                    steps_per_epoch=steps_per_epoch//4 + 1, pct_start=0.4,
                    div_factor=20, final_div_factor=200, last_epoch=last_batch)
            else:
                self.cycle_scheduler = ThreePhaseOneCycleLR(
                    optimizer=self.optimizer, max_lr=[lr, lr, lr*10, lr*10, lr*10], epochs=params.num_epochs, 
                    steps_per_epoch=steps_per_epoch//4 + 1, pct_epoch=[2, 12],
                    div_factor=10, final_div_factor=100, last_epoch=last_batch, three_phase=True) 

        else:
            self.cycle_scheduler = SWALR(
                optimizer=self.optimizer, swa_lr=[lr, lr, lr*10, lr*10, lr*10], anneal_epochs=1,
                anneal_strategy='cos', last_epoch=-1)

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
            if not params.swa:
                self.fit_one_epoch(model, params, device, epoch+1)
            else:
                self.fit_one_swa_epoch(model, params, device, epoch+1)

    @torch.no_grad()
    def test(self, model, params, device, epoch=0):
        if not params.dist_eval and params.local_rank != 0:
            dist.barrier()  # params.distributed will always be true here
            return
        
        model.eval()
        self.counts = [torch.zeros(self.top_k, dtype=np.int).to(device) for _ in range(3)]
        self.weighted_counts = [torch.zeros(self.top_k, dtype=np.int).to(device) for _ in range(3)]

        if params.local_rank==0:
            pbar = tqdm(self.test_dl, desc=f"Epoch {epoch}")
            if self.use_swa:
                model.swa_swap_params()
        else:
            pbar = self.test_dl

        for step, sample in enumerate(pbar): 
            x_batch, attention_mask, labels = sample[0].to(device), sample[1].to(device), sample[2]

            with torch.cuda.amp.autocast():
                all_probs, all_candidates, all_probs_weighted = model(x_batch, attention_mask, use_precomputed=params.train_W)

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

            # pbar.set_postfix({'group_counts': self.group_count.tolist(), 'extreme_counts': self.extreme_count.tolist()})

        if params.local_rank == 0:
            if params.distributed:
                for c, wc in zip(self.counts, self.weighted_counts):
                    dist.reduce(c, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(wc, dst=0, op=dist.ReduceOp.SUM)

            precs = [count.detach().cpu().numpy() * 100.0 / (self.num_test * np.arange(1, self.top_k+1)) for count in self.counts]
            weighted_precs = [count.detach().cpu().numpy() * 100.0 / (self.num_test * np.arange(1, self.top_k+1)) for count in self.weighted_counts]

            for i in range(len(precs)):
                print(f'Level-{i} Test Scores: P@1: {precs[i][0]:.2f}, P@3: {precs[i][2]:.2f}, P@5: {precs[i][4]:.2f}')
                if i != 0:
                    print(f'Level-{i} Weighted Test Scores: P@1: {weighted_precs[i][0]:.2f}, P@3: {weighted_precs[i][2]:.2f}, P@5: {weighted_precs[i][4]:.2f}')

            if self.use_swa:
                model.swa_swap_params()

            if(precs[-1][0]+precs[-1][2]+precs[-1][4] > self.best_test_acc and not params.test):
                self.best_test_acc = precs[-1][0]+precs[-1][2]+precs[-1][4]
                self.save_model(model, epoch, params.model_name + "/model_best_test.pth")
            print()
            
        if params.distributed:
            dist.barrier()
