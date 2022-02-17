import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW
from apex import amp
from tqdm import tqdm
from adahessian import Adahessian

from torch.cuda.amp import GradScaler

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
        model.swa_swap_params()
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, name)
        model.swa_swap_params()

    def load_model(self, model, name):
        # model_name = name.split('/')[-2]
        # print("Loading model: " + model_name)

        checkpoint = torch.load(name)
        model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        init = checkpoint['epoch']
        return model, init
    
    def predict_binarized(self, preds, y_true, ext=None):
        for pred, yt in zip(preds, y_true):
            tr = torch.nonzero(yt, as_tuple=True)[0]
            match = (pred[..., None].cuda() == tr).any(-1)
            self.counts[0] += torch.cumsum(match, dim=0)

    # def get_recall(self, cands, y_true, ext=None):
    #     for pred, yt in zip(preds, y_true):
    #         tr = torch.nonzero(yt, as_tuple=True)[0]
    #         match = (pred[..., None].cuda() == tr).any(-1)
    #         self.counts[i] += torch.sum(match)/len(tr)

    def predict(self, preds, y_true, i, ext = None):
        ext = self.counts if ext is None else ext
        for pred, tr in zip(preds, y_true):
            match = (pred[..., None] == tr).any(-1)
            ext[i] += torch.cumsum(match, dim=0)
        
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


    def fit_one_epoch(self, model, params, epoch):
        trainLoss = 0.0
        self.counts = [torch.zeros(self.top_k, dtype=np.int) for _ in range(3)]
        self.counts[0] = self.counts[0].cuda()

        model.train()
        len_dl = len(self.train_dl)
        
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_dl, desc=f"Epoch {epoch}")
        for step, sample in enumerate(pbar):
            x_batch, attention_mask, labels = sample[0].cuda(), sample[1].cuda(), sample[2]
            labels[0] = labels[0].cuda() # 1st label is binarized

            if params.adahess:
                # no amp with adahess
                all_probs, all_candidates, loss = model(x_batch, attention_mask, labels)
                loss.backward(create_graph=True)
                self.optimizer.step()
            else:
                with torch.cuda.amp.autocast():
                    all_probs, all_candidates, loss = model(x_batch, attention_mask, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            trainLoss += loss.item()
            self.optimizer.zero_grad()
            
            
            # if params.adahess:
            #     loss.backward(create_graph=True)
            # else:
            #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #         scaled_loss.backward()
                

            # self.optimizer.step()
            # self.optimizer.zero_grad()
            # # self.scheduler.step()
            # # self.cycle_scheduler.step()
            
            # if step % params.eval_step == 0 and step != 0:
            if step in [7500]:
                self.test(model, params, epoch)

            # if self.use_swa and step % self.swa_update_step == 0:
            #     model.swa_step()

            all_preds = [torch.topk(probs, self.top_k)[1].cpu() for probs in all_probs]
            if all_candidates is not None:
                all_preds = [
                    candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds].cpu()
                    for candidates, preds in zip(all_candidates, all_preds)
                ]
            
            for i, (preds, label) in enumerate(zip(all_preds, labels)):
                if i==0:
                    self.predict_binarized(preds, label)
                else:
                    self.predict(preds, label, i)
            
            # pbar.set_postfix({'group_counts': self.group_count.tolist(), 'extreme_counts': self.extreme_count.tolist()})
            
        trainLoss /= len_dl

        print(f"Epoch: {epoch},  Train Loss: {trainLoss}")
        precs = [count.detach().cpu().numpy() * 100.0 / (self.num_train * np.arange(1, self.top_k+1)) for count in self.counts]

        for i in range(len(precs)):
            print(f'Level-{i} Training Scores: P@1: {precs[i][0]:.2f}, P@3: {precs[i][2]:.2f}, P@5: {precs[i][4]:.2f}')

        # if trainLoss < self.best_train_Loss:
        #     self.best_train_Loss = trainLoss
        #     self.save_model(model, epoch, params.model_name + "/model_best_epoch.pth")
        
        self.scheduler.step()
        self.test(model, params, epoch)


    def train(self, model, params, shortlist=False):
        self.best_train_Loss = float('Inf')
        self.best_test_acc = 0
        lr = params.lr
        steps_per_epoch = len(self.train_dl)

        model = model.cuda()

        no_decay = ['bias', 'LayerNorm.weight']
        if params.adahess:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-4, lr: lr},
                {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, lr: lr},
                {'params': model.Cn_hidden.parameters(), 'weight_decay': 1e-4, lr: lr},
                {'params': model.Cn.parameters(), 'weight_decay': 1e-4, lr: lr},
                # {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, lr: lr/10},
                # {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.01, lr: lr/10},
                # {'params': model.Cn_hidden.parameters(), 'weight_decay': 1e-4, lr: lr},
                # {'params': model.Cn.parameters(), 'weight_decay': 1e-4, lr: lr},
            ]
        else:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, lr: lr/10},
                {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, lr: lr/10},
                {'params': model.Cn_hidden.parameters(), 'weight_decay': 1e-4, lr: lr},
                {'params': model.Cn.parameters(), 'weight_decay': 1e-4, lr: lr},
            ]
        self.scaler = GradScaler()
        if params.adahess:
            print('Using AdaHessian optimizer')
            # self.optimizer = Adahessian(optimizer_grouped_parameters, betas=(0.9, 0.98))
            self.optimizer = Adahessian(optimizer_grouped_parameters, betas=(0.9, 0.999))
            # model, self.optimizer = amp.initialize(model, optimizer, opt_level="O1")
            
        else:
            self.optimizer = AdamW(optimizer_grouped_parameters)
            # model, self.optimizer = amp.initialize(model, optimizer, opt_level="O1")
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[12])

        init = 0
        last_batch = -1        

        if len(params.load_model) or shortlist:
            model, init = self.load_model(model, params.load_model)
            last_batch = (init-1)*steps_per_epoch
            for param in model.bert.parameters():
                param.requires_grad = False

        if params.test:
            self.test(model, params, init)
            return

        # self.cycle_scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer=self.optimizer, max_lr=(params.batch_size/128)*params.lr,
        #     epochs=params.num_epochs, steps_per_epoch=steps_per_epoch, pct_start=0.33,
        #     div_factor=10, final_div_factor=1e4, last_epoch=last_batch)

        for epoch in range(init, params.num_epochs):
            self.fit_one_epoch(model, params, epoch+1)

    def test(self, model, params, epoch=0):
        model.eval()
        with torch.no_grad():
            self.counts = [torch.zeros(self.top_k, dtype=np.int) for _ in range(3)]
            self.counts[0] = self.counts[0].cuda()
            self.weighted_counts = [torch.zeros(self.top_k, dtype=np.int) for _ in range(3)]

            pbar = tqdm(self.test_dl, desc=f"Epoch {epoch}")
            for step, sample in enumerate(pbar): 
                x_batch, attention_mask, labels = sample[0].cuda(), sample[1].cuda(), sample[2]
                labels[0] = labels[0].cuda() # 1st label is binarized

                all_probs, all_candidates, all_probs_weighted = model(x_batch, attention_mask)

                all_preds = [torch.topk(probs, self.top_k)[1].cpu() for probs in all_probs]
                all_weighted_preds = [torch.topk(probs, self.top_k)[1].cpu() for probs in all_probs_weighted]

                if all_candidates is not None:
                    all_preds = [candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds].cpu()
                        for candidates, preds in zip(all_candidates, all_preds)]

                    all_weighted_preds = [candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds].cpu()
                        for candidates, preds in zip(all_candidates, all_weighted_preds)]
 
                for i, (preds, w_preds, label) in enumerate(zip(all_preds, all_weighted_preds, labels)):
                    if i==0:
                        self.predict_binarized(preds, label)
                    else:
                        self.predict(preds, label, i)
                        self.predict(w_preds, label, i, self.weighted_counts)
                
                # pbar.set_postfix({'group_counts': self.group_count.tolist(), 'extreme_counts': self.extreme_count.tolist()})

            precs = [count.detach().cpu().numpy() * 100.0 / (self.num_test * np.arange(1, self.top_k+1)) for count in self.counts]
            weighted_precs = [count.detach().cpu().numpy() * 100.0 / (self.num_test * np.arange(1, self.top_k+1)) for count in self.weighted_counts]

            for i in range(len(precs)):
                print(f'Level-{i} Test Scores: P@1: {precs[i][0]:.2f}, P@3: {precs[i][2]:.2f}, P@5: {precs[i][4]:.2f}')
                if i != 0:
                    print(f'Level-{i} Weighted Test Scores: P@1: {weighted_precs[i][0]:.2f}, P@3: {weighted_precs[i][2]:.2f}, P@5: {weighted_precs[i][4]:.2f}')

            if(precs[-1][0]+precs[-1][2]+precs[-1][4] > self.best_test_acc and not params.test):
                self.best_test_acc = precs[-1][0]+precs[-1][2]+precs[-1][4]
                self.save_model(model, epoch, params.model_name + "/model_best_test.pth")