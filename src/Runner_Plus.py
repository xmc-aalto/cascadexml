import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW
from apex import amp
from tqdm import tqdm

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
        model_name = name.split('/')[-2]
        print("Loading model: " + model_name)

        checkpoint = torch.load(name)
        model.load_state_dict(checkpoint)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        init = checkpoint['epoch']
        return model, init
    
    def predict_cluster(self, preds, y_true):
        for pred, yt in zip(preds, y_true):
            tr = torch.nonzero(yt, as_tuple=True)[0]
            match = (pred[..., None].cuda() == tr).any(-1)
            self.group_count += torch.cumsum(match, dim=0)

    def predict(self, preds, y_true, ext=None):
        ext = self.extreme_count if ext is None else ext
        for pred, tr in zip(preds, y_true):
            match = (pred[..., None] == tr).any(-1)
            ext += torch.cumsum(match.cuda(), dim=0)
        
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
        train_loss = 0.0
        self.extreme_count = torch.zeros(self.top_k, dtype=np.int).cuda()
        self.group_count = torch.zeros(self.top_k, dtype=np.int).cuda()

        model.train()
        len_dl = len(self.train_dl)
        
        if self.use_swa and epoch == self.swa_warmup_epoch:
            model.swa_init()

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_dl, desc=f"Epoch {epoch}")
        for step, sample in enumerate(pbar):
            x_batch, attention_mask, ext_labels = sample[0].cuda(), sample[1].cuda(), sample[2]
            if len(sample) == 4:
                group_labels = sample[3].cuda()

            loss, probs, group_probs, candidates = model(x_batch, attention_mask, extreme_labels = ext_labels, group_labels = group_labels)

            loss /= self.update_count
            train_loss += loss.item()

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            if step % self.update_count == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                # self.cycle_scheduler.step()
            
            if self.use_swa and step == self.swa_update_step:
                model.eval()

            # if step % params.eval_step == 0 and step != 0:
            # if step in [10000, 20000]:
            #     self.test(model, params, epoch)

            if self.use_swa and step % self.swa_update_step == 0:
                model.swa_step()

            preds = torch.topk(probs, self.top_k)[1]
            if candidates is not None:
                preds = candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds].cpu()
            self.predict(preds, ext_labels)

            if group_probs is not None:                                
                group_preds = torch.topk(group_probs, self.top_k)[1].cpu()
                self.predict_cluster(group_preds, group_labels)
                
            # pbar.set_postfix({'group_counts': self.group_count.tolist(), 'extreme_counts': self.extreme_count.tolist()})
            
        train_loss /= len_dl
        print(f"Epoch: {epoch},  Train Loss: {train_loss}")
        prec = self.extreme_count.detach().cpu().numpy() * 100.0 / (self.num_train * np.arange(1, self.top_k+1))
        print(f'Extreme Training Scores: P@1: {prec[0]:.2f}, P@3: {prec[2]:.2f}, P@5: {prec[4]:.2f}')

        if group_probs is not None:
            group_prec = self.group_count.detach().cpu().numpy() * 100.0 / (self.num_train * np.arange(1, self.top_k+1))
            print(f'Group   Training Scores: P@1: {group_prec[0]:.2f}, P@3: {group_prec[2]:.2f}, P@5: {group_prec[4]:.2f}')

        # if train_loss < self.best_train_Loss:
        #     self.best_train_Loss = train_loss
        #     self.save_model(model, epoch, params.model_name + "/model_best_epoch.pth")

        self.test(model, params, epoch)


    def train(self, model, params, shortlist=False):
        self.best_train_Loss = float('Inf')
        self.best_test_acc = 0
        lr = params.lr
        steps_per_epoch = len(self.train_dl)
        
        model = model.cuda()

        # self.optimizer = optim.Adam(model.parameters(), lr=lr)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        model, self.optimizer = amp.initialize(model, optimizer, opt_level="O1")

        init = 0
        last_batch = -1

        if len(params.load_model):
            model, init = self.load_model(model, params.load_model)
            last_batch = (init-1)*steps_per_epoch
        
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
            testLoss = 0.0
            self.extreme_count = torch.zeros(self.top_k, dtype=torch.int32).cuda()
            self.comb_count = torch.zeros(self.top_k, dtype=torch.int32).cuda()
            self.num = torch.zeros(self.top_k).cuda()
            self.den = torch.zeros(self.top_k).cuda()

            if self.use_swa:
                model.swa_swap_params()

            pbar = tqdm(self.test_dl, desc=f"Epoch {epoch}")
            for step, sample in enumerate(pbar):
                x_batch, attention_mask, y_tr = sample[0].cuda(), sample[1].cuda(), sample[2]
                candidates, probs, comb_probs = model(x_batch, attention_mask)

                preds = torch.topk(probs, self.top_k)[1] 
                preds = candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds].cpu()
                self.predict(preds, y_tr)

                comb_preds = torch.topk(comb_probs, self.top_k)[1] #Original LightXML uses comb_probs not probs. 
                comb_preds = candidates[np.arange(comb_preds.shape[0]).reshape(-1, 1), comb_preds].cpu()
                self.predict(comb_preds, y_tr, self.comb_count)

                # self.psp(preds, y_tr)

            # testLoss /= self.num_test
            # print(f"Test Loss: {testLoss}")
            prec = self.extreme_count.detach().cpu().numpy() * 100.0 / (self.num_test * np.arange(1, self.top_k+1))
            comb_prec = self.comb_count.detach().cpu().numpy() * 100.0 / (self.num_test * np.arange(1, self.top_k+1))
            print(f"Test scores: P@1: {prec[0]:.2f}, P@3: {prec[2]:.2f}, P@5: {prec[4]:.2f}")
            print(f"Comb Test scores: P@1: {comb_prec[0]:.2f}, P@3: {comb_prec[2]:.2f}, P@5: {comb_prec[4]:.2f}")
            # psp = (self.num * 100 / self.den).detach().cpu().numpy()

            if self.use_swa:
                model.swa_swap_params()

        if(comb_prec[0]+comb_prec[2]+comb_prec[4] > self.best_test_acc and not params.test):
            self.best_test_acc = comb_prec[0]+comb_prec[2]+comb_prec[4]
            self.save_model(model, epoch, params.model_name + "/model_best_test.pth")
