import sys, time
from datetime import datetime
import numpy as np
import torch
import os
import logging
import glob
import math
import json
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import TensorDataset, random_split
import utils

# from apex import amp

sys.path.append("./approaches/base/")
from bert_adapter_mask_base import Appr as ApprBase
from my_optimization import BertAdam


class Appr(ApprBase):

    def __init__(self, model, logger, taskcla, args=None):
        super().__init__(model=model, logger=logger, taskcla=taskcla, args=args)
        print('BERT ADAPTER CAPSULE MASK NCL')

        return

    def train(self, t, train, valid, test, num_train_steps, train_data, valid_data):
        global_step = 0
        self.model.to(self.device)

        param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad == True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.args.learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=t_total)

        best_loss = np.inf
        best_model = utils.get_model(self.model)

        # Loop epochs
        for e in range(int(self.args.num_train_epochs)):
            # Train
            # clock0 = time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step = self.train_epoch(t, train, iter_bar, optimizer, t_total, global_step)

            valid_loss, valid_acc, valid_f1_macro = self.eval(t, valid)

            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                # best_model_path = self.args.model_save_path + 'best_model_got_after_epoch_' + str(
                #     e) + '_for_task_' + str(t) + '_' + datetime.now().strftime(
                #     "%Y-%m-%d_%H-%M-%S")
                # torch.save(self.model, best_model_path)
                # torch.save(self.model.state_dict(), best_model_path + '_state_dict')
                best_model = utils.get_model(self.model)
                self.logger.info(' *')
                # print(' *',end='')

        # Restore best
        utils.set_model_(self.model, best_model)

        # Activations mask
        # task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
        mask = self.model.mask(t, s=self.smax)
        for key, value in mask.items():
            mask[key] = torch.autograd.Variable(value.data.clone(), requires_grad=False)

        if t == 0:
            self.mask_pre = mask
        else:
            for key, value in self.mask_pre.items():
                self.mask_pre[key] = torch.max(self.mask_pre[key], mask[key])

        # Weights mask
        self.mask_back = {}
        for n, p in self.model.named_parameters():
            vals = self.model.get_view_for(n, p, self.mask_pre)
            if vals is not None:
                self.mask_back[n] = 1 - vals

        return

    def train_epoch(self, t, data, iter_bar, optimizer, t_total, global_step):
        self.model.train()
        for step, batch in enumerate(iter_bar):
            # print('step: ',step)
            batch = [bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets, _ = batch
            # print('Train Devices', input_ids.get_device(), segment_ids.get_device(), input_mask.get_device(),
            #       targets.get_device(), next(self.model.parameters()).device)
            s = (self.smax - 1 / self.smax) * (step + 1) / len(data) + 1 / self.smax

            output_dict = self.model.forward(t, input_ids, segment_ids, input_mask, targets, s=s)
            # Forward
            masks = output_dict['masks']
            if 'dil' in self.args.scenario:
                output = output_dict['y']
            elif 'til' in self.args.scenario:
                outputs = output_dict['y']
                output = outputs[t]
            loss, reg = self.hat_criterion_adapter(output.view(-1, self.model.tokenizer.vocab_size), targets.view(-1), masks)

            loss.backward()

            # Restrict layer gradients in backprop
            if t > 0:
                for n, p in self.model.named_parameters():
                    if n in self.mask_back and p.grad is not None:
                        p.grad.data *= self.mask_back[n]
                    elif n in self.tsv_para and p.grad is not None:
                        p.grad.data *= self.model.get_view_for_tsv(n, t)  # open for general

            # Compensate embedding gradients
            for n, p in self.model.named_parameters():
                if ('adapter_capsule_mask.e' in n or 'tsv_capsules.e' in n) and p.grad is not None:  # we dont want etsv
                    num = torch.cosh(torch.clamp(s * p.data, -self.thres_cosh, self.thres_cosh)) + 1
                    den = torch.cosh(p.data) + 1
                    p.grad.data *= self.smax / s * num / den

            lr_this_step = self.args.learning_rate * \
                           self.warmup_linear((global_step+1) / t_total, self.args.warmup_proportion)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Constrain embeddings
            for n, p in self.model.named_parameters():
                if 'adapter_capsule_mask.e' in n or 'tsv_capsules.e' in n:
                    p.data = torch.clamp(p.data, -self.thres_emb, self.thres_emb)

            # break

        return global_step

    def eval(self, t, data, test=None, trained_task=None, filter_eval=False):
        total_loss = 0
        total_acc = 0
        total_num = 0
        target_list = []
        pred_list = []

        with torch.no_grad():
            self.model.eval()

            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                if filter_eval:
                    input_ids, segment_ids, input_mask, targets, _, vocab_vector = batch
                else:
                    input_ids, segment_ids, input_mask, targets, _ = batch
                real_b = input_ids.size(0)

                output_dict = self.model.forward(t, input_ids, segment_ids, input_mask, targets, s=self.smax)
                masks = output_dict['masks']
                if 'dil' in self.args.scenario:
                    output = output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs = output_dict['y']
                    output = outputs[t]
                # Forward
                loss, _ = self.hat_criterion_adapter(output.view(-1, self.model.tokenizer.vocab_size), targets.view(-1),
                                                     masks)

                if filter_eval:
                    filtered_output = output[(targets != -100).nonzero(as_tuple=True)] * vocab_vector
                    pred = filtered_output.argmax(1)
                else:
                    pred = output.argmax(2)[(targets != -100).nonzero(as_tuple=True)]
                targets = targets[(targets != -100).nonzero(as_tuple=True)]
                hits = (pred == targets).float()

                target_list.append(targets)
                pred_list.append(pred)

                # Log
                total_loss += loss.data.cpu().numpy().item() * real_b
                total_acc += hits.sum().data.cpu().numpy().item()
                total_num += real_b

            f1 = self.f1_compute_fn(y_pred=torch.cat(pred_list, 0), y_true=torch.cat(target_list, 0), average='macro')

        return total_loss / total_num, total_acc / total_num, f1
