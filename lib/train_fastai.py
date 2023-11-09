import torch
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
import fastcore.all as fc
from operator import attrgetter
from functools import partial
from copy import copy
from collections.abc import Mapping
import math
from torch.optim.lr_scheduler import ExponentialLR
from fastprogress import progress_bar,master_bar
from torcheval.metrics import Mean

def_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass

def run_cbs(cbs,method_nm):
    for cb in sorted(cbs,key = attrgetter('order')):
        method = getattr(cb, method_nm, None)
        if method is not None: method()
    
class Learner():
    def __init__(self,model,dls,loss_func,opt,acc,cbs):
        fc.store_attr()
        for cb in cbs: cb.learn = self
        self.train = True
        
    @contextmanager
    def callback_ctx(self,nm):
        try:
            self.callback(f'before_{nm}')
            yield
            self.callback(f'after_{nm}')
        except globals()[f'Cancel{nm.title() }Exception']: pass
            
    def one_epoch(self):
        self.model.train(self.train)
        self.dl = self.dls['train'] if self.train else self.dls['valid']
        with self.callback_ctx('epoch'):
            for self.iter, self.batch in enumerate(self.dl):
                with self.callback_ctx('batch'):
                    self.predict()
                    self.get_loss()
                    if self.model.training:
                        self.backward()
                        self.step()
                        self.zero_grad()
    
    def fit(self, n_epochs):
        self.n_epochs = n_epochs
        self.epochs = range(n_epochs)
        # self.opt = self.opt_func(self.model.parameters(), self.lr)
        with self.callback_ctx('fit'):
            for self.epoch in self.epochs:
                self.one_epoch()
                self.train = False
                self.one_epoch()
                self.train = True
    
    def callback(self,method_nm): run_cbs(self.cbs,method_nm)
    
    def __getattr__(self,name):
        if name in ('predict','get_loss','backward','step','zero_grad'):
            return partial(self.callback,name)
        raise AttributeError(name)

#############
# Callbacks #
#############

# base class
class Callback(): order = 0

# completion callback
class CompletionCB(Callback):
    def before_fit(self): self.count = 0
    def after_batch(self): self.count += 1
    def after_fit(self): print(f'Completed {self.count} batches')
    
# single batch callback
class SingleBatchCB(Callback):
    order = 1
    def after_batch(self): raise CancelEpochException()

# function for sending data to device
def to_device(x, device=def_device):
    if isinstance(x, torch.Tensor): return x.to(device)
    if isinstance(x, Mapping): return {k:v.to(device) for k,v in x.items()}
    return type(x)(to_device(o, device) for o in x)
    
    
# callback for putting model and batches on device
class DeviceCB(Callback):
    def __init__(self,device = def_device): fc.store_attr()
    def before_fit(self): self.learn.model.to(self.device)
    def before_batch(self):
        self.learn.batch = to_device(self.learn.batch,device=def_device)

# function for sending data to cpu (needed by metrics callback)
def to_cpu(x):
    if isinstance(x,Mapping): return {k:to_cpu(v) for k,v in x.items()}
    if isinstance(x,list): return [to_cpu(o) for o in x]
    if isinstance(x,tuple): return tuple(to_cpu(list(x)))
    return x.detach().cpu()

# callback for computing and logging metrics
# class MetricsCB(Callback):
#     def __init__(self, *ms, **metrics):
#         for o in ms: metrics[type(o).__name__] = o
#         self.metrics = metrics
#         self.all_metrics = copy(metrics)
#         self.all_metrics['loss'] = self.loss = Mean()
        
#     def _log(self,d): print(d)
#     def before_fit(self): self.learn.metrics = self
#     def before_epoch(self): [o.reset() for o in self.all_metrics.values()]
#     def after_epoch(self):
#         log = {k:f'{v.compute():.4f}' for k,v in self.all_metrics.items()}
#         log['epoch'] = self.learn.epoch
#         log['train'] = self.learn.model.training
#         self._log(log)
        
#     def after_batch(self):
#         x,y = to_cpu(self.learn.batch)
#         for m in self.metrics.values(): m.update(to_cpu(self.learn.preds),y)
#         self.loss.update(to_cpu(self.learn.loss),weight = len(x))

# callback for defining various attributes used in training
class TrainCB(Callback):
    def predict(self):
        if self.learn.train:
            self.learn.preds = self.learn.model(self.learn.batch[0])
        else:
            # swap the augmentation index to the front
            image, _ , do_flips, _ = self.learn.batch
            image = torch.transpose(image, 0,1)
            num_augments = image.shape[0]
            # These will have shape (B,C,W,H,D) for outputs
            # and (B,C,W,H,D) or (B,W,H,D) for mask
            in_shape = list(image[0].shape)
            avg_outputs = torch.zeros(in_shape).to(def_device)
            for aug_idx in range(num_augments):
                # image_aug has shape (B, C,W,H,D) and mask_aug
                # has shape (B,C,W,H,D) or (B,W,H,D)
                image_aug = image[aug_idx]
                outputs_aug = self.learn.model(image_aug)
                for axis in range(len(image_aug.shape)-2):
                    outputs_aug = torch.flip(outputs_aug,dims = (axis+2,)) if do_flips[aug_idx][axis] else outputs_aug
                avg_outputs += outputs_aug
            avg_outputs /= num_augments
            self.learn.preds = avg_outputs
            
    def get_loss(self):
        self.learn.loss = self.learn.loss_func(self.learn.preds, self.learn.batch[1])
    def backward(self): self.learn.acc.backward(self.learn.loss)
    def step(self): self.learn.opt.step()
    def zero_grad(self): self.learn.opt.zero_grad()
    
class LRFinderCB(Callback):
    def __init__(self,gamma=1.3): fc.store_attr()
    
    def before_fit(self):
        self.sched = ExponentialLR(self.learn.opt,self.gamma)
        self.lrs,self.losses = [],[]
        self.min = math.inf
        
    def after_batch(self):
        if not self.learn.model.training: raise CancelEpochException()
        self.lrs.append(self.learn.opt.param_groups[0]['lr'])
        loss = to_cpu(self.learn.loss)
        self.losses.append(loss)
        if loss < self.min: self.min = loss
        if loss > self.min*3: raise CancelFitException()
        self.sched.step()
        
    def plot(self):
        plt.plot(self.lrs,self.losses)
        plt.xscale('log')
                            
    
# class ProgressCB(Callback):
#     order = MetricsCB.order+1
#     def __init__(self,plot=False): self.plot = plot
#     def before_fit(self):
#         self.learn.epochs = self.mbar = master_bar(self.learn.epochs)
#         if hasattr(self.learn,'metrics'): self.learn.metrics._log = self._log
#         self.losses = []
#     def _log(self,d): self.mbar.write(str(d))
#     def before_epoch(self): self.learn.dl = progress_bar(self.learn.dl, leave=False, parent = self.mbar)
#     def after_batch(self):
#         self.learn.dl.comment = f'{self.learn.loss:.4f}'
#         if self.plot and hasattr(self.learn, 'metrics') and self.learn.model.training:
#             self.losses.append(self.learn.loss.item())
#             self.mbar.update_graph( [[fc.L.range(self.losses),self.losses]] )