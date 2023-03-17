__author__ = 'clcp'
__version__ = '1.0'

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import datetime
import pytz

### GENERAL

def current_pst_str():
    return datetime.datetime.now(pytz.timezone('US/Pacific')).strftime('%y_%m_%d-%H_%M')


class DotDict:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, k):
        return getattr(self, k)

    def __len__(self):
        return len(self.__dict__)
    
    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __getstate__(self): # for pickling
        return self.__dict__

    def __setstate__(self, d): # for pickling
        self.__dict__ = d

## METRICS
class ScalarMetric:
    def __init__(self, name, input_names, compute_func, report_decimals=4):
        self.name = name
        self.input_names = input_names
        self.compute_func = compute_func
        self.data = []
        self.dec = report_decimals

    def report(self):
        out = f"{self.name}: "
        if len(self.data) < 1:
            out += '-.--'
        else:
            out += f"{round(self.data[-1], self.dec)}"
            if len(self.data) < 2:
                out += " (-.--)"
            else:
                out += f" ({'%+f' % round(self.data[-1] - self.data[-2], self.dec)})"
        return out

    def update(self, *inputs):
        self.data.append(self.compute_func(*inputs))
    
    def last(self):
        return self.data[-1] if len(self.data) > 0 else None

### PLOTTING
class AxisWrapper:
    def __init__(self, ax_obj, ncols):
        self.ax = ax_obj if isinstance(ax_obj, np.ndarray) else np.array([ax_obj])
        self.c = ncols

    def __getitem__(self, i):
        if self.c == 1:
            return self.ax[i]
        return self.ax[i//self.c, i%self.c]

def get_rect_subplots(n, figscale=5):
    ncols = int(np.sqrt(n))
    nrows = int(np.ceil(n/ncols))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*figscale, nrows*figscale))
    fig.set_facecolor('white')
    return fig, AxisWrapper(ax, ncols)

### LOGGING
class LogOutput:
    def __init__(self, *metrics, eval_metric=None):
        self.metrics = metrics
        self.eval_metric = eval_metric
        self.last_input = None

    def update(self, inputs):
        self.last_input = inputs
        for metric in self.metrics:
            metric.update(*[inputs[in_name] for in_name in metric.input_names])
            
    def pass_log(self): 
        '''
        use for wandb/other logger integration
        passes last computed metrics as {metric_name: value} dict
        '''
        return {m.name: m.last() for m in self.metrics}

    def report(self, joiner='\n'):
        return joiner.join([m.report() for m in self.metrics])

    def plot(self):
        fig, ax = get_rect_subplots(len(self.metrics))
        fig.set_facecolor('lightgrey')
        for i, m in enumerate(self.metrics):
            ax[i].plot(m.data)
            ax[i].set_xlabel('Epoch')
            ax[i].set_ylabel(m.name)

### TRAINING (GENERIC)
def do_epoch_loop(dl, model, loss_func, optimizer=None, scheduler=None, device='cpu', pbar=None):
    do_train = optimizer is not None
    iters = 0
    epoch_loss = 0
    preds = []
    ys = []
    model.train() if do_train else model.eval()
    for i,(x,y) in enumerate(dl):
        if do_train:
            optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        output = out.detach().cpu()
        preds.append(torch.argmax(output, dim=-1))
        ys.append(y.detach().cpu())
        loss = loss_func(out, y)
        if do_train:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        epoch_loss += loss.item()
        iters += 1
        if pbar is not None:
            pbar.update(1)
    epoch_loss /= iters
    ys = torch.cat(ys, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()
    prefix = 'trn' if do_train else 'val'
    return ( (f'{prefix}_loss', epoch_loss), (f'{prefix}_y', ys), (f'{prefix}_pred', preds) )
