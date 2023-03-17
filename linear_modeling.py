import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_utils import *
import wandb
from contextlib import nullcontext
from tqdm import tqdm

# random seeding
SEED = 269
np.random.seed(SEED)
np_rng = np.random.default_rng(seed=SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

class MultiLayerMLP(nn.Module):
    def __init__(self, dims, dropouts=None):
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts]*(len(dims) - 2)
        assert dropouts is None or len(dropouts) == len(dims) - 2
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                if dropouts is not None:
                    layers.append(nn.Dropout(p=dropouts[i]))
                layers.append(nn.ReLU())
        self.arch = nn.Sequential(*layers)

    def forward(self, x):
        return self.arch(x)
    
# data loading
def get_data_tuple():
    print('loading data')
    data_path = 'data/raw/geoSSL-fmow.pt'
    data = torch.load(data_path)

    X_tr = data['train']['Z']
    y_tr = data['train']['y']
    X_te = data['test']['Z']
    y_te = data['test']['y']
    return X_tr, y_tr, X_te, y_te

default_cfg = DotDict(
        features_base='geoSSL-fmow',
        compress_type='raw',
        n_epochs=100,
        lr=1e-4,
        wd=0,
        bsz=256,
        device='cuda:0'
    )
# define training stuff
def run_training(X_tr, y_tr, X_te, y_te, run_name=None, cfg=default_cfg, model=None):
    d_feat = X_tr.shape[-1]
    print('prepping for training')

    train_dataset = torch.utils.data.dataset.TensorDataset(X_tr, y_tr)
    eval_dataset = torch.utils.data.dataset.TensorDataset(X_te, y_te)

    train_dl = DataLoader(train_dataset, batch_size=cfg.bsz, shuffle=True, num_workers=2)
    eval_dl = DataLoader(eval_dataset, batch_size=cfg.bsz, num_workers=2)

    if model is None:
        model = nn.Linear(d_feat, 62)
    model.to(cfg.device)

    # torch.optim.SGD(fc.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    loss_func = nn.CrossEntropyLoss().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.wd)
    scheduler = None#torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3, total_steps=len(train_dl)*n_epochs)
    cfg.model_str = str(model)
    cfg.optimizer = str(optimizer)
    cfg.scheduler = str(scheduler)

    dummy = lambda x: x
    get_acc = lambda preds, ys: np.mean(preds == ys)
    metrics = [
        ScalarMetric('train loss', input_names=['trn_loss'], compute_func=dummy),
        ScalarMetric('train acc', input_names=['trn_pred', 'trn_y'], compute_func=get_acc),
        ScalarMetric('val loss', input_names=['val_loss'], compute_func=dummy),
        ScalarMetric('val acc', input_names=['val_pred', 'val_y'], compute_func=get_acc),
    ]
    log_outputs = LogOutput(*metrics, eval_metric=metrics[-1])
    print('starting training')
    time_id = current_pst_str()
    unique_run_name = None if run_name is None else f'{run_name}-{time_id}'
    
    with wandb.init(name=unique_run_name, project='ee269_final', \
                    entity='cpolzak', config=cfg.__dict__) \
        if unique_run_name is not None else nullcontext() \
    as run:
        do_save = run is not None
        if not do_save:
            print('WARNING: results will not be saved')
        eval_best = 0
        TOTAL_STEPS = cfg.n_epochs * (len(train_dl) + len(eval_dl))
        bar_fmt = '{desc}{percentage:.0f}%|{bar}|{n}/{total_fmt}'
        with tqdm(total=TOTAL_STEPS, desc=f'best={eval_best:3f}', position=0, leave=True, bar_format=bar_fmt) as pbar:
            for epoch in range(cfg.n_epochs):
                pbar.set_description(f'best={eval_best:.3f}')
                cur_epoch_data = {}
                train_outputs = do_epoch_loop(train_dl, model, loss_func, optimizer=optimizer, \
                                              scheduler=scheduler, device=cfg.device, \
                                              pbar=pbar)
                for name, value in train_outputs:
                    cur_epoch_data[name] = value
                val_outputs = do_epoch_loop(eval_dl, model, loss_func, \
                                            device=cfg.device, pbar=pbar)
                for name, value in val_outputs:
                    cur_epoch_data[name] = value
                log_outputs.update(cur_epoch_data)
                if do_save:
                    wandb.log(log_outputs.pass_log())
                if log_outputs.eval_metric.last() > eval_best:
                    eval_best = log_outputs.eval_metric.last()
                    if do_save:
                        torch.save(model.state_dict(), f"ckpts/{unique_run_name}.pt")
                # print(log_outputs.report())

        if do_save:
            wandb.log({f'best_{log_outputs.eval_metric.name}': eval_best})

if __name__ == '__main__':
    data_tuple = get_data_tuple()
    run_training(*data_tuple)