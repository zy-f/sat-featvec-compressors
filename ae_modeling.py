import torch
import torch.nn as nn
from linear_modeling import MultiLayerMLP
from torch.utils.data import DataLoader
from train_utils import *
import wandb
from contextlib import nullcontext
from tqdm import tqdm

class MLPAutoEncoder(nn.Module):
    def __init__(self, dims, dropouts):
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts]*(len(dims) - 2)
        assert dropouts is None or len(dropouts) == len(dims) - 2
        self.encoder = MultiLayerMLP(dims, dropouts=dropouts)
        self.bottleneck_activation = nn.Sigmoid()
        # decoder = inverted encoder
        self.decoder = MultiLayerMLP(dims[::-1], dropouts=dropouts[::-1])
    
    def encode(self, x):
        return self.bottleneck_activation(self.encoder(x))
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)

default_cfg = DotDict(
        n_epochs=50,
        lr=1e-4,
        wd=0,
        bsz=256,
        device='cuda:0'
    )

def do_ae_epoch_loop(dl, model, loss_func, optimizer=None, scheduler=None, device='cpu', pbar=None):
    do_train = optimizer is not None
    iters = 0
    epoch_loss = 0
    model.train() if do_train else model.eval()
    for i, (x, x_true) in enumerate(dl):
        if do_train:
            optimizer.zero_grad()
        x = x.to(device)
        x_true = x_true.to(device)
        x_hat = model(x)
        loss = loss_func(x_hat, x_true)
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
    prefix = 'trn' if do_train else 'val'
    return ((f'{prefix}_loss', epoch_loss),)

def run_ae_training(X_tr, X_te, model, run_name=None, cfg=default_cfg):
    d_feat = X_tr.shape[-1]
    print('prepping for training')

    train_dataset = torch.utils.data.dataset.TensorDataset(X_tr, X_tr)
    eval_dataset = torch.utils.data.dataset.TensorDataset(X_te, X_te)

    train_dl = DataLoader(train_dataset, batch_size=cfg.bsz, shuffle=True, num_workers=8)
    eval_dl = DataLoader(eval_dataset, batch_size=cfg.bsz, num_workers=8)

    model.to(cfg.device)

    loss_func = nn.MSELoss().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.wd)
    scheduler = None#torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3, total_steps=len(train_dl)*n_epochs)
    cfg.model_str = str(model)
    cfg.optimizer = str(optimizer)
    cfg.scheduler = str(scheduler)

    dummy = lambda x: x
    metrics = [
        ScalarMetric('train loss', input_names=['trn_loss'], compute_func=dummy),
        ScalarMetric('val loss', input_names=['val_loss'], compute_func=dummy),
    ]
    log_outputs = LogOutput(*metrics, eval_metric=metrics[-1])
    print('starting training')
    time_id = current_pst_str()
    unique_run_name = None if run_name is None else run_name#f'{run_name}-{time_id}'
    
    with wandb.init(name=unique_run_name, project='ee269_final', \
                    entity='cpolzak', config=cfg.__dict__) \
        if unique_run_name is not None else nullcontext() \
    as run:
        do_save = run is not None
        if not do_save:
            print('WARNING: results will not be saved')
        eval_best = np.inf
        TOTAL_STEPS = cfg.n_epochs * (len(train_dl) + len(eval_dl))
        bar_fmt = '{desc}{percentage:.0f}%|{bar}|{n}/{total_fmt}'
        with tqdm(total=TOTAL_STEPS, desc=f'best={eval_best:3f}', position=0, leave=True, bar_format=bar_fmt) as pbar:
            for epoch in range(cfg.n_epochs):
                pbar.set_description(f'best={eval_best:.3f}')
                cur_epoch_data = {}
                train_outputs = do_ae_epoch_loop(train_dl, model, loss_func, optimizer=optimizer, \
                                                 scheduler=scheduler, device=cfg.device, \
                                                 pbar=pbar)
                for name, value in train_outputs:
                    cur_epoch_data[name] = value
                val_outputs = do_ae_epoch_loop(eval_dl, model, loss_func, \
                                               device=cfg.device, pbar=pbar)
                for name, value in val_outputs:
                    cur_epoch_data[name] = value
                log_outputs.update(cur_epoch_data)
                if do_save:
                    wandb.log(log_outputs.pass_log())
                if log_outputs.eval_metric.last() < eval_best: # b/c lower loss is better
                    eval_best = log_outputs.eval_metric.last()
                    if do_save:
                        torch.save(model.state_dict(), f"compress_models/{unique_run_name}.pt")
                # print(log_outputs.report())

        if do_save:
            wandb.log({f'best_{log_outputs.eval_metric.name}': eval_best})