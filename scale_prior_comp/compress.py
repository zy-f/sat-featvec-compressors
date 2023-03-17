from scale_prior_comp.array_compressor import CompressionTrainWrapper, ArrayCompressor
import sys
if '..' not in sys.path:
    sys.path.append('..')
from train_utils import *

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import torchvision
import numpy as np

dummy = lambda x: x
metrics = [
    ScalarMetric('loss', input_names=['trn_loss'], compute_func=dummy),
    ScalarMetric('rate', input_names=['trn_rate'], compute_func=dummy),
    ScalarMetric('distortion', input_names=['trn_distortion'], compute_func=dummy),
    ScalarMetric('coder_loss', input_names=['trn_coder_loss'], compute_func=dummy),
]

def do_compress_train_loop(train_wrapper, dataloader, device, train=True):
    iters = 0
    outputs = []
    if train:
        train_wrapper.module.train()
    else:
        train_wrapper.module.eval()
    for batch in tqdm(dataloader):
        z, y = tuple(t.to(device) for t in batch)
        outputs.append(train_wrapper.step(z, y, train=train))
    cumul_metrics = np.array(outputs).mean(axis=0)
    prefix = 'trn' if train else 'val'
    return ( (f'{prefix}_loss',cumul_metrics[0]), \
             (f'{prefix}_rate',cumul_metrics[1]), \
             (f'{prefix}_distortion',cumul_metrics[2]), \
             (f'{prefix}_coder_loss', cumul_metrics[1]))
    
def train_compressor(compressor, train_dataset, device, hparams=None, get_logs=False):
    if hparams is None:
        hparams = DotDict(
            z_dim=512,
            lmbda=4e-2,
            lr=2e-1,
            lr_step=5,
            n_epochs=10
        )
    elif isinstance(hparams, dict):
        hparams = DotDict(**hparams)
    train_wrapper = CompressionTrainWrapper(compressor, hparams)
    train_wrapper.to(device)
    train_wrapper.configure_optimizers()
    
    dummy = lambda x: x
    metrics = [
        ScalarMetric('loss', input_names=['trn_loss'], compute_func=dummy),
        ScalarMetric('rate', input_names=['trn_rate'], compute_func=dummy),
        ScalarMetric('distortion', input_names=['trn_distortion'], compute_func=dummy),
        ScalarMetric('coder_loss', input_names=['trn_coder_loss'], compute_func=dummy),
    ]

    log_outputs = LogOutput(*metrics)
    train_dl = DataLoader(train_dataset, shuffle=True, num_workers=2, batch_size=128, pin_memory=True)
    
    for epoch in range(hparams.n_epochs):
        print(f"EPOCH {epoch+1}/{hparams.n_epochs}:")
        cur_epoch_data = {}
        train_outputs = do_compress_train_loop(train_wrapper, dataloader=train_dl, device=device, train=True)
        for name, value in train_outputs:
            cur_epoch_data[name] = value
        log_outputs.update(cur_epoch_data)
        print(log_outputs.report())
    log_outputs.plot()
    return compressor, log_outputs if get_logs else compressor

def create_compressor(hparams):
    return ArrayCompressor(hparams['z_dim'])
    
    
def baseline(): # NOT FIXED
    import clip
    import torch
    from torchvision.datasets import CIFAR10, STL10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # pretrained CLIP
    pretrained, preprocess = clip.load("ViT-B/32", device)

    # train data
    cifar = CIFAR10('/atlas2/u/clcp/mae_compression/data', download=False, train=True, transform=preprocess)
    def clip_featurize_data(dataset, device, pretrained):
        """Featurize a dataset using the pretrained CLIP model."""
        with torch.no_grad():
            Z, Y = [], []
            for x, y in tqdm(DataLoader(dataset, batch_size=128, num_workers=2)):
                Z += [pretrained.encode_image(x.to(device).half()).cpu().numpy()]
                Y += [y.cpu().numpy()]
        return np.concatenate(Z), np.concatenate(Y)
    
    Z_cifar, Y_cifar = clip_featurize_data(cifar, device, pretrained)
    train_dataset = torch.utils.data.dataset.TensorDataset(torch.from_numpy(Z_cifar), torch.from_numpy(Y_cifar))
    
    hparams = DotDict(
        z_dim=512,
        lmbda=4e-2,
        lr=2e-1,
        lr_step=5,
        n_epochs=10
    )
    
    train_compressor(train_dataset, device, hparams=hparams)
    
    
    
if __name__ == '__main__':
    
    baseline()
    
