import numpy as np
import torch
data_path = 'data/raw/geoSSL-fmow.pt'
data = torch.load(data_path)

X_tr = data['train']['Z']
y_tr = data['train']['y']
X_te = data['test']['Z']
y_te = data['test']['y']

print(X_tr.min(), X_tr.max())
print(X_te.min(), X_te.max())