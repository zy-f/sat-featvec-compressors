import numpy as np
import torch
import os

SAVE_PATH = 'data/raw/geoSSL-ucmerced.pt'
data_path = '/atlas2/u/xiluo/compression_data/UCMerced_rgb_feature_vecs'
assert not os.path.exists(SAVE_PATH)
mode = 'raw'
features = '/atlas2/u/xiluo/compression_data/UCMerced_rgb_feature_vecs/*_features.npz'
if features.endswith('.npz'):
    load = np.load
    to_tensor = torch.from_numpy
elif features.endswith('.pt'):
    load = torch.load
    to_tensor = lambda x: x
train_data = load(f"{features}".replace('*', 'train'))
test_data = load(f"{features}".replace('*', 'val'))
print(to_tensor(test_data['Z']).dtype)
print(to_tensor(test_data['y']).dtype)
data = {
    'train':{
        'Z': to_tensor(train_data['Z']),
        'y': to_tensor(train_data['y'])
    },
    'val': {
        'Z': to_tensor(test_data['Z']),
        'y': to_tensor(test_data['y'])
    }
}
print('saving...')
torch.save(data, SAVE_PATH)