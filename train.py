import json
import numpy as np
import torch
import os

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from FOD.Trainer import Trainer
from FOD.dataset import AutoFocusDataset

with open('config.json', 'r') as f:
    config = json.load(f)
np.random.seed(config['General']['seed'])

list_data = config['Dataset']['paths']['list_datasets']

## train set
autofocus_datasets_train = []
for dataset_name in list_data:
    autofocus_datasets_train.append(AutoFocusDataset(config, dataset_name, 'train'))
train_data = ConcatDataset(autofocus_datasets_train)
train_dataloader = DataLoader(train_data, batch_size=config['General']['batch_size'], shuffle=True)

## validation set
autofocus_datasets_val = []
for dataset_name in list_data:
    autofocus_datasets_val.append(AutoFocusDataset(config, dataset_name, 'val'))
val_data = ConcatDataset(autofocus_datasets_val)
val_dataloader = DataLoader(val_data, batch_size=config['General']['batch_size'], shuffle=True)

trainer = Trainer(config)
if config['General']['pretained']:
    path_model = os.path.join(config['General']['path_model'], 'FocusOnDepth_{}.p'.format(config['General']['model_timm']))
    trainer.load_checkpoint(path_model)
trainer.train(train_dataloader, val_dataloader)
