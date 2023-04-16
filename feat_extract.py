import os
import os.path as osp
import random
import yaml
import torch
import numpy as np
from tqdm import tqdm

from data import get_dataloaders
from models import load_model
from utils import convert_dict_to_tuple

CONFIG_FILE = 'config/inception.yml'


if __name__ == '__main__':
    with open(CONFIG_FILE) as f:
        data = yaml.safe_load(f)

    config = convert_dict_to_tuple(data)

    seed = config.dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    outdir = osp.join('./datasets', f'VPRC2023_{config.name}')
    print("output directory: {}".format(outdir))
    if not os.path.exists(osp.join(outdir, 'train')):
        os.makedirs(osp.join(outdir, 'train'))
    if not os.path.exists(osp.join(outdir, 'test')):
        os.makedirs(osp.join(outdir, 'test'))

    train_loader, test_loader = get_dataloaders(config)

    print("Loading model...")
    model = load_model(config)
    model.eval()
    print("Done!")

    with torch.no_grad():
        for epoch in range(config.dataset.augmentation.epoch):
            train_iter = tqdm(train_loader, desc='Train Set Feature Extraction', dynamic_ncols=True, position=1)
            for step, batch in enumerate(train_iter):
                x = batch['image']
                y = batch['label']
                names = batch['name']
                out = model(x.cuda().to(memory_format=torch.contiguous_format)).cpu().numpy()

                for i in range(len(x)):
                    np.save(os.path.join(outdir, 'train', f"{y[i]:04d}_{epoch}_{names[i]}"), out[i])

        test_iter = tqdm(test_loader, desc='Test Set Feature Extraction', dynamic_ncols=True, position=1)
        for step, batch in enumerate(test_iter):
            x = batch['image']
            y = batch['label']
            names = batch['name']
            out = model(x.cuda().to(memory_format=torch.contiguous_format)).cpu().numpy()

            for i in range(len(x)):
                np.save(os.path.join(outdir, 'test', f"{y[i]:04d}_{names[i]}"), out[i])

