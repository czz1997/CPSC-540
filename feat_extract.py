import os
import os.path as osp
import random
import yaml
import torch
import numpy as np
from tqdm import tqdm

from sklearn.decomposition import PCA
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
    if config.model.arch == 'PCA':
        model = PCA(whiten=True)
    else:
        model = load_model(config)
        model.eval()
    print("Done!")

    if config.model.arch == 'PCA':
        train_iter = tqdm(train_loader, desc='Train Set Feature Extraction', dynamic_ncols=True, position=1)
        X_train, y_train, names_train = [], [], []
        for step, batch in enumerate(train_iter):
            X_train.append(batch['image'].numpy().reshape((len(batch['image']), -1)))
            y_train.append(batch['label'].numpy()[:, None])
            names_train += batch['name']
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0).squeeze(1)
        print(X_train.shape)
        print(y_train.shape)

        model.fit(X_train)
        X_train = model.transform(X_train)
        print(X_train.shape)
        for i in range(len(X_train)):
            np.save(os.path.join(outdir, 'train', f"{y_train[i]:04d}_0_{names_train[i]}"), X_train[i])

        test_iter = tqdm(test_loader, desc='Test Set Feature Extraction', dynamic_ncols=True, position=1)
        X_test, y_test, names_test = [], [], []
        for step, batch in enumerate(test_iter):
            X_test.append(batch['image'].numpy().reshape((len(batch['image']), -1)))
            y_test.append(batch['label'].numpy()[:, None])
            names_test += batch['name']
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0).squeeze(1)

        X_test = model.transform(X_test)
        for i in range(len(X_test)):
            np.save(os.path.join(outdir, 'test', f"{y_test[i]:04d}_0_{names_test[i]}"), X_test[i])
    else:
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

