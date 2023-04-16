from torch.utils.data import DataLoader
from .dataset import VPRCDataset
from .transforms import get_transforms


def get_dataloaders(config):
    """
    Function for creating training and validation dataloaders
    :param config:
    :return:
    """
    print("Preparing train reader...")
    train_dataset = VPRCDataset(
        root=config.dataset.train_prefix, annotation_file=config.dataset.train_list,
        transforms=get_transforms(config, augmentation=config.dataset.augmentation.methods != 'none')
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        drop_last=False
    )
    print("Done.")

    print("Preparing test reader...")
    test_dataset = VPRCDataset(
        root=config.dataset.test_prefix, annotation_file=config.dataset.test_list,
        transforms=get_transforms(config, augmentation=False)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        drop_last=False
    )
    print("Done.")
    return train_loader, test_loader
