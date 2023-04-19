import torchvision.transforms as transforms


def get_transforms(config, augmentation=False):
    transform_list = []

    if augmentation:
        if 'rotate' in config.dataset.augmentation.methods:
            transform_list.append(transforms.RandomRotation(config.dataset.augmentation.rotate.degree))

        if 'perspective' in config.dataset.augmentation.methods:
            transform_list.append(
                transforms.RandomPerspective(
                    config.dataset.augmentation.perspective.distortion_scale,
                    config.dataset.augmentation.perspective.p
                ))

        if 'crop' in config.dataset.augmentation.methods:
            transform_list.append(
                transforms.RandomResizedCrop(
                    (config.dataset.input_size, config.dataset.input_size),
                    scale=config.dataset.augmentation.crop.scale
                ))

        if 'flip' in config.dataset.augmentation.methods:
            transform_list.append(transforms.RandomHorizontalFlip())

        if 'color' in config.dataset.augmentation.methods:
            transform_list.append(
                transforms.ColorJitter(
                    config.dataset.augmentation.color.brightness,
                    config.dataset.augmentation.color.contrast,
                    config.dataset.augmentation.color.saturation,
                    config.dataset.augmentation.color.hue,
                ))
    else:
        transform_list.append(transforms.Resize((config.dataset.input_size, config.dataset.input_size)))

    transform_list.append(transforms.ToTensor())
    if config.model.arch.startswith('resnet') or config.model.arch == 'inception_v3':
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(transform_list)
