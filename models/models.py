from torch import nn, hub
from torchvision import models


def load_model(config):
    """
    The function of loading a model by name from a configuration file
    :param config:
    :return:
    """
    arch = config.model.arch
    if arch.startswith('resnet') or arch == 'inception_v3':
        model = hub.load('pytorch/vision:v0.10.0', arch, pretrained=True)
        # model = models.__dict__[arch](weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
    elif arch.startswith('vgg'):
        raise NotImplementedError
    else:
        raise Exception('model type is not supported:', arch)
    model.to('cuda')
    return model
