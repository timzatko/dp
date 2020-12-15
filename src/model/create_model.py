from src.model.cnn_3d import cnn_3d
from src.model.res_net_50_v2 import res_net_50_v2
from src.model.vgg_16 import vgg_16
from src.model.res_net import res_net


def create_model(model_type, options):
    if model_type == 'ResNet18':
        return res_net(**options)
    elif model_type == 'ResNet50V2':
        return res_net_50_v2(**options)
    elif model_type == 'VGG16':
        return vgg_16(**options)
    else:
        return cnn_3d(**options)
