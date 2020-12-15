from src.model.cnn_3d import cnn_3d
from src.model.res_net import res_net


def create_model(model_type, options):
    if model_type == 'ResNet50V2':
        return res_net(**options)
    else:
        return cnn_3d(**options)
