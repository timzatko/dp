from src.model.cnn_3d import cnn_3d
from src.model.deep_3d import deep_3d
from src.model.res_net import res_net


def create_model(model_type, options):
    if model_type == 'ResNet18':
        return res_net(is_3D=False, **options)
    elif model_type == '3DResNet18':
        return res_net(is_3D=True, **options)
    elif model_type == '3DResNetCustom':
        return deep_3d(**options)
    else:
        return cnn_3d(**options)
