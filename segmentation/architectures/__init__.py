import os
import torch
import torch.nn as nn

from .unet import UNetWithResnet50Encoder
from .pspnet import PSPNet


## Declare models
MODELS = {
    "pspsqueezenet": {
        "class": PSPNet,
        "params": dict(
            sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend="squeezenet"
        )
    },
    "pspdensenet": {
        "class": PSPNet,
        "params": dict(
            sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend="densenet"
        )
    },
    "pspresnet18": {
        "class": PSPNet,
        "params": dict(
            sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend="resnet18"
        )
    },
    "pspresnet34": {
        "class": PSPNet,
        "params": dict(
            sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend="resnet34"
        )
    },
    "pspresnet50": {
        "class": PSPNet,
        "params": dict(
            sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend="resnet50"
        )
    },
    "pspresnet101": {
        "class": PSPNet,
        "params": dict(
            sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend="resnet101"
        )
    },
    "pspresnet152": {
        "class": PSPNet,
        "params": dict(
            sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend="resnet152"
        )
    },
    "unetresnet50": {
        "class": UNetWithResnet50Encoder,
        "params": {}
    }
}


# Building network
def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()

    # Instantiate network model
    ModelClass = MODELS[backend]["class"]
    net = ModelClass(n_classes=3, **MODELS[backend]["params"])
    
    # Parallel training
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    # Load a pretrained network
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split("_")
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))

    # Sending model to GPU
    if torch.cuda.is_available():
        net = net.cuda()

    return net, epoch


def load_model(backend, weights_path):
    model, _ = build_network(None, backend)

    if weights_path:
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
        except Exception:
            raise Exception("Error loading weights. You must train the model first.")

    if torch.cuda.is_available():
        model.cuda()

    return model
