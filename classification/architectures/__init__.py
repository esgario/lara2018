import torch

from .resnet import resnet34, resnet50, resnet101
from .googlenet import googlenet
from .shallow import shallow
from .vgg import vgg16
from .alexnet import alexnet
from .mobilenetv2 import mobilenet_v2


MODELS = {
    "shallow": shallow,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "alexnet": alexnet,
    "googlenet": googlenet,
    "vgg16": vgg16,
    "mobilenet_v2": mobilenet_v2,
}


def cnn_model(model_name, pretrained=False, num_classes=(5, 5), weights_path=None):
    model = MODELS[model_name](pretrained=pretrained, num_classes=num_classes)

    if weights_path:
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
        except Exception:
            raise Exception("Error loading weights. You must train the model first.")

    if torch.cuda.is_available():
        model.cuda()

    return model
