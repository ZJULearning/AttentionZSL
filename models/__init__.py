from .lfgaa_vgg19 import *
from .lfgaa_resnet101 import *
from .lfgaa_inception_v3 import *

def load_model(model_name, **kwargs):
    if model_name == "VGG19":
        fnet = LFGAAVGG19(**kwargs)
        im_size = (224, 224)
    elif model_name == "ResNet101":
        fnet = LFGAAResNet101(**kwargs)
        im_size = (224, 224)
    elif model_name == "GoogleNet":
        fnet = LFGAAGoogleNet(**kwargs)
        im_size = (299, 299)
    else:
        raise NotImplementedError
    return fnet, im_size
