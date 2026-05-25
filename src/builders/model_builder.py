from src.core import model_ECG_backbone
from . import generator


def get_network(args):
    gen = generator.NetGenerator(args)
    loader_name = args.backbone_name
    net = getattr(gen, loader_name)()
    return net


def build_ECG(config):
    backbone = get_network(config)
    config['backbone'] = backbone
    return model_ECG_backbone.SpaceTimeFactorizedViViT_ECG(**config)
