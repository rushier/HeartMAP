from src.core.model import SpaceTimeFactorizedViViT
from src.core.model_ECG import SpaceTimeFactorizedViViT_ECG
from src.core import model_single_ECG, model_ECG_backbone, model_backbone, model_LgeValue_ECG_backbone, model_LgeValue_ECG_backbone_seg, model_LgeValue_ECG, model_lge_seg
import torch
from . import generator

def get_network(args):
    gen = generator.NetGenerator(args)
    loader_name = args.backbone_name
    net = getattr(gen, loader_name)()
    return net


def build(config):
    if config.backbone_name:
        backbone = get_network(config)
        config['backbone']=backbone
        if config['vid_seq_len'] == 1:
            return model_backbone.single_view_net(**config)
        else:
            return model_backbone.SpaceTimeFactorizedViViT(**config)
    else:
        config.pop('backbone_name')
        return SpaceTimeFactorizedViViT(**config)

def build_ECG(config):
    if config.backbone_name:
        backbone = get_network(config)
        config['backbone']=backbone
        return model_ECG_backbone.SpaceTimeFactorizedViViT_ECG(**config)
    else:
        config.pop('backbone_name')
        return SpaceTimeFactorizedViViT_ECG(**config)

def build_lge_value(config):
    if config.backbone_name:
        backbone = get_network(config)
        config['backbone']=backbone
        return model_LgeValue_ECG_backbone.SpaceTimeFactorizedViViT_ECG(**config)
    else:
        config.pop('backbone_name')
        return model_LgeValue_ECG.SpaceTimeFactorizedViViT_ECG(**config)

def build_lge_value_seg(config):
    if config.backbone_name:
        backbone = get_network(config)
        config['backbone']=backbone
        config.pop('seg_classes')
        return model_LgeValue_ECG_backbone_seg.SpaceTimeFactorizedViViT_ECG(**config)

def build_lge_value_seg_woPerfusion(config):
    if config.backbone_name:
        backbone = get_network(config)
        config['backbone']=backbone
        config.pop('seg_classes')
        return model_lge_seg.SpaceTimeFactorizedViViT_ECG(**config)






def build_single_ECG(config):
    config.pop('backbone_name')
    return model_single_ECG.SpaceTimeFactorizedViViT_ECG(**config)


# def build_ECG_backbone(config):
#     backbone = get_network(config)
#     config['backbone']=backbone
#     return model_ECG_backbone.SpaceTimeFactorizedViViT_ECG(**config)
