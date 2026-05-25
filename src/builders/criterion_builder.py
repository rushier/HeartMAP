import torch.nn as nn
from src.core.criterion import sample_WeightedBCELoss, LvAttnLoss, FrameAttnLoss, WeightedSumMSELoss, HybridLoss, HybridLoss_inter
from monai.losses import DiceLoss as diceloss
from monai.networks.utils import one_hot


def build(config, mode):
    print(config)
    criterion = dict()

    criterion["regression"] = nn.MSELoss()
    criterion["aux_classification"] = nn.CrossEntropyLoss()
    criterion["classification"] = nn.CrossEntropyLoss()
    criterion["bce"] = nn.BCEWithLogitsLoss()
    criterion["weighted_bce"] = weighted_BCELoss()
    criterion["sample_weighted_bce"] = sample_WeightedBCELoss()
    criterion["spatial_location"] = LvAttnLoss(
        config["frame_size"], config["patches"][0], config["n_sampled_frames"]
    )
    criterion["temporal_location"] = FrameAttnLoss()
    criterion["weighted_mse"] = WeightedSumMSELoss()
    criterion["dice_ce"] = DICE_CELoss(int(config["seg_classes"]))
    criterion["mse_correlation"] = HybridLoss()
    criterion["mse_correlation_inter"] = HybridLoss_inter()

    return criterion

class weighted_BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        # 计算未加权的损失
        loss = self.loss(logits, targets)
        # 创建权重张量，阳性样本的权重为3，阴性样本的权重为1
        weights = targets.detach().clone() + 1
        # 应用权重
        weighted_loss = loss * weights
        loss = weighted_loss.sum()/weights.sum()
        # 计算加权损失的平均值
        return loss

class DICE_CELoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = diceloss(softmax=True)

    def forward(self, logits, targets):
        targets = targets.reshape(-1, 1, 16, 224, 224)
        mask_onehot = one_hot(targets, num_classes=self.num_classes)
        ce_loss = self.celoss(logits, mask_onehot)
        dice_loss = self.diceloss(logits, mask_onehot)
        loss = ce_loss + dice_loss
        return loss


