from torch import nn
import torch
 
def get_full_model(model_name = 'deeplabv3_plus_resnet', loss_name = 'DiceLoss2', n_classes=5, alpha=None, if_closs=0, pretrained=False):
    if loss_name == 'CombinedLoss':
        from .loss_funs import CombinedLoss
        loss = CombinedLoss(alpha=alpha, if_closs=if_closs)
        
    if model_name == 'ResNetC1':
        from .my_resnet import ResNetC1
        net = ResNetC1()
    return net, loss
