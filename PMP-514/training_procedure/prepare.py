import torch
import torch.nn as nn
from importlib import import_module
import os.path as osp
from DataHelper.datasetHelper import DatasetHelper
from model.SAGE import GraphSAGE_DGL

from model.LASAGE_S import LASAGE_S

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        p = torch.exp(-ce)
        loss = (1 - p) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, base_loss, alpha=0.1, margin=1.0):
        super().__init__()
        self.base_loss = base_loss
        self.alpha = alpha
        self.margin = margin
        self.last_cls_loss  = 0.0
        self.last_rank_loss = 0.0

    def forward(self, logits, labels):
        base = self.base_loss(logits, labels)

        fraud_mask  = labels == 1
        benign_mask = labels == 0
        if fraud_mask.sum() == 0 or benign_mask.sum() == 0:
            self.last_cls_loss  = base.item()
            self.last_rank_loss = 0.0
            return base

        fraud_scores  = logits[fraud_mask,  1]
        benign_scores = logits[benign_mask, 1]

        diff = fraud_scores.unsqueeze(1) - benign_scores.unsqueeze(0)  # (n_pos, n_neg)
        ranking = torch.clamp(self.margin - diff, min=0).mean()

        self.last_cls_loss  = base.item()
        self.last_rank_loss = ranking.item()

        return base + self.alpha * ranking


def prepare_train(self, model, datasetHelper: DatasetHelper):
    config = self.config
    scheduler = None
    optimizer = getattr(torch.optim, config['optimizer'])(  params          = model.parameters(),
                                                            lr              = config['lr'] ,
                                                            weight_decay    = config.get('weight_decay', 0) )
    if config.get('lr_scheduler', False):
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['step_size'],gamma=config['gamma'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = config['resi'], min_lr=1e-3)
    weight = (1-datasetHelper.labels[datasetHelper.train_mask]).sum() / datasetHelper.labels[datasetHelper.train_mask].sum()
    w = torch.tensor([1., weight]).cuda() if config['weighted_loss'] else None
    if config.get('focal_loss', False):
        loss_func = FocalLoss(gamma=config.get('focal_gamma', 2.0), weight=w, reduction=config['reduction'])
    else:
        loss_func = nn.CrossEntropyLoss(weight=w, reduction=config['reduction'])
    if config.get('ranking_alpha', 0.0) > 0.0:
        loss_func = CombinedLoss(loss_func, alpha=config['ranking_alpha'], margin=config.get('ranking_margin', 1.0))
    return optimizer, loss_func, scheduler

def prepare_model(self, datasetHelper: DatasetHelper):
    config = self.config
    model_name = config['model_name']


    if model_name == 'LA-SAGE-S':
        mlp_act = config.get('mlp_activation', 'relu')
        if mlp_act == 'relu':
            mlp_activation = nn.ReLU(inplace=True)
        elif mlp_act == 'elu':
            mlp_activation = nn.ELU(inplace=True)

        model = LASAGE_S( in_size       = datasetHelper.feat_dim,
                          hid_size      = config['hid_dim'],
                          out_size      = datasetHelper.num_classes if not config['proj'] else config['hid_dim'],
                          num_layers    = config['n_layer'],
                          dropout       = config['dropout'],
                          proj          = config['proj'],
                          num_trans     = config['num_trans'],
                          out_proj_size = datasetHelper.num_classes,
                          agg           = config['agg'],
                          batch_size    = config['batch_size'],
                          num_relations = datasetHelper.num_relations,
                          mlp_activation= mlp_activation,
                          relation_agg  = config['relation_agg'],
                          feat_drop     = config['dropout']).cuda()

    return model


def init(self, datasetHelper: DatasetHelper):
    config = self.config
    model = self.prepare_model(datasetHelper)
    optimizer, loss_func, scheduler = self.prepare_train(model, datasetHelper)
    
    return model, optimizer, loss_func, scheduler