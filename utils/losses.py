import torch
import torch.nn as nn


class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        # batch equal to True means views all batch images as an entity and calculate loss
        # batch equal to False means calculate loss of every single image in batch and get their mean
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.00001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)

        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred.to(dtype=torch.float32), y_true)


class dice_focal_loss(nn.Module):

    def __init__(self):
        super(dice_focal_loss, self).__init__()
        self.focal_loss = nn.BCEWithLogitsLoss()
        self.binnary_dice = dice_loss()

    def __call__(self, scores, labels):
        diceloss = self.binnary_dice(torch.sigmoid(scores.clone()), labels)
        foclaloss = self.focal_loss(scores.clone(), labels)

        return diceloss, foclaloss


def FCCDN_loss_without_seg(scores, labels):
    # scores = change_pred
    # labels = binary_cd_labels
    scores = scores.squeeze(1) if len(scores.shape) > 3 else scores
    labels = labels.squeeze(1) if len(labels.shape) > 3 else labels
    # if len(scores.shape) > 3:
    #     scores = scores.squeeze(1)
    # if len(labels.shape) > 3:
    #     labels = labels.squeeze(1)
    """ for binary change detection task"""
    criterion_change = dice_focal_loss()

    # change loss
    diceloss, foclaloss = criterion_change(scores, labels)

    loss_change = diceloss + foclaloss

    return loss_change, diceloss, foclaloss

# 新增：多分类交叉熵损失（支持20类）
class CrossEntropyLoss2D(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(CrossEntropyLoss2D, self).__init__()
        self.weight = weight  # 类别权重，解决类别不平衡
        self.ignore_index = ignore_index  # 忽略的标签值（如背景）
        self.criterion = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='mean'
        )

    def forward(self, preds, labels):
        """
        preds: 模型输出 [B, 20, H, W]（未经过softmax的logits）
        labels: 标签 [B, H, W]（值为0~19的类别索引）
        """
        loss = self.criterion(preds, labels)
        return loss, torch.tensor(0.0, device=loss.device), torch.tensor(0.0, device=loss.device)
        # 后两个返回值为兼容原有代码结构（原损失函数返回三个值）
