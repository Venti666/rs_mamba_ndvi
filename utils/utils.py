from pathlib import Path
import time
import numpy as np
import torch.nn.functional as F
from utils.path_hyperparameter import ph
import torch
import logging
from tqdm import tqdm
import wandb


def save_model(model, path, epoch, mode, optimizer=None):
    Path(path).mkdir(parents=True, exist_ok=True)
    localtime = time.asctime(time.localtime(time.time()))
    if mode == 'checkpoint':
        state_dict = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state_dict, f"{path}checkpoint_epoch{epoch}_{localtime}.pth")
    else:
        torch.save(model.state_dict(), f"{path}best_{mode}_epoch{epoch}_{localtime}.pth")
    logging.info(f'best {mode} model {epoch} saved at {localtime}!')


def train_val_test(
        mode, dataset_name,
        dataloader, device, log_wandb, net, optimizer, total_step,
        lr, criterion, metric_collection, to_pilimg, epoch,
        warmup_lr=None, grad_scaler=None,
        best_metrics=None, checkpoint_path=None,
        best_f1score_model_path=None, best_loss_model_path=None, non_improved_epoch=None,
        visualize_fn=None
):
    assert mode in ['train', 'val'], 'mode should be train or val'
    epoch_loss = 0
    net.train() if mode == 'train' else net.eval()
    logging.info(f'SET model mode to {mode}!')
    batch_iter = 0
    tbar = tqdm(dataloader)
    n_iter = len(dataloader)
    sample_batch = np.random.randint(low=0, high=n_iter)

    for i, (image, labels, name) in enumerate(tbar):
        tbar.set_description(f"epoch {epoch} info {batch_iter} - {batch_iter + ph.batch_size}")
        batch_iter += ph.batch_size
        total_step += 1

        if mode == 'train':
            optimizer.zero_grad()
            if total_step < ph.warm_up_step:
                for g in optimizer.param_groups:
                    g['lr'] = warmup_lr[total_step]

        # 数据预处理：图像5通道，标签为20类索引（0~19）
        image = image.float().to(device)
        labels = labels.long().to(device)  # 交叉熵要求标签为long类型

        # 下采样（图像用双线性，标签用最近邻避免类别值错误）
        b, c, h, w = image.shape
        image = F.interpolate(
            image, size=(h // ph.downsample_raito, w // ph.downsample_raito),
            mode='bilinear', align_corners=False
        )
        labels = F.interpolate(
            labels.unsqueeze(1).float(),
            size=(h // ph.downsample_raito, w // ph.downsample_raito),
            mode='nearest'
        ).squeeze(1).long()  # 保持标签为整数

        # 图像分块
        crop_size = ph.image_size
        image_patches = image.unfold(2, crop_size, crop_size).unfold(3, crop_size, crop_size)
        B, C, new_H, new_W, _, _ = image_patches.size()
        image = image_patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, crop_size, crop_size).contiguous()

        labels_patches = labels.unfold(1, crop_size, crop_size).unfold(2, crop_size, crop_size)
        labels = labels_patches.reshape(-1, crop_size, crop_size).contiguous()

        # 前向传播与损失计算（交叉熵）
        if mode == 'train':
            with torch.cuda.amp.autocast():
                preds = net(image)  # [B, 20, H, W]（logits）
                loss_total, _, _ = criterion(preds, labels)  # 交叉熵损失
            cd_loss = loss_total.mean()
            grad_scaler.scale(cd_loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1, norm_type=2)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            preds = net(image)
            loss_total, _, _ = criterion(preds, labels)
            cd_loss = loss_total.mean()

        epoch_loss += cd_loss
        
        # 多分类预测：取概率最大的类别（0~19）
        pred_classes = torch.argmax(preds, dim=1)  # [B, H, W]

        # 日志图像采样
        if i == sample_batch:
            sample_index = np.random.randint(low=0, high=image.shape[0])
            t1_img_log = torch.round(image[sample_index]).cpu().clone().float()
            label_log = labels[sample_index].cpu().clone().float()  # 类别索引
            pred_log = pred_classes[sample_index].cpu().clone().float()  # 预测类别

            if visualize_fn is not None:
                t1_img_log = visualize_fn(t1_img_log)  # 转为3通道可视化

        # 计算指标（用softmax输出概率分布）
        batch_metrics = metric_collection.forward(
            F.softmax(preds, dim=1).float(),  # [B, 20, H, W] 概率分布
            labels
        )

        # 实时日志
        log_wandb.log({
            f'{mode} loss': cd_loss,
            f'{mode} accuracy': batch_metrics['accuracy'],
            f'{mode} precision': batch_metrics['precision'],
            f'{mode} recall': batch_metrics['recall'],
            f'{mode} f1score': batch_metrics['f1score'],
            'learning rate': optimizer.param_groups[0]['lr'],
            'step': total_step,
            'epoch': epoch
        })

        del image, labels, preds, pred_classes  # 清理内存

    # 计算epoch指标
    epoch_metrics = metric_collection.compute()
    epoch_loss /= n_iter
    print(f"{mode} f1score: {epoch_metrics['f1score']:.4f}")
    print(f'{mode} epoch loss: {epoch_loss:.4f}')

    # 日志epoch指标
    for k in epoch_metrics.keys():
        log_wandb.log({f'epoch_{mode}_{k}': epoch_metrics[k], 'epoch': epoch})
    metric_collection.reset()
    log_wandb.log({f'epoch_{mode}_loss': epoch_loss, 'epoch': epoch})

    # 日志图像
    log_wandb.log({
        f'{mode} t1_images': wandb.Image(to_pilimg(t1_img_log)),
        f'{mode} masks': {
            'label': wandb.Image(label_log.numpy()),  # 类别索引可视化
            'pred': wandb.Image(pred_log.numpy()),
        },
        'epoch': epoch
    })

    # 验证阶段保存模型
    if mode == 'val':
        if epoch_metrics['f1score'] > best_metrics['best_f1score']:
            non_improved_epoch = 0
            best_metrics['best_f1score'] = epoch_metrics['f1score']
            if ph.save_best_model:
                save_model(net, best_f1score_model_path, epoch, 'f1score')
        if epoch_loss < best_metrics['lowest loss']:
            best_metrics['lowest loss'] = epoch_loss
            if ph.save_best_model:
                save_model(net, best_loss_model_path, epoch, 'loss')
        else:
            non_improved_epoch += 1
            if non_improved_epoch >= ph.patience:
                lr *= ph.factor
                for g in optimizer.param_groups:
                    g['lr'] = lr
                non_improved_epoch = 0

        if (epoch + 1) % ph.save_interval == 0 and ph.save_checkpoint:
            save_model(net, checkpoint_path, epoch, 'checkpoint', optimizer)

    if mode == 'train':
        return log_wandb, net, optimizer, grad_scaler, total_step, lr
    else:
        return log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch

    # 返回值
    if mode == 'train':
        return log_wandb, net, optimizer, grad_scaler, total_step, lr
    else:
        return log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch