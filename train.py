import sys
import time
import numpy as np
from torch import optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
from utils.path_hyperparameter import ph
import torch
# 替换损失函数：从losses中导入交叉熵损失
from utils.losses import CrossEntropyLoss2D  # 新增
import os
import logging
import random
import wandb
from rs_mamba_ss import RSM_SS
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from utils.utils import train_val_test


def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# 多通道图像转可视化3通道（蓝、绿、红波段，对应索引1、2、3）
def visualize_3ch(img_tensor):
    """从5通道张量中提取中间3个通道（蓝、绿、红）并归一化，用于可视化"""
    rgb_tensor = img_tensor[1:4, :, :].clone()  # 蓝、绿、红波段
    # 归一化到[0, 1]范围
    min_val = rgb_tensor.min()
    max_val = rgb_tensor.max()
    if max_val > min_val:
        rgb_tensor = (rgb_tensor - min_val) / (max_val - min_val)
    return rgb_tensor


def auto_experiment():
    random_seed(SEED=ph.random_seed)
    try:
        train_net(dataset_name=ph.dataset_name)
    except KeyboardInterrupt:
        logging.info('Interrupt')
        sys.exit(0)


def train_net(dataset_name):
    # 1. 创建数据集
    train_dataset = BasicDataset(
        images_dir=f'{ph.root_dir}/{dataset_name}/train/image/',
        labels_dir=f'{ph.root_dir}/{dataset_name}/train/label/',
        train=True
    )
    val_dataset = BasicDataset(
        images_dir=f'{ph.root_dir}/{dataset_name}/val/image/',
        labels_dir=f'{ph.root_dir}/{dataset_name}/val/label/',
        train=False
    )

    # 2. 数据集大小
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 3. 创建数据加载器
    loader_args = dict(
        num_workers=8,
        prefetch_factor=5,
        persistent_workers=True,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=False,
        batch_size=ph.batch_size, **loader_args
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False,
        batch_size=ph.batch_size * ph.inference_ratio,** loader_args
    )

    # 4. 初始化日志
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO)
    localtime = time.asctime(time.localtime(time.time()))
    hyperparameter_dict = ph.state_dict()
    hyperparameter_dict['time'] = localtime
    log_wandb = wandb.init(
        project=ph.log_wandb_project, resume='allow', anonymous='must',
        settings=wandb.Settings(start_method='thread'),
        config=hyperparameter_dict, mode='offline'
    )
    os.environ["WANDB_DIR"] = f"./{ph.log_wandb_project}"

    logging.info(f'''Starting training:
        Epochs:          {ph.epochs}
        Batch size:      {ph.batch_size}
        Learning rate:   {ph.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {ph.save_checkpoint}
        save best model: {ph.save_best_model}
        Device:          {device.type}
        Classes:         20  # 明确标注类别数
    ''')

    # 5. 初始化模型、优化器等
    net = RSM_SS(
        dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state,
        ssm_dt_rank=ph.ssm_dt_rank, ssm_ratio=ph.ssm_ratio,
        mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version,
        patchembed_version=ph.patchembed_version
    ).to(device=device)

    optimizer = optim.AdamW(
        net.parameters(), lr=ph.learning_rate, weight_decay=ph.weight_decay
    )
    warmup_lr = np.arange(
        1e-7, ph.learning_rate, (ph.learning_rate - 1e-7) / ph.warm_up_step
    )
    grad_scaler = torch.cuda.amp.GradScaler()

    # 加载预训练模型
    if ph.load:
        checkpoint = torch.load(ph.load, map_location=device)
        net.load_state_dict(checkpoint['net'])
        logging.info(f'Model loaded from {ph.load}')
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['lr'] = ph.learning_rate
            optimizer.param_groups[0]['capturable'] = True

    total_step = 0
    lr = ph.learning_rate

    # 关键修改1：替换为交叉熵损失（适配20类）
    criterion = CrossEntropyLoss2D(ignore_index=255)  # 忽略255（若有无效标签）

    best_metrics = {'best_f1score': 0, 'lowest loss': float('inf')}

    # 关键修改2：确保类别数为20
    num_classes = 20  # 直接指定为20，避免配置文件错误
    metric_collection = MetricCollection({
        'accuracy': Accuracy(task="multiclass", num_classes=num_classes).to(device),
        'precision': Precision(task="multiclass", num_classes=num_classes).to(device),
        'recall': Recall(task="multiclass", num_classes=num_classes).to(device),
        'f1score': F1Score(task="multiclass", num_classes=num_classes).to(device)
    })

    to_pilimg = T.ToPILImage()

    # 模型保存路径
    checkpoint_path = f'./{ph.project_name}_checkpoint/'
    best_f1score_model_path = f'./{ph.project_name}_best_f1score_model/'
    best_loss_model_path = f'./{ph.project_name}_best_loss_model/'
    non_improved_epoch = 0

    # 6. 开始训练和验证
    for epoch in range(ph.epochs):
        print('Start Train!')
        # 训练阶段：传入可视化函数
        log_wandb, net, optimizer, grad_scaler, total_step, lr = train_val_test(
            mode='train', dataset_name=dataset_name,
            dataloader=train_loader, device=device, log_wandb=log_wandb,
            net=net, optimizer=optimizer, total_step=total_step, lr=lr,
            criterion=criterion, metric_collection=metric_collection,
            to_pilimg=to_pilimg, epoch=epoch, warmup_lr=warmup_lr,
            grad_scaler=grad_scaler,
            visualize_fn=visualize_3ch
        )

        # 验证阶段
        if (epoch + 1) >= ph.evaluate_epoch and (epoch + 1) % ph.evaluate_inteval == 0:
            print('Start Validation!')
            with torch.no_grad():
                log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch = train_val_test(
                    mode='val', dataset_name=dataset_name,
                    dataloader=val_loader, device=device, log_wandb=log_wandb,
                    net=net, optimizer=optimizer, total_step=total_step, lr=lr,
                    criterion=criterion, metric_collection=metric_collection,
                    to_pilimg=to_pilimg, epoch=epoch,
                    best_metrics=best_metrics, checkpoint_path=checkpoint_path,
                    best_f1score_model_path=best_f1score_model_path,
                    best_loss_model_path=best_loss_model_path,
                    non_improved_epoch=non_improved_epoch,
                    visualize_fn=visualize_3ch
                )

    wandb.finish()


if __name__ == '__main__':
    auto_experiment()