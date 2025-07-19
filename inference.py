import sys
import numpy as np
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
import logging
from utils.path_hyperparameter import ph
import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, JaccardIndex
from rs_mamba_ss import RSM_SS
from tqdm import tqdm


def train_net(dataset_name, load_checkpoint=True):
    # 1. 创建数据集
    test_dataset = BasicDataset(
        images_dir=f'./{dataset_name}/test/image/',
        labels_dir=f'./{dataset_name}/test/label/',
        train=False
    )
    
    # 2. 创建数据加载器
    loader_args = dict(
        num_workers=8,
        prefetch_factor=5,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=ph.batch_size * ph.inference_ratio,
        **loader_args
    )

    # 3. 初始化日志
    logging.basicConfig(level=logging.INFO)

    # 4. 设置设备、模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备: {device}')
    
    # 初始化模型
    net = RSM_SS(
        dims=ph.dims,
        depths=ph.depths,
        ssm_d_state=ph.ssm_d_state,
        ssm_dt_rank=ph.ssm_dt_rank,
        ssm_ratio=ph.ssm_ratio,
        mlp_ratio=ph.mlp_ratio,
        downsample_version=ph.downsample_version,
        patchembed_version=ph.patchembed_version
    )
    net.to(device=device)

    # 加载模型权重
    assert ph.load, '模型加载错误：未指定 checkpoint 路径'
    load_model = torch.load(ph.load, map_location=device)
    if isinstance(load_model, dict) and 'net' in load_model:
        net.load_state_dict(load_model['net'])
    else:
        net.load_state_dict(load_model)
    logging.info(f'模型已从 {ph.load} 加载')
    torch.save(net.state_dict(), f'{dataset_name}_best_model.pth')

    # 5. 初始化评估指标
    num_classes = 20
    # 全局指标：OA、mIoU、macro-F1
    global_metrics = MetricCollection({
        'OA': Accuracy(
            task="multiclass",
            num_classes=num_classes,
            average='micro'
        ).to(device),
        'mIoU': JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            average='macro'
        ).to(device),
        'macro-F1': F1Score(
            task="multiclass",
            num_classes=num_classes,
            average='macro'
        ).to(device)
    }).to(device)

    # 类别级指标：每类的精确率、召回率、IoU
    class_precision = Precision(
        task="multiclass",
        num_classes=num_classes,
        average=None
    ).to(device)
    class_recall = Recall(
        task="multiclass",
        num_classes=num_classes,
        average=None
    ).to(device)
    class_iou = JaccardIndex(
        task="multiclass",
        num_classes=num_classes,
        average=None
    ).to(device)

    # 6. 模型评估
    net.eval()
    logging.info('模型已切换至测试模式！')

    with torch.no_grad():
        for batch_img1, labels, name in tqdm(test_loader):
            # 数据预处理
            batch_img1 = batch_img1.float().to(device)
            labels = labels.long().to(device)  # 标签为类别索引（0~19）

            # 模型推理
            ss_preds = net(batch_img1)  # 输出：[B, 20, H, W]
            ss_preds = torch.softmax(ss_preds, dim=1)  # 概率分布
            preds_classes = torch.argmax(ss_preds, dim=1)  # 类别索引：[B, H, W]

            # 确保标签维度正确
            if labels.dim() == 4:
                labels = labels.squeeze(1)  # [B, 1, H, W] → [B, H, W]

            # 更新指标
            global_metrics.update(preds_classes, labels)
            class_precision.update(preds_classes, labels)
            class_recall.update(preds_classes, labels)
            class_iou.update(preds_classes, labels)

            # 清理内存
            del batch_img1, labels, ss_preds, preds_classes

        # 计算最终结果
        global_results = global_metrics.compute()
        class_precision_results = class_precision.compute().cpu().numpy()
        class_recall_results = class_recall.compute().cpu().numpy()
        class_iou_results = class_iou.compute().cpu().numpy()

        # 7. 打印结果
        print("\n" + "="*80)
        print("全局评估指标：")
        print(f"OA（总体准确率）: {global_results['OA'].item():.4f}")
        print(f"mIoU（平均交并比）: {global_results['mIoU'].item():.4f}")
        print(f"macro-F1: {global_results['macro-F1'].item():.4f}")
        print("="*80 + "\n")

        print("各类别详细指标（精确率、召回率、IoU）：")
        # 表头
        print(f"{'类别ID':<10} | {'精确率':<10} | {'召回率':<10} | {'IoU':<10}")
        print("-"*45)
        # 逐行打印每个类别的指标
        for cls in range(num_classes):
            print(
                f"{cls:<10} | {class_precision_results[cls]:<10.4f} | "
                f"{class_recall_results[cls]:<10.4f} | {class_iou_results[cls]:<10.4f}"
            )
        print("="*80)

    print('测试完成')


if __name__ == '__main__':
    try:
        train_net(dataset_name='dataset')
    except KeyboardInterrupt:
        logging.info('程序被手动中断')
        sys.exit(0)
    except Exception as e:
        logging.error(f'运行出错: {str(e)}')
        sys.exit(1)