import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tifffile  # 用于加载多通道/高比特TIFF


class BasicDataset(Dataset):
    """支持多通道输入的数据集类（如5通道遥感图像）"""

    def __init__(self, images_dir: str, labels_dir: str, train: bool):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.train = train

        # 获取所有图像ID（不含后缀）
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        self.ids.sort()

        if not self.ids:
            raise RuntimeError(f'在 {images_dir} 中未找到图像文件，请检查路径是否正确')
        logging.info(f'数据集创建完成，包含 {len(self.ids)} 个样本')

        # 训练时的数据增强（支持多通道）
        self.train_transforms_all = A.Compose([
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            # 其他增强方法（确保支持多通道）
        ], additional_targets={'mask': 'mask'})  # 明确指定mask的增强

        # 归一化（适配多通道）
        self.normalize = A.Compose([
            A.Normalize(
                mean=[0.209, 0.394, 0.380, 0.344, 0.481],
                std=[0.141, 0.027, 0.032, 0.046, 0.069],
                max_pixel_value=255.0  # 图像已在load方法中缩放到0-255
            )
        ])

        # 转换为Tensor
        self.to_tensor = A.Compose([
            ToTensorV2()  # 会将 (H, W, C) 转为 (C, H, W)
        ])

    def __len__(self):
        return len(self.ids)

    @classmethod
    def label_preprocess(cls, label):
        """确保标签为单通道，不修改原始标签值"""
        label = label.astype(np.uint8)
        # 只确保标签是单通道，不进行二值化
        return label

    @classmethod
    @classmethod
    def load(cls, filename, is_label=False):
        """加载图像（保留原始通道数）"""
        filename = str(filename)
        try:
            if filename.lower().endswith(('.tif', '.tiff')):
                # 用tifffile加载TIFF（支持多通道/高比特）
                img = tifffile.imread(filename)
            
                # 如果是标签，直接处理通道，不做归一化
                if is_label:
                    # 确保标签是单通道（H, W）
                    if img.ndim == 3 and img.shape[-1] > 1:
                        img = img[..., 0]
                    if img.ndim == 3:
                        img = img.squeeze(-1)
                    return img.astype(np.uint8)  # 仅转换类型，不修改数值
            
                # 非标签（图像）才执行归一化
                else:
                    img = img.astype(np.float32)
                    min_val = img.min()
                    max_val = img.max()
                    if max_val > min_val:
                        img = (img - min_val) / (max_val - min_val) * 255  # 图像归一化
                    return img.astype(np.uint8)
        
            # 其他格式（非TIFF）的处理逻辑（同样需区分标签）
            else:
                from PIL import Image
                img = Image.open(filename)
                img = np.array(img)
            
                if is_label:
                    # 标签：仅处理通道，不修改数值
                    if img.ndim == 3 and img.shape[-1] > 1:
                        img = img[..., 0].squeeze()
                    return img.astype(np.uint8)
                else:
                    # 图像：转为uint8（PIL加载的图像通常已在0-255范围）
                    return img.astype(np.uint8)
                
        except Exception as e:
            raise RuntimeError(f"加载文件 {filename} 失败: {e}")

    def __getitem__(self, idx):
        name = self.ids[idx]
        # 查找图像和标签文件
        img_files = list(self.images_dir.glob(name + '.*'))
        label_files = list(self.labels_dir.glob(name + '.*'))
        
        assert len(img_files) == 1, f"图像 {name} 未找到或存在多个: {img_files}"
        assert len(label_files) == 1, f"标签 {name} 未找到或存在多个: {label_files}"
        
        # 加载图像（保留原始通道）和标签（单通道）
        img = self.load(img_files[0], is_label=False)
        label = self.load(label_files[0], is_label=True)
        
        # 调试：打印加载后的标签值范围和形状
        # if idx == 0:
        #     print(f"加载的标签 {name} 范围：min={label.min()}, max={label.max()}, 形状={label.shape}")
        
        # 确保图像和标签尺寸匹配（H, W）
        assert img.shape[:2] == label.shape[:2], \
            f"图像与标签尺寸不匹配: 图像 {img.shape[:2]}, 标签 {label.shape[:2]}"
        
        # 标签预处理（不修改原始值）
        label = self.label_preprocess(label)
        
        # 调试：打印预处理后的标签值范围和形状
        # if idx == 0:
        #     print(f"预处理后的标签 {name} 范围：min={label.min()}, max={label.max()}, 形状={label.shape}")
        
        # 确保图像是 (H, W, C) 格式
        if img.ndim == 2:
            img = img[..., np.newaxis]  # 单通道转为 (H, W, 1)
        
        # 训练时的数据增强（同时处理图像和标签）
        if self.train:
            augmented = self.train_transforms_all(image=img, mask=label)
            img = augmented['image']
            label = augmented['mask']
        
        # 归一化（保持 (H, W, C) 格式）
        normalized = self.normalize(image=img)
        img = normalized['image']
        
        # 转为Tensor（将 (H, W, C) 转为 (C, H, W)）
        sample = self.to_tensor(image=img, mask=label)
        tensor = sample['image'].contiguous()  # 形状 (C, H, W)，C为原始通道数
        label_tensor = sample['mask'].contiguous()  # 此时形状可能为 (1, H, W) 或 (H, W)
        
        # 关键修改：移除所有大小为1的维度（无论位置）
        # 确保标签最终形状为 (H, W)，避免存在 (1, H, W) 或 (H, W, 1) 等情况
        label_tensor = label_tensor.squeeze()  # 移除所有维度为1的轴
        
        # 调试：打印最终标签张量的形状
        # if idx == 0:
        #     print(f"最终标签张量形状: {label_tensor.shape}")  # 应输出 (256, 256)
        
        return tensor, label_tensor, name

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser(description='测试标签数值范围')
    parser.add_argument('--images_dir', type=str, required=True, help='图像文件夹路径')
    parser.add_argument('--labels_dir', type=str, required=True, help='标签文件夹路径')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    args = parser.parse_args()
    
    # 创建数据集实例
    dataset = BasicDataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        train=False  # 测试模式，不进行数据增强
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"数据集包含 {len(dataset)} 个样本")
    print("开始检查标签数值范围...")
    
    # 初始化全局统计量
    global_min = float('inf')
    global_max = float('-inf')
    unique_values = set()
    label_counts = {}
    
    # 遍历所有批次
    for i, (images, labels, names) in enumerate(dataloader):
        # 计算当前批次的标签统计量
        batch_min = labels.min().item()
        batch_max = labels.max().item()
        
        # 更新全局统计量
        global_min = min(global_min, batch_min)
        global_max = max(global_max, batch_max)
        
        # 收集所有唯一值
        for b in range(labels.shape[0]):
            unique_values.update(labels[b].unique().tolist())
        
        # 打印每个批次的统计信息
        print(f"批次 {i+1}/{len(dataloader)}: 标签范围 [{batch_min}, {batch_max}]")
        
        # 对前几个批次打印详细的标签值分布
        if i < 3:
            for b in range(min(3, labels.shape[0])):  # 只打印前3个样本
                unique, counts = labels[b].unique(return_counts=True)
                print(f"  样本 {names[b]}: 唯一标签值 = {unique.tolist()}, 数量 = {counts.tolist()}")
    
    # 统计所有唯一标签值及其出现次数
    print("\n=== 全部标签值统计 ===")
    print(f"全局标签范围: [{global_min}, {global_max}]")
    print(f"唯一标签值 ({len(unique_values)} 个): {sorted(unique_values)}")
    
    # 检查是否存在超出预期范围的标签值（假设类别数为20）
    expected_classes = list(range(20))
    unexpected_values = [v for v in unique_values if v not in expected_classes]
    
    if unexpected_values:
        print(f"警告: 发现超出预期范围 [0, 19] 的标签值: {unexpected_values}")
        print("这可能是CUDA断言错误的原因，请检查数据或设置ignore_index。")
    else:
        print("所有标签值均在预期范围内。")