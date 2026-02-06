import os
import torch
from torch.utils.data import DataLoader
# 假设 customDataset.py 在同一目录下
from customDataset import CustomDataset, BalancedSampler

def calculate_phase2_3_samples():
    # ---------------- 配置参数 (保持与 train.py 一致) ----------------
    dataset_name = 'BGL_5'
    dataset_name = 'BGL'
    data_path = r'/data/fangly/shqxBS/log/data/{}/train.csv'.format(dataset_name)
    min_less_portion = 0.3
    micro_batch_size = 4  # train.py 中的配置
    # ---------------------------------------------------------------

    print(f"正在读取数据集: {data_path} ...")
    
    if not os.path.exists(data_path):
        print(f"错误: 找不到文件 {data_path}")
        print("请确保在拥有数据的服务器环境下运行此脚本。")
        return

    # 1. 加载数据集
    dataset = CustomDataset(data_path, drop_duplicates=False)
    original_len = len(dataset)
    print(f"原始数据集大小 (len(dataset)): {original_len}")

    # 2. 初始化第2、3阶段使用的 Sampler
    # train.py 中 Phase 2/3 的定义: sampler=BalancedSampler(dataset, target_ratio=min_less_portion)
    # 注意：这里没有 max_samples 参数
    print(f"正在初始化 BalancedSampler (target_ratio={min_less_portion})...")
    sampler = BalancedSampler(dataset, target_ratio=min_less_portion)

    # 3. 计算 Sampler 产生的样本总数
    # BalancedSampler 通常是一个迭代器，我们需要遍历它或者看它是否实现了 __len__
    try:
        sampler_len = len(sampler)
        print(f"Sampler 报告的长度 (__len__): {sampler_len}")
    except TypeError:
        print("Sampler 未实现 __len__，正在通过遍历计算...")
        sampler_len = sum(1 for _ in sampler)
        print(f"遍历计算得出的 Sampler 样本总数: {sampler_len}")

    # 4. 考虑 DataLoader 的 drop_last=True
    # train.py 中: drop_last=True
    num_batches = sampler_len // micro_batch_size
    actual_samples_seen = num_batches * micro_batch_size
    
    print("-" * 30)
    print(">>> 统计结果 <<<")
    print(f"原始数据行数: {original_len}")
    print(f"Phase 2/3 Sampler 计划采样的总数: {sampler_len}")
    print(f"由于 batch_size={micro_batch_size} 且 drop_last=True:")
    print(f"实际进入训练的样本数 (Epoch size): {actual_samples_seen}")
    print(f"每个 Epoch 的步数 (Steps): {num_batches}")
    print("-" * 30)

if __name__ == '__main__':
    calculate_phase2_3_samples()