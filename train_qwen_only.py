# train_qwen_only.py

import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from model_qwen_only import LogLLM
from torch.utils.data import DataLoader
from customDataset import CustomDataset, CustomCollator, BalancedSampler
from torch import optim

# 消融实验：只有单一阶段训练
n_epochs = 3
dataset_name = 'BGL_10' # 'Thunderbird' 'HDFS_v1' 'BGL' 'Liberty' 'ICS'
batch_size = 16
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size

lr = 1e-4  # 只有一个学习率

max_content_len = 100
max_seq_len = 128

data_path = r'/data/fangly/shqxBS/log/data/{}/train.csv'.format(dataset_name)
min_less_portion = 0.5

# BERT路径不再需要
# Bert_path = r"/data/fangly/shqxBS/models/bert-base-uncased"
# Llama_path = r"/data/fangly/shqxBS/models/Meta-Llama-3-8B"
Llama_path = r"/data/fangly/shqxBS/models/Qwen3-Coder-30B-A3B-Instruct"

ROOT_DIR = Path(__file__).parent
# 修改保存路径名以示区别
ft_path = os.path.join(ROOT_DIR, r"ft_model_qwenonly_{}".format(dataset_name))

device = torch.device("cuda:2")

print(f'n_epochs: {n_epochs}\n'
      f'dataset_name: {dataset_name}\n'
      f'batch_size: {batch_size}\n'
      f'micro_batch_size: {micro_batch_size}\n'
      f'lr: {lr}\n'
      f'max_content_len: {max_content_len}\n'
      f'max_seq_len: {max_seq_len}\n'
      f'min_less_portion: {min_less_portion}\n'
      f'device: {device}')

def print_number_of_trainable_model_parameters(model):
    params = set()
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            params.add(param)
            trainable_model_params += param.numel()
    print(f"all params num: {all_model_params}, trainable param num: {trainable_model_params}")
    return params

def trainModel(model, dataloader, gradient_accumulation_steps, n_epochs, lr):
    criterion = nn.CrossEntropyLoss(reduction='mean')

    trainable_model_params = print_number_of_trainable_model_parameters(model)
    optimizer = torch.optim.AdamW(trainable_model_params, lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    # 获取 Normal/Anomalous 的 token ID 用于计算准确率
    normal_tokens = model.Llama_tokenizer('The sequence is normal.', add_special_tokens=False)['input_ids']
    anomalous_tokens = model.Llama_tokenizer('The sequence is anomalous.', add_special_tokens=False)['input_ids']
    special_normal_tokens = set(normal_tokens) - set(anomalous_tokens)
    special_anomalous_tokens = set(anomalous_tokens) - set(normal_tokens)

    total_steps = n_epochs * len(dataloader)
    scheduler_step = max(int(total_steps / 10), 1)

    print(f'scheduler_step: {scheduler_step}')

    steps = 0
    for epoch in range(int(n_epochs)):
        total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0

        pbar = tqdm(dataloader, desc='Epoch {}/{}'.format(epoch, n_epochs))
        for i_th, bathc_i in enumerate(pbar):
            steps += 1

            inputs = bathc_i['inputs']
            seq_positions = bathc_i['seq_positions']
            labels = bathc_i['labels']

            # inputs 现在是 Qwen 的 tokens，需要移动到 device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # seq_positions 不需要 move 到 GPU，tensor_split 在 CPU 上处理索引也可以，或者在内部处理
            
            outputs, targets = model.train_helper(inputs, seq_positions, labels)

            loss = criterion(outputs, targets)
            loss = loss / gradient_accumulation_steps

            loss.backward()

            if ((i_th + 1) % gradient_accumulation_steps == 0) or ((i_th + 1) == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            acc_mask = torch.zeros_like(targets, device=device).bool()
            for token in special_normal_tokens.union(special_anomalous_tokens):
                acc_mask[targets == token] = True

            if acc_mask.sum() > 0:
                total_acc += (outputs.argmax(1)[acc_mask] == targets[acc_mask]).sum().item()
                total_acc_count += acc_mask.sum()

            train_loss += loss.item() * gradient_accumulation_steps * targets.size(0)
            total_count += targets.size(0)

            if steps % scheduler_step == 0:
                scheduler.step()
            pbar.set_postfix(lr=scheduler.get_last_lr()[0], loss=loss.item() * gradient_accumulation_steps)

            if steps % 10000 == 0:
                train_loss_epoch = train_loss / total_count if total_count > 0 else 0
                train_acc_epoch = total_acc / total_acc_count if total_acc_count > 0 else 0
                print(f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{n_epochs}] "
                      f"[loss: {train_loss_epoch:3f}]"
                      f"[acc: {train_acc_epoch:3f}]")
                total_acc, total_acc_count, total_count, train_loss = 0, 0, 0, 0

        if total_count > 0:
            train_loss_epoch = train_loss / total_count
            train_acc_epoch = total_acc / total_acc_count if total_acc_count > 0 else 0
            print(f"[Epoch {epoch + 1:{len(str(n_epochs))}}/{n_epochs}] "
                  f"[loss: {train_loss_epoch:3f}]"
                  f"[acc: {train_acc_epoch:3f}]")

if __name__ == '__main__':
    print(f'dataset: {data_path}')
    dataset = CustomDataset(data_path, drop_duplicates=False)

    # 仅使用 Qwen/Llama
    model = LogLLM(Llama_path, device=device, max_content_len=max_content_len, max_seq_len=max_seq_len)

    # 关键修改：使用 Qwen 的 Tokenizer 传给 Collator
    # Collator 必须使用 Qwen Tokenizer 来对原始日志文本进行 tokenize
    tokenizer = model.Llama_tokenizer
    collator = CustomCollator(tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len)

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        num_workers=4,
        sampler=BalancedSampler(dataset, target_ratio=min_less_portion),
        collate_fn=collator,
        drop_last=True
    )

    model.set_finetuning_all() # 开启 LoRA 训练
    print("*" * 10 + "Start training ablation model (Qwen Only)" + "*" * 10)
    trainModel(model, dataloader, gradient_accumulation_steps, n_epochs, lr)

    model.save_ft_model(ft_path)