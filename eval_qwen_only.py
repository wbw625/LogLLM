# eval_qwen_only.py

import os
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_qwen_only import LogLLM
from customDataset import CustomDataset, CustomCollator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

max_content_len = 100
max_seq_len = 128
batch_size = 32

dataset_name = 'ICS_log'
data_path = r'/data/fangly/shqxBS/log/data/{}/test.csv'.format(dataset_name)

# Bert_path 不再需要
# Llama_path = r"/data/fangly/shqxBS/models/Meta-Llama-3-8B"
Llama_path = r"/data/fangly/shqxBS/models/Qwen3-Coder-30B-A3B-Instruct"

ROOT_DIR = Path(__file__).parent
ft_path = os.path.join(ROOT_DIR, r"ft_model_qwenonly_{}".format(dataset_name))

device = torch.device("cuda:3")

print(
    f'dataset_name: {dataset_name}\n'
    f'batch_size: {batch_size}\n'
    f'max_content_len: {max_content_len}\n'
    f'max_seq_len: {max_seq_len}\n'
    f'device: {device}')

def evalModel(model, dataloader):
    model.eval()

    preds = []

    with torch.no_grad():
        for bathc_i in tqdm(dataloader):
            inputs = bathc_i['inputs']
            seq_positions = bathc_i['seq_positions']

            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs_ids = model(inputs, seq_positions)
            outputs = model.Llama_tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)

            for text in outputs:
                # 稍微放宽正则匹配，防止生成多余字符
                match = re.search(r'normal|anomalous', text, re.IGNORECASE)
                if match:
                    preds.append(match.group().lower()) # 转小写统一比较
                else:
                    print(f'error prediction format: {text}')
                    preds.append('unknown')

    preds_copy = np.array(preds)
    preds = np.zeros_like(preds_copy, dtype=int)
    preds[preds_copy == 'anomalous'] = 1
    preds[preds_copy != 'anomalous'] = 0 # unknown 也会被当做 0 处理，或者你可以单独处理
    
    gt = dataloader.dataset.get_label()

    precision = precision_score(gt, preds, average="binary", pos_label=1)
    recall = recall_score(gt, preds, average="binary", pos_label=1)
    f = f1_score(gt, preds, average="binary", pos_label=1)
    acc = accuracy_score(gt, preds)

    num_anomalous = (gt == 1).sum()
    num_normal = (gt == 0).sum()

    print(f'Number of anomalous seqs: {num_anomalous}; number of normal seqs: {num_normal}')

    pred_num_anomalous = (preds == 1).sum()
    pred_num_normal =  (preds == 0).sum()

    print(
        f'Number of detected anomalous seqs: {pred_num_anomalous}; number of detected normal seqs: {pred_num_normal}')

    print(f'precision: {precision}, recall: {recall}, f1: {f}, acc: {acc}')


if __name__ == '__main__':
    print(f'dataset: {data_path}')
    dataset = CustomDataset(data_path)
    
    # 只需要 Qwen path
    model = LogLLM(Llama_path, ft_path=ft_path, is_train_mode=False, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len)

    # 使用 Qwen tokenizer
    tokenizer = model.Llama_tokenizer
    collator = CustomCollator(tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )

    evalModel(model, dataloader)