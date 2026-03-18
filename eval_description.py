# eval_description.py

import os
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_description import LogLLM
from customDataset_description import CustomDataset, CustomCollator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

max_content_len = 100
max_seq_len = 128
max_new_tokens = 100
batch_size = 32

dataset_name = 'Sherlock'   # 'Thunderbird' 'HDFS_v1' 'BGL' 'Liberty' 'ICS'
data_path = r'/data/fangly/shqxBS/log/data/{}/test01_desc.csv'.format(dataset_name)

Bert_path = r"/data/fangly/shqxBS/models/bert-base-uncased"
Llama_path = r"/data/fangly/shqxBS/models/Meta-Llama-3-8B"
Qwen_path = r"/data/fangly/shqxBS/models/Qwen3-Coder-30B-A3B-Instruct"

ROOT_DIR = Path(__file__).parent
ft_path = os.path.join(ROOT_DIR, r"ft_model_qwen_desc_{}_01".format(dataset_name))

device = torch.device("cuda:6")

print(
f'dataset_name: {dataset_name}\n'
f'batch_size: {batch_size}\n'
f'max_content_len: {max_content_len}\n'
f'max_seq_len: {max_seq_len}\n'
f'max_new_tokens: {max_new_tokens}\n'
f'device: {device}')


def evalModel(model, dataloader):
    model.eval()

    preds = []
    pred_descriptions = []

    with torch.no_grad():
        for bathc_i in tqdm(dataloader):
            inputs = bathc_i['inputs']
            seq_positions = bathc_i['seq_positions']

            inputs = inputs.to(device)
            seq_positions = seq_positions

            outputs_ids = model(inputs,seq_positions)
            outputs = model.Llama_tokenizer.batch_decode(outputs_ids)

            print(outputs)

            for text in outputs:
                # 提取 label
                match = re.search(r'normal|anomalous', text, re.IGNORECASE)
                if match:
                    label = match.group().lower()
                    preds.append(label)
                else:
                    print(f'error :{text}')
                    preds.append('')

                # 提取 description（label 之后的文本，适用于 normal 和 anomalous）
                desc = ''
                if match:
                    after = text[match.end():]
                    # 去掉开头的标点和空白
                    after = re.sub(r'^[\s\.\,\:]+', '', after)
                    # 去掉结尾的 eos/pad tokens 残留
                    after = re.sub(r'<\|.*?\|>.*$', '', after).strip()
                    if after:
                        desc = after
                pred_descriptions.append(desc)

    preds_copy = np.array(preds)
    preds = np.zeros_like(preds_copy,dtype=int)
    preds[preds_copy == 'anomalous'] = 1
    preds[preds_copy != 'anomalous'] = 0
    gt = dataloader.dataset.get_label()
    gt_descriptions = dataloader.dataset.get_description()

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

    # 打印一些有 description 的样本的对比（包括 normal 和 anomalous）
    print('\n--- Sample predictions with descriptions (GT has description) ---')
    count = 0
    for i in range(len(preds)):
        if gt_descriptions[i] and count < 10:
            label_str = 'anomalous' if gt[i] == 1 else 'normal'
            pred_label_str = 'anomalous' if preds[i] == 1 else 'normal'
            correct = 'OK' if gt[i] == preds[i] else 'WRONG'
            print(f'  [{correct}] GT={label_str}, Pred={pred_label_str}')
            print(f'  [GT desc]   {gt_descriptions[i]}')
            print(f'  [Pred desc] {pred_descriptions[i]}')
            print()
            count += 1


if __name__ == '__main__':
    print(f'dataset: {data_path}')
    dataset = CustomDataset(data_path)
    model = LogLLM(Bert_path, Qwen_path, ft_path=ft_path, is_train_mode=False, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len, max_new_tokens=max_new_tokens)

    tokenizer = model.Bert_tokenizer
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
