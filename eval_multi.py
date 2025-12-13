# eval_multi.py

import os
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_multi import LogLLM
from customDataset_multi import CustomDataset, CustomCollator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

max_content_len = 100
max_seq_len = 128
batch_size = 32

dataset_name = 'ICS_multi'   # 'Thunderbird' 'HDFS_v1' 'BGL' 'Liberty' 'ICS'
data_path = r'/data/fangly/shqxBS/w/data/{}/test.csv'.format(dataset_name)
# data_path = "/data/fangly/shqxBS/w/data/ICS/test.csv"

Bert_path = r"/data/fangly/shqxBS/w/models/bert-base-uncased"
Llama_path = r"/data/fangly/shqxBS/w/models/Meta-Llama-3-8B"
Qwen_path = r"/data/fangly/models/Qwen3-Coder-30B-A3B-Instruct"

ROOT_DIR = Path(__file__).parent
# ft_path = os.path.join(ROOT_DIR, r"ft_model_new_{}".format(dataset_name))
ft_path = os.path.join(ROOT_DIR, r"ft_model_qwen_multi_{}".format(dataset_name))

device = torch.device("cuda:0")

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

            inputs = inputs.to(device)
            seq_positions = seq_positions

            outputs_ids = model(inputs,seq_positions)
            outputs = model.Llama_tokenizer.batch_decode(outputs_ids)

            # print(outputs)

            for text in outputs:



                # match = re.search(r'normal|anomalous', text, re.IGNORECASE)

                LABEL_REGEX = (
                    r"normal|replay_attack|man_in_the_middle_attack|"
                    r"denial_of_service_attack|industrial_network_scanning_attack|"
                    r"unauthorized_command_injection_or_rogue_control_attack"
                )

                match = re.search(LABEL_REGEX, text.lower())



                if match:
                    preds.append(match.group())
                else:
                    print(f'error :{text}')
                    # preds.append('')
                    preds.append('unknown')

    preds_copy = np.array(preds)


    # preds = np.zeros_like(preds_copy,dtype=int)
    # preds[preds_copy == 'anomalous'] = 1
    # preds[preds_copy != 'anomalous'] = 0
    # gt = dataloader.dataset.get_label()

    # 所有可能的标签（包括预测不到 GT 的 'unknown'，方便看模型识别失败情况）
    CLASS_LABELS = [
        "normal",
        "replay_attack",
        "man_in_the_middle_attack",
        "denial_of_service_attack",
        "industrial_network_scanning_attack",
        "unauthorized_command_injection_or_rogue_control_attack",
        "unknown"
    ]

    # 预测值就是前面解析出来的字符串
    preds = preds_copy
    # GT 直接来自数据集的 Label 列，也是字符串
    gt = dataloader.dataset.get_label()

    print(classification_report(gt, preds,
                                labels=CLASS_LABELS,
                                target_names=CLASS_LABELS))



    # precision = precision_score(gt, preds, average="binary", pos_label=1)
    # recall = recall_score(gt, preds, average="binary", pos_label=1)
    # f = f1_score(gt, preds, average="binary", pos_label=1)
    # acc = accuracy_score(gt, preds)

    # num_anomalous = (gt == 1).sum()
    # num_normal = (gt == 0).sum()

    # print(f'Number of anomalous seqs: {num_anomalous}; number of normal seqs: {num_normal}')

    # pred_num_anomalous = (preds == 1).sum()
    # pred_num_normal =  (preds == 0).sum()

    # print(
    #     f'Number of detected anomalous seqs: {pred_num_anomalous}; number of detected normal seqs: {pred_num_normal}')

    # print(f'precision: {precision}, recall: {recall}, f1: {f}, acc: {acc}')


if __name__ == '__main__':
    print(f'dataset: {data_path}')
    dataset = CustomDataset(data_path)
    # model = LogLLM(Bert_path, Llama_path, ft_path=ft_path, is_train_mode=False, device=device,
    #                max_content_len=max_content_len, max_seq_len=max_seq_len)
    model = LogLLM(Bert_path, Qwen_path, ft_path=ft_path, is_train_mode=False, device=device,
                   max_content_len=max_content_len, max_seq_len=max_seq_len)

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