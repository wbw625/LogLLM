# customDataset_multi.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

patterns = [
    r'True', r'true', r'False', r'false',
    r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
    r'\b(Mon|Monday|Tue|Tuesday|Wed|Wednesday|Thu|Thursday|Fri|Friday|Sat|Saturday|Sun|Sunday)\b',
    r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\s+\b',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d{1,5})?',
    r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}',
    r'([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}',
    r'[a-zA-Z0-9]*[:\.]*([/\\]+[^/\\\s\[\]]+)+[/\\]*',
    r'\b[0-9a-fA-F]{8}\b',
    r'\b[0-9a-fA-F]{10}\b',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)\-(\w+[\w\.]*)',
    r'(\w+[\w\.]*)@(\w+[\w\.]*)',
    r'[a-zA-Z\.\:\-\_]*\d[a-zA-Z0-9\.\:\-\_]*'
]
combined_pattern = '|'.join(patterns)

def replace_patterns(text):
    text = re.sub(r'[\.]{3,}', '.. ', text)
    text = re.sub(combined_pattern, '<*>', text)
    return text


class CustomDataset(Dataset):
    def __init__(self, file_path, drop_duplicates=False):
        df = pd.read_csv(file_path)
        print(df['Label'].value_counts())
        df['Content'] = df['Content'].apply(replace_patterns)
        if drop_duplicates:
            df = df.drop_duplicates(subset='Content', keep='first')
        contents = df['Content'].values
        self.sequences = np.array([c.split(' ;-; ') for c in contents], dtype=object)
        self.labels = df['Label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def get_label(self):
        return self.labels


def merge_data(data):
    merged_data, starts = [], []
    cur = 0
    for seq in data:
        starts.append(cur)
        merged_data.extend(seq)
        cur += len(seq)
    return merged_data, starts


class BalancedSampler(Sampler):
    """
    Multi-class balanced sampler preserving class ratio ~target_ratio
    """
    def __init__(self, dataset, target_ratio=0.3, max_samples=None, min_samples=50000):
        self.labels = dataset.get_label()
        self.classes = np.unique(self.labels)
        self.target_ratio = target_ratio
        self.max_samples = max_samples
        self.min_samples = min_samples

        # count indices per class
        self.class_indices = {cls: np.where(self.labels == cls)[0] for cls in self.classes}
        class_sizes = {cls: len(v) for cls, v in self.class_indices.items()}
        self.majority_cls = max(class_sizes, key=class_sizes.get)
        self.majority_indices = self.class_indices[self.majority_cls]
        self.majority_size = len(self.majority_indices)

        # total minority samples target
        self.target_minority_size = int((self.target_ratio * self.majority_size) / (1 - self.target_ratio))
        self.total_size = self.majority_size + self.target_minority_size
        if self.max_samples:
            self.total_size = min(self.total_size, self.max_samples)
        elif self.total_size < self.min_samples:
            self.total_size = self.min_samples

    def __iter__(self):
        indices = list(self.majority_indices)

        # minority per class
        minority_classes = [c for c in self.classes if c != self.majority_cls]
        per_cls_target = max(1, self.target_minority_size // len(minority_classes))

        for cls in minority_classes:
            idxs = self.class_indices[cls]
            sampled = np.random.choice(idxs, size=per_cls_target, replace=True)
            indices.extend(sampled)

        if len(indices) > self.total_size:
            indices = np.random.choice(indices, self.total_size, replace=False)
        else:
            extra = np.random.choice(indices, self.total_size - len(indices), replace=True)
            indices = np.concatenate([indices, extra])

        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.total_size


class CustomCollator:
    def __init__(self, tokenizer, max_seq_len=128, max_content_len=100):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_content_len = max_content_len

    def __call__(self, batch):
        seqs, labels = zip(*batch)
        seqs = [s[:self.max_seq_len] for s in seqs]
        merged, starts = merge_data(seqs)
        starts = starts[1:]
        inputs = self.tokenizer(
            merged,
            return_tensors="pt",
            max_length=self.max_content_len,
            padding=True,
            truncation=True
        )
        labels = np.array(labels, dtype=object)
        return {
            "inputs": inputs,
            "seq_positions": torch.tensor(starts, dtype=torch.long),
            "labels": labels
        }
