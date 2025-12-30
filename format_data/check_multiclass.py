import csv
from collections import Counter

DATASET_DIR = "/data/fangly/shqxBS/w/data/ICS_log_multi"  # 换成你的输出目录

files = ["train.csv", "test.csv"]

# files = ["test.csv"]

# files = ["train.csv"]

label_counter = Counter()

for filename in files:
    path = f"{DATASET_DIR}/{filename}"
    print(f"[INFO] Reading {path} ...")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_counter[row["Label"]] += 1

print("\n====== 📊 CLASS DISTRIBUTION ======")
total = sum(label_counter.values())

for label, count in label_counter.items():
    print(f"{label:<25} : {count} ({count/total:.2%})")

print("====================================")
print(f"Total samples: {total}")
