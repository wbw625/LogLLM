import os
import json
import csv
import random

BASE_DIR = "/data/fangly/shqxBS/data/ics"
OUT_DIR = "/data/fangly/shqxBS/w/data/ICS_log"
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_MAP = {
    "normal": "0",
    "attack": "1",
}

TRAIN_RATIO = 0.9


def normalize_field(name: str) -> str:
    """去掉字段名中的所有空格。"""
    return str(name).replace(" ", "")


def build_log_line_from_row(row, fields):
    """
    把一行 data（list）转成：
    字段1: 值1, 字段2: 值2, ...
    fields 已经过 normalize（去空格）
    """
    parts = []
    n = min(len(row), len(fields))
    for i in range(n):
        parts.append(f"{fields[i]}: {str(row[i])}")
    for i in range(n, len(fields)):
        parts.append(f"{fields[i]}: ")
    return ", ".join(parts)


def collect_samples(base_dir):
    samples = []

    for dirpath, _, filenames in os.walk(base_dir):
        base = os.path.basename(dirpath)
        if base not in LABEL_MAP:
            continue

        label = LABEL_MAP[base]
        print(f"[INFO] Processing dir: {dirpath} (label={label})")

        for fname in sorted(filenames):
            if not fname.endswith(".json"):
                continue

            fpath = os.path.join(dirpath, fname)

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load JSON {fpath}: {e}")
                continue

            input_obj = data.get("input", {})
            raw_fields = input_obj.get("fields", None)
            rows = input_obj.get("data", [])

            if not isinstance(raw_fields, list) or len(raw_fields) == 0:
                print(f"[WARN] Missing/invalid input.fields in {fpath}, skip.")
                continue

            if not rows:
                print(f"[WARN] Empty data in {fpath}")
                continue

            # ✅ 去掉字段名空格
            fields = [normalize_field(x) for x in raw_fields]

            session_length = len(rows)
            log_lines = [build_log_line_from_row(r, fields) for r in rows]
            content = " ;-; ".join(log_lines)

            samples.append((content, label, session_length))

    print(f"[INFO] Total sessions collected: {len(samples)}")
    return samples


def split_and_save(samples, out_dir, train_ratio=0.8, seed=42):
    random.seed(seed)
    random.shuffle(samples)

    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    train_samples = samples[:n_train]
    test_samples = samples[n_train:]

    def save_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["Content", "Label", "session_length"])
            for content, label, slen in rows:
                writer.writerow([content, label, slen])
        print(f"[OK] Saved: {path} ({len(rows)} rows)")

    save_csv(os.path.join(out_dir, "train.csv"), train_samples)
    save_csv(os.path.join(out_dir, "test.csv"), test_samples)


if __name__ == "__main__":
    samples = collect_samples(BASE_DIR)
    split_and_save(samples, OUT_DIR, train_ratio=TRAIN_RATIO)
    print("[ALL DONE] ics_json -> logllm CSV 完成。")
