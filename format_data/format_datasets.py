import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

# ===========================
#   配置路径 & 参数
# ===========================

DATA_DIR = "/data/fangly/shqxBS/w/data/smart_grid_datasets/ics-dataset-for-smart-grids/but-iec104-i"
OUT_DIR  = "/data/fangly/shqxBS/w/data/ICS_new"   # 输出路径

# BUT IEC 104 – I 文件及是否攻击映射
FILE_LABELS = {
    "normal-traffic.csv": 0,
    "connection-loss.csv": 1,
    "dos-attack.csv": 1,
    "injection-attack.csv": 1,
    "rogue-device.csv": 1,
    "scanning-attack.csv": 1,
    "switching-attack.csv": 1,
}

SESSION_SIZE = 100              # 固定 session_length = 100
TRAIN_RATIO  = 0.8              # 80% train，20% test
os.makedirs(OUT_DIR, exist_ok=True)

# 按照原始字段顺序
FIELD_ORDER = [
    "TimeStamp",
    "Relative Time",
    "srcIP",
    "dstIP",
    "srcPort",
    "dstPort",
    "ipLen",
    "len",
    "fmt",
    "uType",
    "asduType",
    "numix",
    "cot",
    "oa",
    "addr",
    "ioa",
]

# ===========================
#   日志行构造函数
# ===========================

def build_log_line(row: dict) -> str:
    """
    将原始 CSV 行转换为：
    ['v1','v2','v3',...,'v16']
    只在内部用单引号，外层交给 CSV 自动加双引号。
    """
    values = [str(row.get(col, "")) for col in FIELD_ORDER]
    # 用单引号手工构造“类 Python list”的字符串
    inner = ", ".join(f"'{v}'" for v in values)
    log = "[" + inner + "], "
    # 保守起见，去掉换行
    log = log.replace("\n", " ").replace("\r", " ")
    return log


# ===========================
#   读取全部 CSV
# ===========================

def load_all_records():
    records = []
    for fname, label in FILE_LABELS.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"[WARN] {path} not found, skip.")
            continue

        # 原文件使用 ; 分隔
        df = pd.read_csv(path, sep=";")

        # 确保所有字段都存在
        for col in FIELD_ORDER:
            if col not in df.columns:
                df[col] = ""

        for _, row in df.iterrows():
            records.append((build_log_line(row.to_dict()), label))

        print(f"[INFO] Loaded {len(df)} rows from {fname} (label={label})")

    print(f"[INFO] Total log lines: {len(records)}")
    return records


# ===========================
#   生成 Session
# ===========================

def build_sessions(records):
    """
    把 [(log_line, item_label), ...] 切成固定窗口的 session。
    每个 session 100 行（SESSION_SIZE），只要有一行 item_label=1 => Label=1。
    丢弃不足 100 的尾巴。
    """
    session_list = []
    total = len(records)
    full = total // SESSION_SIZE
    print(f"[INFO] Total lines={total}, sessions={full}, dropped={total-full*SESSION_SIZE}")

    for i in range(0, full * SESSION_SIZE, SESSION_SIZE):
        chunk = records[i:i+SESSION_SIZE]
        logs   = [x[0] for x in chunk]
        labels = [x[1] for x in chunk]

        # 只要里面有攻击行，就视为异常 session
        seq_label = 1 if any(labels) else 0
        content = " ;-; ".join(logs)
        session_list.append([content, str(seq_label), SESSION_SIZE])

    return session_list


# ===========================
#   保存 CSV（用 csv.writer，加双引号）
# ===========================

def save_csv(path, rows):
    """
    输出格式：
    "['08:13:16.51','63099.518019',... ],  ;-; ['...']", "0", "100"
    内部只含单引号，不会产生 "" 问题。
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["Content", "Label", "session_length"])
        for r in rows:
            w.writerow(r)
    print(f"[OK] Saved: {path} ({len(rows)} sessions)")


# ===========================
#   主流程
# ===========================

if __name__ == "__main__":
    records = load_all_records()
    sessions = build_sessions(records)

    # 80/20 分割
    train, test = train_test_split(
        sessions,
        test_size=1-TRAIN_RATIO,
        shuffle=True,
        random_state=42
    )

    save_csv(os.path.join(OUT_DIR, "train.csv"), train)
    save_csv(os.path.join(OUT_DIR, "test.csv"),  test)

    print("\n[ALL DONE] 🎉 ICS → LogLLM list-style 数据准备完成！\n")
