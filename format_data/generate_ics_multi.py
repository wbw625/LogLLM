import os
import json
import csv
import random

# 原始 ICS 数据集路径
BASE_DIR = "/data/fangly/shqxBS/data/ics"

# 输出路径
OUT_DIR = "/data/fangly/shqxBS/w/data/ICS_log_multi"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.9


# ------- 新标签映射规则（基于合并后的6类） -----------
CATEGORY_MAP = {
    # scanning group
    "scanning_mms": "industrial_network_scanning_attack",
    "vertical_scanning": "industrial_network_scanning_attack",
    "horizontal_scanning": "industrial_network_scanning_attack",
    "mms_scanning": "industrial_network_scanning_attack",

    # MITM modification group
    "mitm_c_dc_modify": "man_in_the_middle_attack",
    "mitm_dco_modification": "man_in_the_middle_attack",
    "mitm_asdu100_unknown_cause": "man_in_the_middle_attack",
    "mitm_value_modification": "man_in_the_middle_attack",
    "mitm_c_ic_resp_mapping": "man_in_the_middle_attack",
    "packet_modification": "man_in_the_middle_attack",
    "masquerading_hmi_ic_cycle": "man_in_the_middle_attack",

    # replay
    "replay_c_ic_actconf": "replay_attack",
    "replay_interrogation_response": "replay_attack",

    # denial of service
    "dos": "denial_of_service_attack",

    # injection / rogue / unauthorized control
    "injection_file_transfer": "unauthorized_command_injection_or_rogue_control_attack",
    "injection_single_command": "unauthorized_command_injection_or_rogue_control_attack",
    "massive_control_commands": "unauthorized_command_injection_or_rogue_control_attack",
    "malicious_write": "unauthorized_command_injection_or_rogue_control_attack",
    "switching_attack": "unauthorized_command_injection_or_rogue_control_attack",
    "cancel_or_error_injection": "unauthorized_command_injection_or_rogue_control_attack",
    "rogue_device": "unauthorized_command_injection_or_rogue_control_attack",
    "lost_connection": "unauthorized_command_injection_or_rogue_control_attack",
}


def normalize_field(name: str) -> str:
    """字段名去掉空格。"""
    return str(name).replace(" ", "")


def build_log_line_from_row(row, fields):
    """
    把一行 data（list）转成：
    字段1: 值1, 字段2: 值2, ...
    fields 来自每个 json 的 input.fields（已去空格）
    """
    parts = []
    n = min(len(row), len(fields))
    for i in range(n):
        parts.append(f"{fields[i]}: {str(row[i])}")
    for i in range(n, len(fields)):
        parts.append(f"{fields[i]}: ")
    return ", ".join(parts)


def map_label(json_data, folder_type):
    if folder_type == "normal":
        return "normal"
    attack_type = json_data.get("output", {}).get("attack_type", "").strip()
    return CATEGORY_MAP.get(attack_type, "unauthorized_command_injection_or_rogue_control_attack")


def collect_samples():
    samples = []

    for dirpath, _, filenames in os.walk(BASE_DIR):
        folder = os.path.basename(dirpath)
        if folder not in ["normal", "attack"]:
            continue

        print(f"[INFO] Reading {folder.upper()} → {dirpath}")

        for fname in sorted(filenames):
            if not fname.endswith(".json"):
                continue

            path = os.path.join(dirpath, fname)

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load JSON {path}: {e}")
                continue

            input_obj = data.get("input", {})
            fields = input_obj.get("fields", None)
            rows = input_obj.get("data", [])

            if not isinstance(fields, list) or len(fields) == 0:
                print(f"[WARN] Missing/invalid input.fields in {path}, skip.")
                continue
            if not rows:
                continue

            # ✅ 字段名去空格（每个json独立）
            fields = [normalize_field(x) for x in fields]

            content = " ;-; ".join(build_log_line_from_row(r, fields) for r in rows)

            label = map_label(data, folder)
            session_len = len(rows)

            samples.append((content, label, session_len))

    print(f"[DONE] Loaded {len(samples)} samples.")
    return samples


def save_dataset(samples):
    random.shuffle(samples)
    split_idx = int(len(samples) * TRAIN_RATIO)

    train, test = samples[:split_idx], samples[split_idx:]

    def write(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["Content", "Label", "session_length"])
            writer.writerows(rows)
        print(f"[OK] Saved: {path} ({len(rows)} records)")

    write(os.path.join(OUT_DIR, "train.csv"), train)
    write(os.path.join(OUT_DIR, "test.csv"), test)


if __name__ == "__main__":
    samples = collect_samples()
    save_dataset(samples)
    print("\n[🚀 Completed] Dataset successfully built in multi-class format.\n")
