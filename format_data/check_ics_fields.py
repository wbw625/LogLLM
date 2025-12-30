import os
import json

BASE_DIR = "/data/fangly/shqxBS/data/ics"

EXPECTED_FIELDS = [
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

def check_fields_in_json(root_dir: str):
    total_files = 0
    matched_files = 0
    mismatched_files = 0
    missing_fields_files = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith(".json"):
                continue

            fpath = os.path.join(dirpath, fname)
            total_files += 1

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load JSON: {fpath}: {e}")
                mismatched_files += 1
                continue

            # 期望结构：data["input"]["fields"]
            input_obj = data.get("input", None)
            if input_obj is None or "fields" not in input_obj:
                print(f"[WARN] No 'input.fields' in file: {fpath}")
                missing_fields_files += 1
                continue

            fields = input_obj["fields"]

            if fields == EXPECTED_FIELDS:
                matched_files += 1
            else:
                mismatched_files += 1
                print(f"[MISMATCH] {fpath}")
                # print("  actual fields:", fields)
                # print("  expected     :", EXPECTED_FIELDS)
                print()

    print("\n========== CHECK SUMMARY ==========")
    print(f"Total JSON files   : {total_files}")
    print(f"Matched fields     : {matched_files}")
    print(f"Mismatched fields  : {mismatched_files}")
    print(f"Missing 'fields'   : {missing_fields_files}")
    print("===================================")


if __name__ == "__main__":
    check_fields_in_json(BASE_DIR)
