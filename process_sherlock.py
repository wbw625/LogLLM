import json
import re
import csv
import random
from datetime import datetime
import pytz

# ================= 配置路径 =================
# EVENTS_FILE = "/data/fangly/shqxBS/log/data/Sherlock/01-Basic/raw/test/events.jsonl"
# LOG_FILE = "/data/fangly/shqxBS/log/data/Sherlock/01-Basic/raw/test/log/mtu-n1-vcc-service.log"
# TRAIN_CSV = "/data/fangly/shqxBS/log/data/Sherlock/train01.csv"
# TEST_CSV = "/data/fangly/shqxBS/log/data/Sherlock/test01.csv"

EVENTS_FILE = "/data/fangly/shqxBS/log/data/Sherlock/02-Semiurban/raw/test/events.jsonl"
LOG_FILE = "/data/fangly/shqxBS/log/data/Sherlock/02-Semiurban/raw/test/log/mtu-n1-vcc-service.log"
TRAIN_CSV = "/data/fangly/shqxBS/log/data/Sherlock/train02.csv"
TEST_CSV = "/data/fangly/shqxBS/log/data/Sherlock/test02.csv"


# ================= 初始化设置 =================
# 匹配日志开头的格式：2025-03-25 16:58:23,170
log_time_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
# WATTSON 仿真器在德国，时区设置为欧洲中部时间
cet_tz = pytz.timezone('Europe/Berlin')

def main():
    print("1. 正在解析 Log 文件...")
    logs = []
    current_ts = 0.0
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            text = line.rstrip('\n')
            match = log_time_pattern.match(text)
            if match:
                time_str = match.group(1)
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
                dt_aware = cet_tz.localize(dt)
                current_ts = dt_aware.timestamp()
            
            logs.append({
                'index': i,
                'timestamp': current_ts,
                'text': text,
                'used': False  # 用于防止日志被重复使用
            })
    print(f"   共加载 {len(logs)} 行日志。")

    print("2. 正在解析 Events 文件...")
    type1_events = {}  # 记录有 mark (start/end) 的事件
    type2_events = []  # 记录没有 mark 的事件
    
    with open(EVENTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            ev = json.loads(line)
            ts = ev.get('timestamp')
            nd = ev.get('notification_data', {})
            malicious = nd.get('malicious', False)
            mark = nd.get('mark')
            ev_id = nd.get('id')
            
            if mark == 'start':
                if ev_id not in type1_events:
                    type1_events[ev_id] = {'start': ts, 'end': None, 'malicious': malicious}
                else:
                    type1_events[ev_id]['start'] = ts
            elif mark in ['end', 'recovery']:
                if ev_id in type1_events:
                    type1_events[ev_id]['end'] = ts
                else:
                    type1_events[ev_id] = {'start': None, 'end': ts, 'malicious': malicious}
            elif mark is None:
                type2_events.append({'timestamp': ts, 'malicious': malicious})

    sessions = []
    
    print("3. 提取 Type 1 (有明确起止时间窗) 的事件日志...")
    for ev_id, ev in type1_events.items():
        start_ts = ev.get('start')
        end_ts = ev.get('end')
        if start_ts and end_ts:
            collected_logs = []
            for log in logs:
                if log['timestamp'] > 0 and start_ts <= log['timestamp'] <= end_ts:
                    if not log['used']:
                        collected_logs.append(log)
                        # 达到 50 行即刻停止收集，确保是从 start 开始的前 50 行
                        if len(collected_logs) == 50:
                            break
            
            # 将收集到的日志标记为 used，并提取文本
            session_lines = []
            for log in collected_logs:
                log['used'] = True
                session_lines.append(log['text'])

            if session_lines:
                sessions.append({
                    'Content': ' ;-; '.join(session_lines),
                    'Label': 1 if ev['malicious'] else 0,
                    'session_length': len(session_lines)
                })

    print("4. 提取 Type 2 (无 mark, 截取前后50行) 的事件日志...")
    for ev in type2_events:
        closest_idx = -1
        min_diff = float('inf')
        for log in logs:
            if log['timestamp'] > 0:
                diff = abs(log['timestamp'] - ev['timestamp'])
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = log['index']
        
        if closest_idx != -1:
            start_idx = max(0, closest_idx - 25)
            end_idx = min(len(logs), closest_idx + 25)
            
            session_lines = []
            for i in range(start_idx, end_idx):
                if not logs[i]['used']:
                    session_lines.append(logs[i]['text'])
                    logs[i]['used'] = True
            if session_lines:
                sessions.append({
                    'Content': ' ;-; '.join(session_lines),
                    'Label': 1 if ev['malicious'] else 0,
                    'session_length': len(session_lines)
                })

    print("5. 提取剩余的无事件正常日志 (按50行分块 Label 0)...")
    bg_chunk = []
    for log in logs:
        if not log['used']:
            bg_chunk.append(log['text'])
            log['used'] = True
        
        # 只要满 50 行就切成一个 session
        if len(bg_chunk) == 50:
            sessions.append({
                'Content': ' ;-; '.join(bg_chunk),
                'Label': 0,
                'session_length': 50
            })
            bg_chunk = []
            
    # 处理文件末尾最后剩下的部分（不足50行的部分也记录下来）
    if len(bg_chunk) > 0:
        sessions.append({
            'Content': ' ;-; '.join(bg_chunk),
            'Label': 0,
            'session_length': len(bg_chunk)
        })

    print(f"   总共生成了 {len(sessions)} 个 Session 数据行。")

    print("6. 随机打乱并按 8:2 划分数据集...")
    random.seed(42) # 固定随机种子以保证可重复性
    random.shuffle(sessions)
    
    split_idx = int(len(sessions) * 0.8)
    train_sessions = sessions[:split_idx]
    test_sessions = sessions[split_idx:]

    def write_to_csv(filepath, data):
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Content', 'Label', 'session_length'], quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    write_to_csv(TRAIN_CSV, train_sessions)
    write_to_csv(TEST_CSV, test_sessions)

    # ================= 统计分布信息 =================
    total_label_1 = sum(1 for s in sessions if s['Label'] == 1)
    total_label_0 = sum(1 for s in sessions if s['Label'] == 0)

    train_label_1 = sum(1 for s in train_sessions if s['Label'] == 1)
    train_label_0 = sum(1 for s in train_sessions if s['Label'] == 0)

    test_label_1 = sum(1 for s in test_sessions if s['Label'] == 1)
    test_label_0 = sum(1 for s in test_sessions if s['Label'] == 0)

    print(f"\n✅ 处理完成！统计信息如下：")
    print(f"   - 【全集】总数据量: {len(sessions)} 行")
    print(f"           ▶ 正例 (异常攻击, Label=1): {total_label_1}")
    print(f"           ▶ 反例 (正常日志, Label=0): {total_label_0}")
    
    print(f"   - 【训练集】已保存至: {TRAIN_CSV} (共 {len(train_sessions)} 行)")
    print(f"           ▶ 包含正例: {train_label_1}")
    print(f"           ▶ 包含反例: {train_label_0}")

    print(f"   - 【测试集】已保存至: {TEST_CSV} (共 {len(test_sessions)} 行)")
    print(f"           ▶ 包含正例: {test_label_1}")
    print(f"           ▶ 包含反例: {test_label_0}")

if __name__ == "__main__":
    main()