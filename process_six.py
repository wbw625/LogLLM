import csv
import os
import json
import random  # 新增：导入随机模块

def get_prompts(feature_headers):
    # 为每一个保留下来的特征加上双引号并格式化
    fields_formatted = ",\n        ".join([f'"{h}"' for h in feature_headers])
    
    pre_prompt = f"""Below is a sequence of wind turbine SCADA operational data and status logs recorded at 10-minute intervals:
{{
    "fields": [
        {fields_formatted}
    ],
    "data": [
"""
    post_prompt = """    ]
}
Is this operational sequence normal or anomalous?
"""
    return pre_prompt, post_prompt

def process_wind_turbine_data():
    base_dir = '/data/fangly/shqxBS/log/data/Six-Month/data'
    train_file = '/data/fangly/shqxBS/log/data/Six-Month/train.csv'
    test_file = '/data/fangly/shqxBS/log/data/Six-Month/test.csv'

    input_files = [os.path.join(base_dir, f'WT{i:02d}_data.csv') for i in range(1, 11)]
    os.makedirs(os.path.dirname(train_file), exist_ok=True)

    # 【重要】：需要从喂给模型的文本中剔除的标签列/泄露列
    exclude_cols = {
        "System Logs First Active Alarm No",
        "First Alarm parameter 1 in 10 min frame",
        "First Alarm parameter 2 in 10 min frame",
        "missing_data",
        "malicious"
    }

    all_train_data = []
    all_test_data = []

    # ================= 配置区 =================
    session_length = 50
    normal_stride = 50
    abnormal_stride = 2
    
    # 【新增参数】：训练集中正常样本的丢弃比例 (0.0 到 1.0 之间)
    # 例如：0.8 表示随机删掉 80% 的正常样本，只保留 20%
    normal_drop_rate = 0.8
    # ==========================================

    print("session_length: {}, normal_stride: {}, abnormal_stride: {}, normal_drop_rate: {}".format(
        session_length, normal_stride, abnormal_stride, normal_drop_rate))

    print("开始处理风机数据：组装 JSON Prompt 并剔除数据泄露列...\n")

    for file_path in input_files:
        if not os.path.exists(file_path):
            continue
            
        valid_rows = []
        feature_headers = []
        pre_prompt, post_prompt = "", ""

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            # 初始化时过滤掉不能泄露给模型的列
            if not feature_headers:
                feature_headers = [h for h in headers if h not in exclude_cols]
                pre_prompt, post_prompt = get_prompts(feature_headers)

            for row in reader:
                # 1. 判断是否丢弃该行
                if str(row.get('missing_data', '')).strip().lower() == 'true':
                    continue

                # 2. 判断 Label 
                try:
                    alarm_no = float(row.get('System Logs First Active Alarm No', 0.0))
                    param1 = float(row.get('First Alarm parameter 1 in 10 min frame', 0.0))
                    param2 = float(row.get('First Alarm parameter 2 in 10 min frame', 0.0))
                    cond_alarm = (alarm_no != 0.0 and param1 != 0.0 and param2 != 0.0)
                except ValueError:
                    cond_alarm = False
                
                cond_malicious = str(row.get('malicious', 'false')).strip().lower() == 'true'
                current_label = 1 if (cond_alarm or cond_malicious) else 0

                # 3. 仅提取干净的特征列组装成 JSON 数组的一行
                row_values = [str(row.get(h, '')) for h in feature_headers]
                row_json_str = "        " + json.dumps(row_values)
                
                valid_rows.append((row_json_str, current_label))

        # 4. 动态步长滑动窗口截取
        file_samples = []
        i = 0
        total_rows = len(valid_rows)

        while i <= total_rows - session_length:
            window = valid_rows[i : i + session_length]
            is_abnormal = any(label == 1 for _, label in window)
            
            # 将 50 行干净的 JSON 数据用逗号和换行拼起来
            window_data_str = ",\n".join([row_str for row_str, _ in window]) + "\n"
            
            # 组合成最终的完整 Content
            full_content = pre_prompt + window_data_str + post_prompt
            session_label = 1 if is_abnormal else 0
            
            file_samples.append({
                "Content": full_content,
                "Label": session_label,
                "session_length": session_length
            })
            
            if is_abnormal:
                i += abnormal_stride
            else:
                i += normal_stride

        # 5. 针对【当前风机】按 8:2 划分
        total_file_samples = len(file_samples)
        if total_file_samples > 0:
            split_index = int(total_file_samples * 0.8)
            all_train_data.extend(file_samples[:split_index])
            all_test_data.extend(file_samples[split_index:])

    # ================= 新增：降采样逻辑 =================
    if normal_drop_rate > 0.0:
        print(f"\n执行训练集正常样本降采样... (丢弃比例: {normal_drop_rate*100}%)")
        random.seed(42)  # 固定随机种子，保证每次运行结果一致，方便实验对比
        downsampled_train_data = []
        for item in all_train_data:
            if item["Label"] == 1:
                # 异常样本 100% 保留
                downsampled_train_data.append(item)
            else:
                # 正常样本根据设置的概率决定是否保留
                if random.random() >= normal_drop_rate:
                    downsampled_train_data.append(item)
        
        # 将训练集替换为降采样后的数据集
        all_train_data = downsampled_train_data
    # ====================================================

    # 6. 全局统计正负样本数量
    train_normal = sum(1 for item in all_train_data if item["Label"] == 0)
    train_abnormal = sum(1 for item in all_train_data if item["Label"] == 1)
    test_normal = sum(1 for item in all_test_data if item["Label"] == 0)
    test_abnormal = sum(1 for item in all_test_data if item["Label"] == 1)

    # 7. 统一导出到 CSV
    def export_csv(out_path, data):
        if not data:
            return
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Content", "Label", "session_length"], 
                                    quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(data)

    print("\n正在将合并后的数据写入硬盘...")
    export_csv(train_file, all_train_data)
    export_csv(test_file, all_test_data)

    print(f"\n================ 全局数据导出成功 ================")
    print(f"【训练集 (Train)】总数: {len(all_train_data)} 条 -> {train_file}")
    print(f"   ├─ 正常 (Label 0): {train_normal} 条")
    print(f"   └─ 异常 (Label 1): {train_abnormal} 条")
    print(f"【测试集 (Test)】 总数: {len(all_test_data)} 条 -> {test_file}")
    print(f"   ├─ 正常 (Label 0): {test_normal} 条")
    print(f"   └─ 异常 (Label 1): {test_abnormal} 条")
    print(f"==================================================")

if __name__ == "__main__":
    process_wind_turbine_data()