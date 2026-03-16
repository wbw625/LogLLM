import csv
import os

def export_pre_prompt_to_txt():
    # 输入文件路径（读取其中一个风机文件获取表头即可）
    input_file = '/data/fangly/shqxBS/log/data/Six-Month/data/WT01_data.csv'
    # 输出的 txt 文件路径
    output_file = '/data/fangly/shqxBS/log/data/Six-Month/pre_prompt.txt'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(input_file):
        print(f"找不到文件: {input_file}。请检查路径是否正确。")
        return

    print(f"正在读取表头: {input_file} ...")

    with open(input_file, 'r', encoding='utf-8') as f:
        # 使用普通的 csv.reader 只读第一行即可
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            print("文件为空，无法读取表头。")
            return

        # 1. 格式化表头：为每个字段加上双引号，并处理缩进换行
        fields_formatted = ",\n        ".join([f'"{h}"' for h in headers])

        # 2. 组装完整的 pre_prompt
        # 注意大括号 {{ 和 }} 是为了在 f-string 中转义输出真实的 { 和 }
        pre_prompt = f"""Below is a sequence of wind turbine SCADA operational data and status logs recorded at 10-minute intervals:
{{
    "fields": [
        {fields_formatted}
    ],
    "data": [
"""

        # 3. 写入到 txt 文件
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write(pre_prompt)

        print(f"完整的 pre_prompt 已成功提取！")
        print(f"文件已保存至：{output_file}")

if __name__ == "__main__":
    export_pre_prompt_to_txt()