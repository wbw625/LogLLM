import pandas as pd
import os

# 1. 替换为你下载的 Excel 文件的实际路径
excel_file_path = '/data/fangly/shqxBS/log/data/Six-Month/download/SCADA_monitoring_outliers_filter.xlsx' 
# 2. 指定你想把拆分后的 CSV 保存到哪个文件夹
output_dir = '/data/fangly/shqxBS/log/data/Six-Month/outlier/'

# 如果输出文件夹不存在，则自动创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"正在读取完整的 Excel 文件: {excel_file_path}")
print("（这可能需要几分钟时间，请耐心等待...）")

# sheet_name=None 是关键！它会一次性读取 Excel 中的所有工作表，
# 并返回一个字典：{ 'Sheet名字': DataFrame }
all_sheets = pd.read_excel(excel_file_path, sheet_name=None)

print(f"成功读取！共发现 {len(all_sheets)} 个工作表。开始提取...")

# 遍历字典，将每个 DataFrame 分别保存为 CSV
for sheet_name, df in all_sheets.items():
    # 文档中提到 Sheet 名字本身可能就带有 .csv 后缀
    # 为了避免保存成 WT01_logs.csv.csv，我们先清理一下名字
    clean_name = sheet_name.replace('.csv', '')
    output_csv_path = os.path.join(output_dir, f"{clean_name}.csv")
    
    print(f"正在保存 -> {output_csv_path} (共 {len(df)} 行数据)")
    
    # 保存为纯文本 CSV，index=False 防止写入多余的行号列
    df.to_csv(output_csv_path, index=False, encoding='utf-8')

print("\n🎉 全部拆分完成！")
print(f"你可以前往 {output_dir} 目录查看独立的日志和传感器文件了。")