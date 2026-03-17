import os

# 定义数据所在的目录路径
data_dir = '/data/fangly/shqxBS/log/download/Morris/'

# 定义预期的标准表头 (去除首尾空格/换行符)
expected_header = "R1-PA1:VH,R1-PM1:V,R1-PA2:VH,R1-PM2:V,R1-PA3:VH,R1-PM3:V,R1-PA4:IH,R1-PM4:I,R1-PA5:IH,R1-PM5:I,R1-PA6:IH,R1-PM6:I,R1-PA7:VH,R1-PM7:V,R1-PA8:VH,R1-PM8:V,R1-PA9:VH,R1-PM9:V,R1-PA10:IH,R1-PM10:I,R1-PA11:IH,R1-PM11:I,R1-PA12:IH,R1-PM12:I,R1:F,R1:DF,R1-PA:Z,R1-PA:ZH,R1:S,R2-PA1:VH,R2-PM1:V,R2-PA2:VH,R2-PM2:V,R2-PA3:VH,R2-PM3:V,R2-PA4:IH,R2-PM4:I,R2-PA5:IH,R2-PM5:I,R2-PA6:IH,R2-PM6:I,R2-PA7:VH,R2-PM7:V,R2-PA8:VH,R2-PM8:V,R2-PA9:VH,R2-PM9:V,R2-PA10:IH,R2-PM10:I,R2-PA11:IH,R2-PM11:I,R2-PA12:IH,R2-PM12:I,R2:F,R2:DF,R2-PA:Z,R2-PA:ZH,R2:S,R3-PA1:VH,R3-PM1:V,R3-PA2:VH,R3-PM2:V,R3-PA3:VH,R3-PM3:V,R3-PA4:IH,R3-PM4:I,R3-PA5:IH,R3-PM5:I,R3-PA6:IH,R3-PM6:I,R3-PA7:VH,R3-PM7:V,R3-PA8:VH,R3-PM8:V,R3-PA9:VH,R3-PM9:V,R3-PA10:IH,R3-PM10:I,R3-PA11:IH,R3-PM11:I,R3-PA12:IH,R3-PM12:I,R3:F,R3:DF,R3-PA:Z,R3-PA:ZH,R3:S,R4-PA1:VH,R4-PM1:V,R4-PA2:VH,R4-PM2:V,R4-PA3:VH,R4-PM3:V,R4-PA4:IH,R4-PM4:I,R4-PA5:IH,R4-PM5:I,R4-PA6:IH,R4-PM6:I,R4-PA7:VH,R4-PM7:V,R4-PA8:VH,R4-PM8:V,R4-PA9:VH,R4-PM9:V,R4-PA10:IH,R4-PM10:I,R4-PA11:IH,R4-PM11:I,R4-PA12:IH,R4-PM12:I,R4:F,R4:DF,R4-PA:Z,R4-PA:ZH,R4:S,control_panel_log1,control_panel_log2,control_panel_log3,control_panel_log4,relay1_log,relay2_log,relay3_log,relay4_log,snort_log1,snort_log2,snort_log3,snort_log4,marker"

def check_csv_headers():
    all_match = True
    print(f"正在检查目录: {data_dir}\n")

    # 遍历 data1.csv 到 data15.csv
    for i in range(1, 16):
        filename = f"data{i}.csv"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print(f"[错误] 文件不存在: {filename}")
            all_match = False
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # 读取第一行并去除末尾的换行符
                header = f.readline().strip()
                
                # 比较
                if header == expected_header:
                    print(f"[通过] {filename}: 表头一致")
                else:
                    print(f"[失败] {filename}: 表头不匹配!")
                    print(f"       实际表头: {header[:50]}... (显示前50字符)")
                    all_match = False
        except Exception as e:
            print(f"[错误] 读取文件 {filename} 时出错: {e}")
            all_match = False

    print("\n" + "="*30)
    if all_match:
        print("所有文件的表头均验证通过！")
    else:
        print("存在表头不一致或缺失的文件，请检查。")

if __name__ == "__main__":
    check_csv_headers()