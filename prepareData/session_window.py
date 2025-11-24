import os
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# from prepareData.helper import structure_log

try:
    from prepareData.helper import structure_log  # when run as module: python -m prepareData.session_window
except ModuleNotFoundError:
    from helper import structure_log  # when run as script: python prepareData/session_window.py

data_dir = r'/data/fangly/shqxBS/data/HDFS_v1'
log_name = "HDFS.log"

output_dir = data_dir


if __name__ == '__main__':
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
    structure_log(data_dir, output_dir, log_name, log_format)

    spliter = ' ;-; '
    train_ratio = 0.8

    log_structured_file = os.path.join(output_dir, log_name + "_structured.csv")

    df = pd.read_csv(log_structured_file, engine='c',
            na_filter=False, memory_map=True, dtype={'Date':object, "Time": object})

    print(f'number of messages in {log_structured_file} is {len(df)}')
    # df = df[:100000]

    data_dict_content = defaultdict(list) #preserve insertion order of items
    for idx, row in tqdm(df.iterrows(),total=len(df)):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            data_dict_content[blk_Id].append(row["Content"])

    data_df = pd.DataFrame(list(data_dict_content.items()), columns=['BlockId', 'Content'])

    blk_label_dict = {}
    blk_label_file = os.path.join(data_dir, "anomaly_label.csv")
    blk_df = pd.read_csv(blk_label_file)
    for _, row in tqdm(blk_df.iterrows(),total=len(blk_df)):
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    data_df["Label"] = data_df["BlockId"].apply(lambda x: blk_label_dict.get(x))  # add label to the sequence of each blockid

    train_len = int(train_ratio * len(data_df))

    data_df = data_df.sample(frac=1).reset_index(drop=True)  ##shuffle

    session_train_df = data_df[:train_len]

    session_test_df = data_df[train_len:]
    session_test_df = session_test_df.reset_index(drop=True)

    session_train_df['session_length'] = session_train_df["Content"].apply(len)
    session_train_df["Content"] = session_train_df["Content"].apply(lambda x: spliter.join(x))

    mean_session_train_len = session_train_df['session_length'].mean()
    max_session_train_len = session_train_df['session_length'].max()
    num_anomalous_train = session_train_df['Label'].sum()
    num_normal_train = len(session_train_df['Label']) - session_train_df['Label'].sum()

    session_test_df['session_length'] = session_test_df["Content"].apply(len)
    session_test_df["Content"] = session_test_df["Content"].apply(lambda x: spliter.join(x))

    mean_session_test_len = session_test_df['session_length'].mean()
    max_session_test_len = session_test_df['session_length'].max()
    num_anomalous_test = session_test_df['Label'].sum()
    num_normal_test = len(session_test_df['Label']) - session_test_df['Label'].sum()

    session_train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    session_test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print('Train dataset info:')
    print(f"max session length: {max_session_train_len}; mean session length: {mean_session_train_len}\n")
    print(
        f"number of anomalous sessions: {num_anomalous_train}; number of normal sessions: {num_normal_train}; number of total sessions: {len(session_train_df['Label'])}\n")

    print('Test dataset info:')
    print(f"max session length: {max_session_test_len}; mean session length: {mean_session_test_len}\n")
    print(
        f"number of anomalous sessions: {num_anomalous_test}; number of normal sessions: {num_normal_test}; number of total sessions: {len(session_test_df['Label'])}\n")

    with open(os.path.join(output_dir, 'train_info.txt'), 'w') as file:
        # 写入内容到文件
        file.write(f"max session length: {max_session_train_len}; mean session length: {mean_session_train_len}\n")
        file.write(
            f"number of anomalous sessions: {num_anomalous_train}; number of normal sessions: {num_normal_train}; number of total sessions: {len(session_train_df['Label'])}\n")

    with open(os.path.join(output_dir, 'test_info.txt'), 'w') as file:
        # 写入内容到文件
        file.write(f"max session length: {max_session_test_len}; mean session length: {mean_session_test_len}\n")
        file.write(
            f"number of anomalous sessions: {num_anomalous_test}; number of normal sessions: {num_normal_test}; number of total sessions: {len(session_test_df['Label'])}\n")

