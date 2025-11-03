import pandas as pd

# 读取文件
file_path = 'E:/newCode-left/summoner/Summoner/datasets/WDC/column_gt.csv'
df = pd.read_csv(file_path, encoding='latin1')

# 获取 ColumnLabel 列的唯一值
unique_values = df['ColumnLabel'].unique()
print(len(unique_values))
import os
import pandas as pd

# 指定文件夹路径
folder_path = r"E:/newCode-left/summoner/Summoner/datasets/GDS/Test"

total_columns = 0  # 列总数
file_counts = 0    # 文件数

for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        #try:
        df = pd.read_csv(file_path, encoding='latin1')
        num_columns = len(df.columns)
        total_columns += num_columns
        file_counts += 1
            #print(f"{file_name}: {num_columns} 列")
        #except Exception as e:
            #print(f"{file_name}: 读取失败 ({e})")
print(f"所有文件列数总和: {total_columns}")
