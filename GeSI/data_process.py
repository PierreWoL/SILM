"""import os
import pandas as pd

# 设置你的CSV文件所在的文件夹路径
folder_path = 'E:\datasets\hospital'  # 替换为实际路径

# 获取所有CSV文件名
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 存储文件名和行数
summary = []

# 遍历并读取每个文件的行数
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    try:
        df = pd.read_csv(file_path, encoding='latin1')
        num_rows = len(df)
    except Exception as e:
        num_rows = '读取失败'
    summary.append({'filename': file, 'row_count': num_rows})

# 转为DataFrame并保存为CSV
summary_df = pd.DataFrame(summary)
output_path = os.path.join(folder_path, 'E:\datasets\summary.csv')
summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"统计结果已保存为: {output_path}")"""

import os

import pandas as pd

# 文件夹路径
folder_A = r"E:\newCode-left\newCode\datasets\GDS\Test"
folder_B = r"E:\Project\datasets\GoogleSearch\Test"

# 获取两个文件夹中的文件名集合（不包括路径，只是名字）
files_A = set(os.listdir(folder_A))
files_B = set(os.listdir(folder_B))

# 找出重合的文件名
common_files = files_A.intersection(files_B)

# 输出结果
print("两个文件夹中重合的文件名如下：")
for filename in sorted(common_files):
    print(filename)
ground_truth_path = r"E:\Project\datasets\GoogleSearch\groundTruth.csv"
df = pd.read_csv(ground_truth_path)

# 假设文件名在 'fileName' 列中（无扩展名），保留不在 common_files 中的行
common_files_no_ext = {os.path.splitext(f)[0] for f in common_files}
print(common_files_no_ext)
filtered_df = df[~df['fileName'].isin(common_files_no_ext)]

# 保存结果
output_path = r"E:\Project\datasets\GoogleSearch\filter_output.csv"
filtered_df.to_csv(output_path, index=False)

print(f"已保存过滤后的结果到：{output_path}")