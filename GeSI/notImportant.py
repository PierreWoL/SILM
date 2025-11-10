import pandas as pd

df = pd.read_csv("E:/newCode-left/newCode/datasets/WDC/column_gt.csv", encoding='latin1')

result = df.groupby(["TopClass", "ColumnLabel"]).size().reset_index(name="count")
result_dict = (
    result.groupby("TopClass")
    .apply(lambda g: dict(zip(g["ColumnLabel"], g["count"])))
    .to_dict()
)# 读入 DataFrame
sorted_result_dict = {
    top: dict(sorted(cols.items(), key=lambda x: x[1], reverse=True))
    for top, cols in result_dict.items()
}
final_result_dict = {
    top.strip("[]'"): {k.title(): v for k, v in cols.items()}
    for top, cols in sorted_result_dict.items()
}


"""for key, value in final_result_dict.items():
    print(key)
    for K,V in value.items():
        print(K,f"({V})")
"""


# 目标 fileName 列表
target_files = [
    "SOTAB_136.csv", "SOTAB_16.csv", "SOTAB_167.csv", "SOTAB_171.csv",
    "SOTAB_201.csv", "SOTAB_22.csv", "SOTAB_24.csv", "SOTAB_25.csv",
    "SOTAB_44.csv", "SOTAB_70.csv", "SOTAB_85.csv", "SOTAB_95.csv",
    "T2DV2_115.csv", "T2DV2_116.csv", "T2DV2_134.csv", "T2DV2_135.csv",
    "T2DV2_136.csv", "T2DV2_143.csv", "T2DV2_144.csv", "T2DV2_146.csv",
    "T2DV2_15.csv", "T2DV2_150.csv", "T2DV2_166.csv", "T2DV2_17.csv",
    "T2DV2_174.csv", "T2DV2_181.csv", "T2DV2_199.csv", "T2DV2_214.csv",
    "T2DV2_218.csv", "T2DV2_22.csv", "T2DV2_233.csv", "T2DV2_237.csv",
    "T2DV2_246.csv", "T2DV2_249.csv", "T2DV2_25.csv", "T2DV2_250.csv",
    "T2DV2_37.csv", "T2DV2_47.csv", "T2DV2_48.csv", "T2DV2_49.csv",
    "T2DV2_50.csv", "T2DV2_51.csv", "T2DV2_54.csv", "T2DV2_55.csv",
    "T2DV2_56.csv", "T2DV2_6.csv", "T2DV2_69.csv", "T2DV2_76.csv",
    "T2DV2_81.csv", "T2DV2_87.csv", "T2DV2_9.csv", "T2DV2_96.csv",
    "T2DV2_97.csv", "Test_corpus_30.csv", "Test_corpus_48.csv",
    "Test_corpus_54.csv", "Test_corpus_69.csv", "Test_corpus_76.csv",
    "Test_corpus_8.csv", "Test_corpus_83.csv", "Test_corpus_91.csv"
]
target_files1 = [
    "SOTAB_26.csv","SOTAB_151.csv","T2DV2_33.csv","SOTAB_77.csv","T2DV2_102.csv",
    "SOTAB_192.csv","Test_corpus_57.csv","SOTAB_194.csv","T2DV2_213.csv","T2DV2_89.csv",
    "T2DV2_119.csv","T2DV2_188.csv","Test_corpus_26.csv","Test_corpus_108.csv","Test_corpus_42.csv",
    "Test_corpus_86.csv","Test_corpus_34.csv","T2DV2_46.csv","T2DV2_205.csv","Test_corpus_14.csv",
    "Test_corpus_89.csv","Test_corpus_49.csv","Test_corpus_7.csv","Test_corpus_60.csv",
    "Test_corpus_78.csv","Test_corpus_79.csv","SOTAB_221.csv","Test_corpus_92.csv",
    "Test_corpus_31.csv","Test_corpus_82.csv","Test_corpus_106.csv","Test_corpus_115.csv",
    "T2DV2_114.csv","Test_corpus_135.csv","Test_corpus_63.csv","Test_corpus_144.csv",
    "T2DV2_253.csv"
]
# 过滤出 fileName 在目标列表中的行
filtered = df[df["fileName"].isin(target_files)]

# 统计 columnLabel 的唯一值和计数
stats = filtered["ColumnLabel"].value_counts().reset_index()
stats.columns = ["ColumnLabel", "count"]


for index, row in stats.iterrows():
    count = row["count"]
    print(row["ColumnLabel"],f"({count})")