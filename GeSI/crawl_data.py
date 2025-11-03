import csv

def rrf_to_csv(rrf_path, csv_path, header):
    with open(rrf_path, 'r', encoding='utf-8') as infile, \
         open(csv_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile, delimiter='|')
        writer = csv.writer(outfile)

        writer.writerow(header)  # 写入表头
        for row in reader:
            if row and row != ['']:
                writer.writerow(row[:-1])  # 去掉最后一个空字段（因为结尾有多余的 `|`）

# 示例：转换 RXNCONSO.RRF
header_rxnconso = [
    'RXCUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF',
    'RXAUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY',
    'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF'
]

rrf_to_csv('E:/datasets/hospital/RxNorm_full_prescribe_07072025/rrf/RXNSAT.RRF',
           'E:/datasets/hospital/RxNorm_full_prescribe_07072025/RXNSAT.csv', header_rxnconso)
