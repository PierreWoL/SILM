import matplotlib.pyplot as plt
import pandas as pd
from experimentalData import get_files
import os

"""
Method3_MODE2Sim0.3DistributionLSH_metrics
# Create a sample dataframe
data = {'year': [2010, 2011, 2012, 2013, 2014, 2015],
        'line1': [10, 15, 13, 17, 20, 25],
        'line2': [5, 10, 8, 12, 15, 20],
        'line3': [20, 25, 23, 27, 30, 35]}
df = pd.DataFrame(data)

# Plot the line chart
plt.plot(df['year'], df['line1'], label='Line 1')
plt.plot(df['year'], df['line2'], label='Line 2')
plt.plot(df['year'], df['line3'], label='Line 3')

# Add chart title and axis labels
plt.title('Multi-Line Chart')
plt.xlabel('Year')
plt.ylabel('Value')

# Add legend
plt.legend()

# Show the chart
plt.show()
"""
DATA_PATH = ['open_data', 'SOTAB', 'Test_corpus', 'T2DV2']
absolute_path = os.getcwd() + "/result/metrics/"
kinds = ["Distribution","Value","Format","Header","Embed"]
for kind in kinds:
    BIRCH_RI, Hierarchical_RI = {}, {}
    BIRCH_Purity, Hierarchical_Purity = {}, {}
    for dataset in DATA_PATH:
        BIRCH_RI[dataset] = []
        Hierarchical_RI[dataset] = []
        BIRCH_Purity[dataset] = []
        Hierarchical_Purity[dataset] = []
        result = absolute_path + dataset + "/Method3/MODE2/"
        files = get_files(result)
        print(files)
        column = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for file in files:
            file_last = file.lstrip("Method3_MODE2Sim")
            if kind in file_last or kind.lower() in file_last:
                # print(file_last,result + file + ".csv" )
                data = pd.read_csv(result + file + ".csv", header=0, index_col=0)
                # print( data.iloc[3, 0],data.iloc[3, 1],data.iloc[6, 0],data.iloc[6, 1])
                BIRCH_RI[dataset].append(data.iloc[3, 0])
                Hierarchical_RI[dataset].append(data.iloc[3, 1])
                BIRCH_Purity[dataset].append(data.iloc[6, 0])
                Hierarchical_Purity[dataset].append(data.iloc[6, 1])
    df_BIRCH_RI = pd.DataFrame.from_dict(BIRCH_RI, orient='index', columns=column)
    df_Hierarchical_RI = pd.DataFrame.from_dict(Hierarchical_RI, orient='index', columns=column)
    df_BIRCH_Purity = pd.DataFrame.from_dict(BIRCH_Purity, orient='index', columns=column)
    df_Hierarchical_Purity = pd.DataFrame.from_dict(Hierarchical_Purity, orient='index', columns=column)
    dfs = {'BIRCH Rand Index': df_BIRCH_RI,
           'HierarchicalClustering Rand Index': df_Hierarchical_RI,
           'BIRCH Purity': df_BIRCH_Purity,
           'HierarchicalClustering Purity': df_Hierarchical_Purity}
    print(df_BIRCH_RI, "\n", df_Hierarchical_RI, "\n", df_BIRCH_Purity, "\n", df_Hierarchical_Purity)
    for key, df in dfs.items():
        print(key, df)
        for data_path in DATA_PATH:
            plt.plot(df.columns, df.loc[data_path], label=data_path)
        # Add chart title and axis labels
        plt.title(key + " Tuning Similarity threshold in \n" + kind + " LSH indexes")
        plt.xlabel('Similarity threshold')
        plt.ylabel(key.split(' ', 1)[1])
        # Add legend
        plt.legend()
        plt.savefig("fig/" + kind + key + ".png")
        # Show the chart
        plt.show()

        plt.close()
"""
    

    # Plot the line chart
    

"""