import pandas as pd
import os
"""
# Define the output dataframe with the specified columns
output_df = pd.DataFrame(columns=['fileName', 'Attribute Name', 'Attribute Values', 'Attribute Type', 'Table Type', 'Table Top Level Type'])

target_path = "datasets/WDC/Test"
ground_Truth = "datasets/WDC/groundTruth.csv"

gt = pd.read_csv(ground_Truth,index_col=0)
# Iterate through all the files in the current directory
for filename in os.listdir(target_path):
    if filename.endswith('.csv'):
        # Read the csv file into a dataframe
        df = pd.read_csv(os.path.join(target_path,filename ))
        if filename in gt.index:
            # Iterate through each column
            for column in df.columns:
                # Concatenate the column values into a single string, separated by commas
                if isinstance(df[column][0],str) and len(df[column][0])>100:
                        attribute_values = ', '.join(df[column].head(0).astype(str).to_list())
                else:
                    attribute_values = ', '.join(df[column].head(50).astype(str).to_list())

                # Append a new row to the output dataframe
                output_df = output_df._append({
                    'fileName': filename,
                    'Attribute Name': column,
                    'Attribute Values': attribute_values,
                    'Attribute Type': '',  # Left blank as per the requirement
                    'Table Type': gt.loc[filename, gt.columns[0]],  # Left blank as per the requirement
                    'Table Top Level Type': gt.loc[filename, gt.columns[1]]
                }, ignore_index=True)

# Write the output dataframe to "WDC.csv"
output_df.to_csv('datasets/WDC/WDC.csv', index=False)"""
p1="datasets/TabFact/groundTruth.csv"
p2="datasets/TabFact/column_gt.csv"
#class	superclass  LowestClass

df1 = pd.read_csv(p1, encoding="latin1")
df2 = pd.read_csv(p2, encoding="latin1")
for index, row in df1.iterrows():
    fileName = row.loc["fileName"]
    filtered_rows = df2[df2['fileName'].isin([fileName])]
    if len(filtered_rows)>0:

        df1.iloc[index, 1] = filtered_rows["LowestClass"].unique()[0]
        df1.iloc[index, 2] = filtered_rows["TopClass"].unique()[0]
df1.to_csv(p1)


