import pickle

F=open(r'C:\Users\1124a\OneDrive - The University of Manchester\BaselineCode\CurrentDataset\result\embedding\starmie\vectors\open_data\cl_shuffle_col,sample_row_head_column_0_none.pkl','rb')

content=pickle.load(F)
print(content,type(content))