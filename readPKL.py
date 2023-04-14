import pickle

F=open('/Users/user/My Drive/CurrentDataset/result/embedding/starmie/vectors/SOTAB/cl_shuffle_col,sample_row_head_column_0_none.pkl','rb')

content=pickle.load(F)
print(content,type(content))