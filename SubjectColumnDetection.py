from d3l.input_output.dataloaders import CSVDataLoader

class subjectColumn:

 def read_table(self, data_path):
     #  collection of tables
     dataloader = CSVDataLoader(
         root_path=(data_path),
         # sep=",",
         encoding='latin-1'
     )
