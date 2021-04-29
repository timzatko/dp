import pandas as pd
import os


class CSVSaver:
    def __init__(self, fpath, col_index):
        if os.path.exists(fpath):
            self.df = pd.read_csv(fpath)
        else:
            self.df = None
        self.fpath = fpath
        self.col_index = col_index
        
    def add_row(self, row):
        if self.col_index not in row:
            raise Exception(f"no col {self.col_index} in row")
        if self.df is not None:
            self.df = self.df[self.df[self.col_index] != row[self.col_index]]
            self.df = self.df.append(row, ignore_index=True)
        else:
            data = {}
            for col, value in row.items():
                data[col] = [value]
            self.df = pd.DataFrame(data=data)
        print(f'saving to {self.fpath}')
        self.df.to_csv(self.fpath, index=False)
    