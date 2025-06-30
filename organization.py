import pandas as pd
import os

class organize_data:
    def __init__(self):
        self.df = None

    def load_file(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".csv":
            self.df = pd.read_csv(filepath)
        elif ext in [".xls", ".xlsx"]:
            self.df = pd.read_excel(filepath)
        elif ext == ".json":
            self.df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        print(f"Loaded data with shape {self.df.shape}")
        return self.df

    def clean_data(self):
        if self.df is None:
            raise ValueError("No data loaded")
        self.df.dropna(how="all", inplace=True)
        self.df.dropna(axis=1, how="all", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        print("Removed empty rows/cols, reset index")
        return self.df

    def get_df(self):
        return self.df