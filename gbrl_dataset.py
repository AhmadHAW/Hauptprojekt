import os
import pandas as pd
import numpy as np

class GBRL_Dataset():
    
    def __init__(self, path, create = False):
        self.dataset_path = path + "\\datasets.csv"
        self.path = path
        if not (create or os.path.isdir(path)):
            raise Exception(f"Dataset dir {path} does not exist and set \"create = True\" for first init.")
        if not os.path.isdir(path):
            os.mkdir(path)
        if not os.path.isfile(self.dataset_path):
            self._init_dataset_dataframe()
        self.df = pd.read_csv(self.dataset_path)
                
    def _add_row(self, row, name, train_path, train_length, val_path, val_length, test_path, test_length, domain, task, run_tests):
        if row['name'] == name:
            row['train_path'] = train_path
            row['train_length'] = train_length
            row['val_path'] = val_path
            row['val_length'] = val_length
            row['test_path'] = test_path
            row['test_length'] = test_length
            row['domain'] = domain
            row['task'] = task
            row['run_tests'] = run_tests
        return row

    def add_dataset(self, name, task, domain = "CommonSense", train_df = None, val_df = None, test_df = None, run_tests = 0):
        if domain not in ["CommonSense", "Legal", "Ethics", "NatureScience"]:
            raise Exception(f"Expected domain of \"CommonSense\", \"Legal\", \"Ethics\", \"NatureScience\", but got \"{domain}\" instead.")
        train_path, train_length = self._generate_dataset_subset(name, train_df, "train")
        val_path, val_length = self._generate_dataset_subset(name, val_df, "val")
        test_path, test_length = self._generate_dataset_subset(name, test_df, "test")
        found_rows = self.df[self.df["name"] == name]
        if len(found_rows) != 1:
            new_row = {"name": name, "train_path": train_path, "train_length": train_length, "val_path": val_path, "val_length": val_length, "test_path": test_path, "test_length": test_length, "domain": domain, "task": task, "run_tests": run_tests}   
            self.df.loc[len(self.df)] = new_row
        else:
            self.df = self.df.apply(lambda row: self._add_row(row, name, train_path, train_length, val_path, val_length, test_path, test_length, domain, task, run_tests), axis = 1)
        self.df.to_csv(self.dataset_path, index=False)

    def chose_rows_for_test(self, df, n = 100):
        df = df.copy()
        df["test"] = False
        random_indices = np.random.choice(df.index, n, replace=False)
        df.loc[random_indices, "test"] = True
        return df


    def _generate_dataset_subset(self, name, df, type):
        if type not in ["train", "test", "val"]:
            raise Exception(f"Expected type of \"train\", \"test\" or \"val\", but got \"{type}\" instead.") 
        if isinstance(df, pd.DataFrame):
            df = df[["queries", "labels"]].explode(["queries", "labels"], ignore_index=True).drop_duplicates(["queries", "labels"])
            path = f"{self.path}\\{name}_{type}.csv"
            df.to_csv(path, index=False)
            length = len(df)
        else:
            path = None
            length = 0
        return path, length
    
    def _get_unique_values_of_lists_in_column(self, df, column):
        unique_values = set()
        for unique_list in df[column].unique():
            unique_values.update(unique_list)
        return unique_values
    
    def _init_dataset_dataframe(self):
        pd.DataFrame(columns=["name", "train_path", "train_length", "val_path", "val_length", "test_path", "test_length", "domain", "tasks", "run_tests"]).to_csv(self.dataset_path, index=False)

