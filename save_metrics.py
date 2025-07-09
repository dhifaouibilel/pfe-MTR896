import os
import pandas as pd
from typing import List
from db.db_manager import MongoManager
import utils.helpers as hpr

class MetricsMerger:
    def __init__(self, folder_path: str="Files/Metrics"):
        self.folder_path = folder_path
        self.dfs: List[pd.DataFrame] = []
        self.metrics_collection = 'metrics'
        self.deps_collection = 'dependencies2'
        self.mongo_manager = MongoManager()

    def load_files(self, extension: str = ".csv"):
        """Load all files with a given extension as DataFrames."""
        files = [f for f in os.listdir(self.folder_path) if f.endswith(extension)]
        for file in files:
            file_path = os.path.join(self.folder_path, file)
            df = pd.read_csv(file_path)
            self.dfs.append(df)
        print(f"✅ Loaded {len(self.dfs)} files.")

    def merge_on_number(self, how: str = "outer") -> pd.DataFrame:
        """Merge all DataFrames on the 'number' column."""
        if not self.dfs:
            raise ValueError("No dataframes loaded.")
        merged_df = self.dfs[0]
        for df in self.dfs[1:]:
            merged_df = pd.merge(merged_df, df, on="number", how=how)
        return merged_df


    def run(self):
        df_deps = self.mongo_manager.read_all(self.deps_collection)
        dependent_changes = hpr.flatten_list(df_deps[["Source", "Target"]].values)

        self.load_files(extension=".csv")  # change if your files are .tsv, .json, etc.
        merged_df = self.merge_on_number(how="left")
        merged_df['is_dependent'] = merged_df['number'].map(lambda nbr: 1 if nbr in dependent_changes else 0)
        print('Target assigned successfully')
        # Display or save
        self.mongo_manager.save_to_db(self.metrics_collection, merged_df)
        print(merged_df.head())
        # Optionally save to file:
        # merged_df.to_csv("merged_metrics.csv", index=False)


if __name__ == "__main__":
    merger = MetricsMerger()
    merger.run()
