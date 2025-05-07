import pandas as pd
import re
import os
import utils.helpers as hpr


class OpenStackDependencyGenerator:
    """
    A class for generating OpenStack dependency relationships from commit messages.
    Extracts 'Depends-On' and 'Needed-By' relationships to build evolution data.
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize the dependency generator with a base directory.
        
        Args:
            base_dir: The base directory for data processing. If None, uses the default from helpers.
        """
        self.DIR = base_dir if base_dir is not None else hpr.DIR
        self.df = None
        
    def prepare_directories(self):
        """
        Create necessary directories for storing output files.
        """
        files_path = f"{self.DIR}Files/"
        for d in [f"{files_path}Number", f"{files_path}Repo", f"{files_path}Metrics"]:
            if not os.path.exists(d):
                os.makedirs(d)
        print(f"Directories prepared for output files")
    
    def extract_attr(self, commit_message, attr):
        """
        Extract parameter values from commit messages.
        
        Args:
            commit_message: The commit message to analyze.
            attr: The attribute to extract (e.g., 'Depends-On', 'Needed-By').
            
        Returns:
            list or None: List of extracted values or None if none found.
        """
        rs = re.findall(f"{attr}:\\s[a-zA-Z0-9/\\.\\:\\+\\-\\#]{{6,}}", commit_message)
        result = []
        
        for row in rs:
            row = row[len(attr) + 2:]
            # Look for change_id pattern (41 characters)
            change_id_pattern = re.search(r"[a-zA-Z0-9]{41}", row)
            if change_id_pattern:
                result.append(change_id_pattern[0])
                continue
                
            # Look for URLs with numbers at the end
            number_pattern = re.search(r"#?https?[\:][/]{2}review[\.](opendev|openstack)[\.]org([a-z0-9A-Z\-\+/\.#]*)\d+", row)
            if number_pattern:
                result.append(re.search(r"\d+$", number_pattern[0][0:])[0])
                
        return result if len(result) != 0 else None
    
    def build_depends_chain(self, row):
        """
        Flatten the depends_on columns for each change.
        
        Args:
            row: DataFrame row with change data.
            
        Returns:
            dict: Object with Source, Target, Source_repo, Target_repo information.
        """
        obj = {}
        depends_on = row["depends_on"]
        obj["Target"] = row["number"]
        obj["Target_repo"] = row["project"]
        
        row_src = None
        if depends_on.isnumeric():
            row_src = self.df[self.df["number"] == int(depends_on)]
        else:
            row_src = self.df[self.df["change_id"] == depends_on]

        if len(row_src) != 0:
            source_numbers = hpr.flatten_list(row_src[["number"]].to_numpy())
            source_numbers = list(dict.fromkeys(source_numbers))
            obj["Source"] = source_numbers
            obj["Source_repo"] = row_src["project"].head(1).tolist()[0]

        return obj
    
    def build_needed_chain(self, row):
        """
        Flatten the needed_by columns for each change.
        
        Args:
            row: DataFrame row with change data.
            
        Returns:
            dict: Object with Source, Target, Source_repo, Target_repo information.
        """
        obj = {}
        needed_by = row["needed_by"]
        obj["Source"] = row["number"]
        obj["Source_repo"] = row["project"]
        
        if needed_by.isnumeric():
            row_target = self.df[self.df["number"] == int(needed_by)]
        else:
            row_target = self.df[self.df["change_id"] == needed_by]

        if len(row_target) != 0:
            target_numbers = hpr.flatten_list(row_target[["number"]].to_numpy())
            target_numbers = list(dict.fromkeys(target_numbers))
            obj["Target"] = target_numbers
            obj["Target_repo"] = row_target["project"].head(1).tolist()[0]

        return obj
    
    def generate_os_evolution_data(self):
        """
        Generate OpenStack evolution files containing:
        ["Source", "Target", "Source_repo", "Target_repo"]
        
        Returns:
            tuple: Three DataFrames (df_depends_on, df_needed_by, df_depends_needed)
        """
        print("Generating OpenStack evolution data...")
        evolution_columns = ["Source", "Target", "Source_repo", "Target_repo"]

        # Extract 'Depends-On' and 'Needed-By' attributes from commit messages
        self.df["depends_on"] = self.df["commit_message"].apply(self.extract_attr, args=("Depends-On",))
        self.df["needed_by"] = self.df["commit_message"].apply(self.extract_attr, args=("Needed-By",))

        # Explode the lists to create separate rows
        self.df = self.df.explode(column="depends_on")
        self.df = self.df.explode(column="needed_by")

        # Process depends_on relationships
        print("Processing 'Depends-On' relationships...")
        subset_depends_columns = ["change_id", "project", "depends_on", "number"]
        df_depends_on = self.df.loc[self.df["depends_on"].notnull(), subset_depends_columns].copy()
        df_depends_on = df_depends_on.apply(self.build_depends_chain, axis=1)
        df_depends_on = pd.json_normalize(data=df_depends_on, errors="ignore")
        df_depends_on.dropna(inplace=True)
        df_depends_on = df_depends_on.explode(column="Source")
        df_depends_on = df_depends_on.loc[:, evolution_columns]
        df_depends_on.drop_duplicates(subset=["Source", "Target"], inplace=True)
        df_depends_on["Source"] = df_depends_on[["Source"]].astype(int)
        
        # Process needed_by relationships
        print("Processing 'Needed-By' relationships...")
        subset_needed_columns = ["change_id", "project", "needed_by", "number"]
        df_needed_by = self.df.loc[self.df["needed_by"].notnull(), subset_needed_columns].copy()
        df_needed_by = df_needed_by.apply(self.build_needed_chain, axis=1)
        df_needed_by = pd.json_normalize(data=df_needed_by, errors="ignore")
        df_needed_by.dropna(inplace=True)
        df_needed_by = df_needed_by.explode(column="Target").reset_index(drop=True)
        df_needed_by = df_needed_by.loc[:, evolution_columns]
        df_needed_by["Target"] = df_needed_by[["Target"]].astype(int)

        # Combine depends_on and needed_by dataframes
        print("Combining relationship data...")
        df_depends_needed = pd.concat((df_depends_on, df_needed_by)).reset_index(drop=True)
        df_depends_needed.drop_duplicates(subset=["Source", "Target"], inplace=True)

        # Add additional metadata through merges
        df_depends_needed = pd.merge(
            left=df_depends_needed,
            right=self.df[["number", "status", "owner_account_id", "created"]],
            left_on=['Source'],
            right_on=['number'],
            how='left',
            suffixes=('_target', '_source')
        )
        
        df_depends_needed = pd.merge(
            left=df_depends_needed,
            right=self.df[["number", "status", "owner_account_id", "created"]],
            left_on=['Target'],
            right_on=['number'],
            how='left',
            suffixes=('_source', '_target')
        )
        
        # Rename columns for clarity
        df_depends_needed.rename(columns={
            "status_source": "Source_status",
            "status_target": "Target_status",
            "author_source": "Source_author",
            "status_author": "Target_status",
            "created_source": "Source_created",
            "created_target": "Target_created"}, inplace=True)

        df_depends_needed = df_depends_needed.reset_index(drop=True)

        # Save results to CSV files
        print("Saving results to CSV files...")
        df_depends_on.to_csv(f"{self.DIR}/Files/source_target_depends.csv", index=False)
        df_needed_by.to_csv(f"{self.DIR}/Files/source_target_needed.csv", index=False)
        df_depends_needed.to_csv(f"{self.DIR}/Files/source_target_evolution.csv", index=False)

        return df_depends_on, df_needed_by, df_depends_needed
    
    def generate_dependencies(self):
        """
        Main method to generate all dependency relationships.
        
        Returns:
            tuple: Three DataFrames (df_depends_on, df_needed_by, df_depends_needed)
        """
        print("OpenStack dependencies generation started...")
        
        start_date, start_header = hpr.generate_date("This generation started at")
        
        # Prepare directories
        self.prepare_directories()
        
        # Load OpenStack data
        print("Loading OpenStack data...")
        self.df = hpr.combine_openstack_data()
        
        # Generate evolution data
        results = self.generate_os_evolution_data()
        
        end_date, end_header = hpr.generate_date("This generation ended at")
        
        print(start_header)
        print(end_header)
        hpr.diff_dates(start_date, end_date)
        
        print("OpenStack dependencies generation completed\n")
        
        return results


if __name__ == "__main__":
    generator = OpenStackDependencyGenerator()
    generator.generate_dependencies()