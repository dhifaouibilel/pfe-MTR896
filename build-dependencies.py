import pandas as pd
import re
import os
from datetime import datetime
import utils.helpers as hpr
import ast


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
    
    def time_diff(self, start, end):
        if start > end:
            start, end = end, start
        current_date =  datetime.strptime(end, "%Y-%m-%d %H:%M:%S") 
        previous_date = datetime.strptime(start, "%Y-%m-%d %H:%M:%S") 
        diff = current_date - previous_date
        diff = float("{:.2f}".format(diff.total_seconds() / 3600))
        return diff


    def extract_attr(self, x, attr):
        '''Extracts the passed-on parameter values out of the commit message 
        '''
        rs = re.findall("%s:\s[a-zA-Z0-9/\.\:\+\-\#]{6,}" % (attr), x)
        result = []
        for row in rs:
            row = row[len(attr) + 2:]
            change_id_pattern = re.search(r"[a-zA-Z0-9]{41}", row)
            if change_id_pattern:
                result.append(change_id_pattern[0])
                continue
            number_pattern = re.search("#?https?[\:][/]{2}review[\.](opendev|openstack)[\.]org([a-z0-9A-Z\-\+/\.#]*)\d+", row)
            if number_pattern:
                result.append(int(re.search("\d+$", number_pattern[0][0:])[0]))
        return result if len(result) != 0 else None


    def retrieve_revision_date(self, row, attr, return_revision_date=True):
        number = None
        second_number = None

        if attr == "Depends-On":
            number = row["Target"]
            second_number = row["Source"]
            change_id = row["Source_change_id"]
        else:
            number = row["Source"]
            second_number = row["Target"]
            change_id = row["Target_change_id"]

        df_row = self.df.loc[self.df["number"] == number]
        revisions = ast.literal_eval(df_row["revisions"].values[0])
        revisions = sorted(revisions, key=lambda x: x["created"])
        if  len(revisions) == 1:
            if return_revision_date:
                return revisions[0]["created"][:-11]
            else:
                return 1

        first_revision = revisions[0]
        first_message = first_revision["message"]

        results = self.extract_attr(first_message, attr)

        if results and ((change_id in results) or (second_number in results)):
            if return_revision_date:
                return first_revision["created"][:-11]
            else:
                return 1

        for i in range(1,len(revisions)):
            current_message = revisions[i]["message"]
            created = revisions[i]["created"]
            results = self.extract_attr(current_message, attr)
            
            if results and ((change_id in results) or (second_number in results)):

                if return_revision_date:
                    return created[:-11]
                else:
                    return i + 1

    def is_same_developer(self, row):
        return "Same" if row["Source_dev"] == row["Target_dev"] else "Different"

    def identify_dependency(self, row):
        source_date = row["Source_date"] 
        target_date = row["Target_date"]
        link_date = datetime.strptime(row["link_date"], "%Y-%m-%d %H:%M:%S")
        delta2 = (source_date - link_date).total_seconds() / (60 * 60)

        return abs(delta2)

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

        # Get Source fields using merge
        df_depends_on = pd.merge(
            df_depends_on,
            self.df[["number", "status", "change_id", "is_owner_bot", "owner_account_id", "created"]].rename(
                columns={
                    "number": "Source",
                    "status": "Source_status",
                    "change_id": "Source_change_id",
                    "is_owner_bot": "is_source_bot",
                    "owner_account_id": "Source_author",
                    "created": "Source_date"
                }
            ),
            on="Source",
            how="left"
        )

        # Get Target fields using merge
        df_needed_by = pd.merge(
            df_needed_by,
            df[["number", "status", "change_id", "revisions", "is_owner_bot", "owner_account_id", "created"]].rename(
                columns={
                    "number": "Target",
                    "status": "Target_status",
                    "change_id": "Target_change_id",
                    "is_owner_bot": "is_target_bot",
                    "owner_account_id": "Target_dev",
                    "created": "Target_date"
                }
            ),
            on="Target",
            how="left"
        )

        df_depends_on["is_cross"] = df_depends_on.apply(lambda row: "Cross" if row["Source_repo"] != row["Target_repo"] else "Same", axis=1)

        df_depends_on["link_date"] = df_depends_on.apply(self.retrieve_revision_date, args=("Depends-On",), axis=1)
        df_depends_on["worked_revisions"] = df_depends_on.apply(self.retrieve_revision_date, args=("Depends-On",False,), axis=1)
        df_depends_on["same_dev"] = df_depends_on.apply(self.is_same_developer, axis=1)
        df_depends_on["when_identified"] = df_depends_on[["Source_date", "Target_date", "link_date"]].apply(self.identify_dependency, axis=1)
        df_depends_on["deps_label"] = "Depends-On"
        
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

        df_needed_by["is_cross"] = df_needed_by.apply(lambda row: "Cross" if row["Source_repo"] != row["Target_repo"] else "Same", axis=1)

        df_needed_by["link_date"] = df_needed_by.apply(self, self.retrieve_revision_date, args=("Depends-On",), axis=1)
        df_needed_by["worked_revisions"] = df_needed_by.apply(self, self.retrieve_revision_date, args=("Depends-On",False,), axis=1)
        df_needed_by["same_dev"] = df_needed_by.apply(self, self.is_same_developer, axis=1)
        df_needed_by["when_identified"] = df_needed_by[["Source_date", "Target_date", "link_date"]].apply(self.identify_dependency, axis=1)
        df_needed_by["deps_label"] = "Needed-By"

        # Combine depends_on and needed_by dataframes
        print("Combining relationship data...")
        df_depends_needed = pd.concat((df_depends_on, df_needed_by)).reset_index(drop=True)
        df_depends_needed.drop_duplicates(subset=["Source", "Target"], inplace=True)

        # Add additional metadata through merges
        df_depends_needed = pd.merge(
            left=df_depends_needed,
            right=self.df[["number", "status", "owner_account_id", "created", "change_id", "is_owner_bot"]],
            left_on=['Source'],
            right_on=['number'],
            how='left',
            suffixes=('_target', '_source')
        )
        
        df_depends_needed = pd.merge(
            left=df_depends_needed,
            right=self.df[["number", "status", "owner_account_id", "created", "change_id", "is_owner_bot"]],
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
            "created_source": "Source_date",
            "created_target": "Target_date",
            "change_id_target": "Target_change_id",
            "change_id_source": "Source_change_id",
            "is_owner_bot_target": "Target_is_owner_bot",
            "is_owner_bot_source": "Source_is_owner_bot",
        }, inplace=True)

        df_depends_needed = df_depends_needed.reset_index(drop=True)

        # Save results to CSV files
        print("Saving results to CSV files...")
        df_depends_on.to_csv(f"{self.DIR}/Files/source_target_depends2.csv", index=False)
        df_needed_by.to_csv(f"{self.DIR}/Files/source_target_needed2.csv", index=False)
        df_depends_needed.to_csv(f"{self.DIR}/Files/source_target_evolution2.csv", index=False)

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
        # self.prepare_directories()
        
        # Load OpenStack data
        print("Loading OpenStack data...")
        self.df = hpr.combine_openstack_data(changes_path="/Changes3/")
        
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