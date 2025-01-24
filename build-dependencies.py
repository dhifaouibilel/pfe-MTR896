import pandas as pd
import re
import os
import utils.helpers as hpr

DIR = hpr.DIR


def extract_attr(x, attr):
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
            result.append(re.search("\d+$", number_pattern[0][0:])[0])
    return result if len(result) != 0 else None

def build_depends_chain(row):
    '''Flatten the depends_on columns for each change
    '''
    obj = {}
    depends_on = row["depends_on"]
    obj["Target"] = row["number"]
    obj["Target_repo"] = row["project"]
    row_src = None
    if depends_on.isnumeric():
        row_src = df[df["number"] == int(depends_on)]
    else:
        row_src = df[df["change_id"] == depends_on]

    if len(row_src) != 0:
        source_numbers = hpr.flatten_list(row_src[["number"]].to_numpy())

        source_numbers = list(dict.fromkeys(source_numbers))
        obj["Source"] = source_numbers
        obj["Source_repo"] = row_src["project"].head(1).tolist()[0]

    return obj

def build_needed_chain(row):
    '''Flatten the needed_by columns for each change
    '''
    obj = {}
    needed_by = row["needed_by"]
    obj["Source"] = row["number"]
    obj["Source_repo"] = row["project"]
    row_src = None
    if needed_by.isnumeric():
        row_target = df[df["number"] == int(needed_by)]
    else:
        row_target = df[df["change_id"] == needed_by]

    if len(row_target) != 0:
        target_numbers = hpr.flatten_list(row_target[["number"]].to_numpy())

        target_numbers = list(dict.fromkeys(target_numbers))
        obj["Target"] = target_numbers
        obj["Target_repo"] = row_target["project"].head(1).tolist()[0]

    return obj


def generate_os_evolution_data(df):
    '''Generate Openstack evolution files containing following the 
    columns ["Source", "Target", "Source_repo", "Target_repo"]
    '''
    evolution_columns = ["Source", "Target", "Source_repo", "Target_repo"]

    df["depends_on"] = df["commit_message"].apply(extract_attr, args=("Depends-On",))
    df["needed_by"] = df["commit_message"].apply(extract_attr, args=("Needed-By",))

    df = df.explode(column="depends_on")
    df = df.explode(column="needed_by")

    # process depends_on
    subset_depends_columns = ["change_id", "project", "depends_on", "number"]

    df_depends_on = df.loc[df["depends_on"].notnull(), subset_depends_columns].copy()

    df_depends_on = df_depends_on.apply(build_depends_chain, axis=1)

    df_depends_on = pd.json_normalize(data=df_depends_on, errors="ignore")

    df_depends_on.dropna(inplace=True)

    df_depends_on = df_depends_on.explode(column="Source")

    df_depends_on = df_depends_on.loc[:, evolution_columns]

    df_depends_on.drop_duplicates(subset=["Source", "Target"],inplace=True)
    
    df_depends_on["Source"] = df_depends_on[["Source"]].astype(int)
    
    # process needed_by
    subset_needed_columns = ["change_id", "project", "needed_by", "number"]

    df_needed_by = df.loc[df["needed_by"].notnull(), subset_needed_columns].copy()

    df_needed_by = df_needed_by.apply(build_needed_chain, axis=1)

    df_needed_by = pd.json_normalize(data=df_needed_by, errors="ignore")

    df_needed_by.dropna(inplace=True)

    df_needed_by = df_needed_by.explode(column="Target").reset_index(drop=True)

    df_needed_by = df_needed_by.loc[:, evolution_columns]

    df_needed_by["Target"] = df_needed_by[["Target"]].astype(int)

    # combine depends_on and needed_by dataframes
    df_depends_needed = pd.concat((df_depends_on, df_needed_by)).reset_index(drop=True)

    # process combined dataframe
    df_depends_needed.drop_duplicates(subset=["Source", "Target"],inplace=True)

    # df_depends_needed = df_depends_needed.loc[df_depends_needed["Source_repo"] != df_depends_needed["Target_repo"]]

    df_depends_needed = df_depends_needed.reset_index(drop=True)

    df_depends_on.to_csv("%sFiles/source_target_depends.csv" % DIR, index=False)

    df_needed_by.to_csv("%sFiles/source_target_needed.csv" % DIR, index=False)
    
    df_depends_needed.to_csv("%sFiles/source_target_evolution.csv" % DIR, index=False)


if __name__ == "__main__":

    print("Script openstack-dependencies-generation.py started...")

    start_date, start_header = hpr.generate_date("This script started at")

    files_path = "%sFiles/" % DIR
    for d in ["%sNumber" % (files_path), "%sRepo" % (files_path), "%sMetrics" % (files_path)]:
        if not os.path.exists(d):
            os.makedirs(d)

    df = hpr.combine_openstack_data()

    generate_os_evolution_data(df)

    end_date, end_header = hpr.generate_date("This script ended at")

    print(start_header)

    print(end_header)

    hpr.diff_dates(start_date, end_date)

    print("Script openstack-dependencies-generation.py ended\n")
