'''Preparing the catechol single solvent dataset for training and testing. This includes loading the raw data,
preprocessing it, featurizing the solvent column using spange descriptors, and splitting into train/test sets. 
I would rather vibe code Pandas commands than learn how to use excel.'''

import pandas as pd 
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # CNODE/
DATA_PATH = PROJECT_ROOT / "datasets" / "catechol_single_solvent" # CNODE/datasets/catechol_single_solvent

INPUT_LABELS_SINGLE_SOLVENT = [
    "EXP NUM",
    "Residence Time",
    "Temperature",
    "SOLVENT NAME"
]
TARGET_LABELS = [
    "EXP NUM",
    "Residence Time",
    "Temperature",
    "SM",
    "Product 2",
    "Product 3",
    "SOLVENT NAME"
]


#-------MISC PREPROCESSING FUNCS-------
def group_by_solvent(data_df: pd.DataFrame) -> pd.DataFrame:
    '''Rearrange rows in data_df so that all rows with the same "SOLVENT NAME" are grouped together.
    The order of first appearance of each solvent is preserved.'''
    first_occurrence_order = data_df["SOLVENT NAME"].unique()
    solvent_order = pd.CategoricalDtype(categories=first_occurrence_order, ordered=True)
    return data_df.sort_values(
        "SOLVENT NAME",
        key=lambda col: col.astype(solvent_order),
        kind="stable"
    ).reset_index(drop=True)

def sort_by_residence_time(data_df: pd.DataFrame) -> pd.DataFrame:
    '''Within each ("SOLVENT NAME", "Temperature") group, sort rows by ascending "Residence Time".
    The relative order of solvent groups is preserved, and sorting resets when "Temperature" changes.'''
    first_occurrence_order = data_df["SOLVENT NAME"].unique()
    solvent_order = pd.CategoricalDtype(categories=first_occurrence_order, ordered=True)
    return data_df.sort_values(
        ["SOLVENT NAME", "Temperature", "Residence Time"],
        key=lambda col: col.astype(solvent_order) if col.name == "SOLVENT NAME" else col,
        kind="stable"
    ).reset_index(drop=True)

def reassign_exp_num(data_df: pd.DataFrame) -> pd.DataFrame:
    '''Reassign "EXP NUM" so that it starts at 0 and increments by 1 each time
    either "SOLVENT NAME" or "Temperature" changes between consecutive rows.'''
    group_keys = data_df[["SOLVENT NAME", "Temperature"]]
    new_exp_num = (group_keys != group_keys.shift()).any(axis=1).cumsum() - 1
    data_df = data_df.copy()
    data_df["EXP NUM"] = new_exp_num.values
    return data_df

def remove_float_temperatures(data_df: pd.DataFrame) -> pd.DataFrame:
    '''Remove rows from data_df where the "Temperature" value is a non-integer float (e.g. 22.5).'''
    return data_df[data_df["Temperature"] == data_df["Temperature"].astype(int)]

def remove_solvent_mixtures(data_df: pd.DataFrame) -> pd.DataFrame:
    '''Remove rows from data_df where the "SOLVENT Ratio" column indicates a mixture of solvents.'''
    return data_df[data_df["SOLVENT Ratio"] == "[1.0]"]

def remove_duplicate_residence_times(data_df: pd.DataFrame) -> pd.DataFrame:
    '''Remove duplicate "Residence Time" entries within each ("SOLVENT NAME", "Temperature") group,
    keeping only the first occurrence.'''
    return data_df.drop_duplicates(subset=["SOLVENT NAME", "Temperature", "Residence Time"], keep="first")


#----------DATA LOADING FUNCS-------
def load_single_solvent_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Load the train input data and label dataframes for the single solvent experiments.'''
    path = DATA_PATH / "catechol_single_solvent_yields.csv"
    assert path.exists(), f"Experiment data does not exist at {path.absolute()}"
    experiments = remove_solvent_mixtures(pd.read_csv(path))
    experiments = remove_float_temperatures(experiments)
    experiments = remove_duplicate_residence_times(experiments)
    experiments = group_by_solvent(experiments)
    experiments = sort_by_residence_time(experiments)
    experiments = reassign_exp_num(experiments)
    input_cols = [
        column for column in experiments.columns if column in INPUT_LABELS_SINGLE_SOLVENT
    ]

    return experiments[input_cols], experiments[TARGET_LABELS]

def load_spange_featurization_data() -> pd.DataFrame:
    '''Load the spange featurization lookup table for solvents.'''
    path = DATA_PATH / "spange_descriptors_lookup.csv"
    assert path.exists(), f"Spange featurization data does not exist at {path.absolute()}"

    return pd.read_csv(path, index_col=0)


#-------SOLVENT FEATURIZATION----------
def spange_featurize_input_df(data_df: pd.DataFrame) -> pd.DataFrame:
    '''Replace the data_df['SOLVENT NAME'] column entries with their spange-featurized representation.'''
    solvents = data_df["SOLVENT NAME"]
    feat_lookup = load_spange_featurization_data().rename_axis(
        solvents.name, axis="index"
    ) 
    feat = feat_lookup.loc[solvents].reset_index().set_index(solvents.index)
    data_df = pd.concat(
        [
            data_df.drop(columns=["SOLVENT NAME"]), 
            feat.drop(columns=[solvents.name])
        ],
        axis="columns"
    )

    return data_df 


#-------TRAIN/TEST SPLITTING-------
def train_test_split(
    data_df: pd.DataFrame, 
    labels_df: pd.DataFrame,
    train_percentage: float, 
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into a train/test pair based on experiment number, 
    so that the test set contains experiments (trajectories) that are not in the training set.
    """
    num_experiments = max(data_df["EXP NUM"]) + 1
    num_train_experiments = int(num_experiments * train_percentage)
    data_train_df = data_df[data_df["EXP NUM"] <= num_train_experiments]
    data_test_df = data_df[data_df["EXP NUM"] > num_train_experiments]
    labels_train_df = labels_df.loc[data_train_df.index]
    labels_test_df = labels_df.loc[data_test_df.index]

    return data_train_df, labels_train_df, data_test_df, labels_test_df

'''data_df, label_df = load_single_solvent_data()
data_df = spange_featurize_input_df(data_df)
train_data_df, train_labels_df, test_data_df, test_labels_df = train_test_split(data_df, label_df, train_percentage=0.7)
train_data_df.to_csv(DATA_PATH / "train_data_catechol_single_solvent.csv", index=False)
train_labels_df.to_csv(DATA_PATH / "train_labels_catechol_single_solvent.csv", index=False)
test_data_df.to_csv(DATA_PATH / "test_data_catechol_single_solvent.csv", index=False)
test_labels_df.to_csv(DATA_PATH / "test_labels_catechol_single_solvent.csv", index=False)'''
