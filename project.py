import pandas as pd
import itertools
import mysql.connector
from project_utils import (
    connect_with_database,
    save_data_to_database,
    load_data_from_database,
    run_model,
)
from tqdm import tqdm

cnx = connect_with_database("project_adm", "sTroke123", "stroke_project")
cur = cnx.cursor()
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

save_data_to_database(cur, data, "UNPROCESSED_DATA")
cnx.commit()

trainset_whole = load_data_from_database("PREPROCESSED_DATA_TRAIN", cnx)
testset = load_data_from_database("PREPROCESSED_DATA_TEST", cnx)

# DATA CONFIGURATION
seed = 17
experiment_cfg = {
    "no_stroke_counts": [200, 1000, 2000, None],
    "columns_to_drop": [
        [],  # No columns dropped
        ["Ever_married", "Work_type_children"],
        ["Ever_married", "Work_type_children", "Bmi"],
        ["Ever_married", "Work_type_children", "Bmi", "Residence_type"],
        [
            "Ever_married",
            "Work_type_children",
            "Work_type_Never_worked",
            "Work_type_Private",
            "Work_type_Govt_job",
            "Work_type_Self-employed",
        ],
        [
            "Ever_married",
            "Work_type_children",
            "Work_type_Never_worked",
            "Work_type_Private",
            "Work_type_Govt_job",
            "Work_type_Self-employed",
            "Bmi",
            "Residence_type",
        ],
        [
            "Ever_married",
            "Work_type_children",
            "Work_type_Never_worked",
            "Work_type_Private",
            "Work_type_Govt_job",
            "Work_type_Self-employed",
            "Bmi",
            "Smoking_status_Unknown",
            "Smoking_status_formerly smoked",
            "Smoking_status_never smoked",
            "Smoking_status_smokes",
            "Gender",
            "Residence_type",
        ],
    ],
    "thresholds": [0.3, 0.4, 0.5],
}

param_grid_rf = {
    "n_estimators": [50, 100, 200, 500],  # number of trees
    "max_depth": [5, 10, None],  # maximum number of levels in tree
    "min_samples_leaf": [1, 2],  # samples required at each leaf
    "min_samples_split": [2, 4],  # samples required to split the node
    "max_features": [
        "sqrt",
        "log2",
    ],  # number of features to consider at every split
    "random_state": [seed],
    "class_weight": ["balanced"],
}

param_grid_xgb = {
    "n_estimators": [50, 100, 200],  # number of trees
    "learning_rate": [0.2, 0.3, 0.4],  # learning rate
    "max_depth": [4, 6],
    "min_child_weight": [
        1,
        3,
        5,
    ],  # child cannot split if the hessian is too small (if the leaf corresponds to rare case)
    "scale_pos_weight": [
        1,
        5,
        sum(trainset_whole["Stroke"] == 0) / sum(trainset_whole["Stroke"] == 1),
    ],
}

param_grid_dmc = {
    "epochs": [15, 30, 50, 100],
    "batch_size": [32, 64],
    "in_features": len(trainset_whole.columns),
    "hidden_layer": [64, 128],
    "hidden_layer2": [128, 256],
    "dropout": [0, 0.2],
}

combinations = itertools.product(
    experiment_cfg["no_stroke_counts"], experiment_cfg["columns_to_drop"]
)


for no_stroke_no, dropped in tqdm(combinations, desc="ExperimentsRT"):
    current_config = {"no_stroke_counts": no_stroke_no, "dropped_cols": dropped}
    run_model(
        cur,
        "RandomForestClassifier",
        param_grid_rf,
        trainset_whole,
        testset,
        current_config,
        experiment_cfg["thresholds"],
        seed,
    )
    cnx.commit()

combinations = itertools.product(
    experiment_cfg["no_stroke_counts"], experiment_cfg["columns_to_drop"]
)

for no_stroke_no, dropped in tqdm(combinations, desc="ExperimentsXGB"):
    current_config = {"no_stroke_counts": no_stroke_no, "dropped_cols": dropped}
    run_model(
        cur,
        "XGBClassifier",
        param_grid_xgb,
        trainset_whole,
        testset,
        current_config,
        experiment_cfg["thresholds"],
        seed,
    )
    cnx.commit()

combinations = itertools.product(
    experiment_cfg["no_stroke_counts"], experiment_cfg["columns_to_drop"]
)

for no_stroke_no, dropped in tqdm(combinations, desc="ExperimentsDMC"):
    current_config = {"no_stroke_counts": no_stroke_no, "dropped_cols": dropped}
    run_model(
        cur,
        "DeepModelClassifier",
        param_grid_dmc,
        trainset_whole,
        testset,
        current_config,
        experiment_cfg["thresholds"],
        seed,
    )
