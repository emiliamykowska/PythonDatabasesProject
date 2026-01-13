import pandas as pd
import itertools
import mysql.connector
from project_utils import (
    connect_with_database,
    save_data_to_database,
    load_data_from_database,
    run_random_forest,
)
import tqdm

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
    "no_stroke_counts": [250, 500, 1000, 2000, None],
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
            "Gender_Female",
            "Gender_Male",
            "Gender_Other",
            "Residence_type",
        ],
    ],
    "thresholds": [0.2, 0.3, 0.5],
}

combinations = itertools.product(
    experiment_cfg["no_stroke_counts"], experiment_cfg["columns_to_drop"]
)

for no_stroke_no, dropped in tqdm(combinations, desc="Experiments"):
    current_config = {"no_stroke_counts": no_stroke_no, "dropped_cols": dropped}
    run_random_forest(
        cur, trainset_whole, testset, current_config, experiment_cfg["thresholds"], seed
    )
    cnx.commit()

# print(
#     pd.DataFrame(
#         rfc.feature_importances_, index=X_train.columns, columns=["importance"]
#     ).sort_values(by="importance", ascending=False)
# )
# print(recall, pr_auc)
# print(tn, fp, fn, tp)
