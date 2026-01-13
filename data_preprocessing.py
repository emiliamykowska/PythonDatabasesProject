import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from project_utils import (
    encode_data,
    scale_data,
    save_data_to_database,
    connect_with_database,
    load_data_from_database,
)

cnx = connect_with_database("project_adm", "sTroke123", "stroke_project")
cur = cnx.cursor()

data = load_data_from_database("UNPROCESSED_DATA", cnx)
data["Ever_married"] = data["Ever_married"].map({"Yes": 1, "No": 0})
data["Residence_type"] = data["Residence_type"].map({"Urban": 1, "Rural": 0})


print(data.loc[data["Stroke"] == 1, "Bmi"].isna().sum())

y = data["Stroke"]
X = data.drop(columns="Stroke")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=17, stratify=y
)

print(y_test.values.sum())

bmi_train_median = X_train["Bmi"].median()

X_train["Bmi"] = X_train["Bmi"].fillna(bmi_train_median)
X_test["Bmi"] = X_test["Bmi"].fillna(bmi_train_median)

categorical_columns = data.select_dtypes(include=["object"]).columns.to_list()
numerical_columns = ["Age", "Avg_glucose_level", "Bmi"]

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
scaler = StandardScaler()

train_encoded = encode_data(
    encoder, X_train, is_train=True, categorical_columns=categorical_columns
)
test_encoded = encode_data(
    encoder, X_test, is_train=False, categorical_columns=categorical_columns
)

train_final = scale_data(
    scaler, train_encoded, is_train=True, numerical_columns=numerical_columns
)
test_final = scale_data(
    scaler, test_encoded, is_train=False, numerical_columns=numerical_columns
)

save_data_to_database(
    cur, pd.concat([train_final, y_train], axis=1), "PREPROCESSED_DATA_TRAIN"
)
save_data_to_database(
    cur, pd.concat([test_final, y_test], axis=1), "PREPROCESSED_DATA_TEST"
)

cnx.commit()
