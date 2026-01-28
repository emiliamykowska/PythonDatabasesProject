import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from project_utils import (
    encode_data,
    scale_data,
    save_data_to_sql,
    connect_to_sql,
    load_data_from_sql,
)

cnx = connect_to_sql("project_adm", "sTroke123", "stroke_project")
cur = cnx.cursor()

seed = 17

data = load_data_from_sql("UNPROCESSED_DATA", cnx)
data["Ever_married"] = data["Ever_married"].map({"Yes": 1, "No": 0})
data["Residence_type"] = data["Residence_type"].map({"Urban": 1, "Rural": 0})
data["Gender"] = data["Gender"].map({"Female": 1, "Male": 0})

# print(data.loc[data["Stroke"] == 1, "Bmi"].isna().sum())

y = data["Stroke"]
X = data.drop(columns="Stroke")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)

# print(y_test.values.sum())

bmi_train_median = X_train["Bmi"].median()

X_train["Bmi"] = X_train["Bmi"].fillna(bmi_train_median)
X_test["Bmi"] = X_test["Bmi"].fillna(bmi_train_median)

categorical_columns = data.select_dtypes(include=["object"]).columns.to_list()
numerical_columns = ["Age", "Avg_glucose_level", "Bmi"]

encoder = OneHotEncoder(sparse_output=False)
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

save_data_to_sql(
    cur, pd.concat([train_final, y_train], axis=1), "PREPROCESSED_DATA_TRAIN"
)
save_data_to_sql(cur, pd.concat([test_final, y_test], axis=1), "PREPROCESSED_DATA_TEST")

cnx.commit()
