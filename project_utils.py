import mysql.connector
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
)
import json
import tqdm


def connect_with_database(user, password, database):
    """
    Connect to a SQL database using the provided username, password, and database name.

    Args:
        user (str): Database username.
        password (str): Database password.
        database (str): The database to connect to.

    Returns:
        mysql.connector.connection.MySQLConnection: A connection object used to run queries.

    Raises:
        mysql.connector.Error: If the connection fails (wrong credentials, wrong database, or server not reachable).
    """
    cnx = mysql.connector.connect(
        host="127.0.0.1", port=3306, user=user, password=password, database=database
    )

    return cnx


def save_data_to_database(cur, data, table_name):
    type_map = {"int64": "INTEGER", "float64": "FLOAT", "object": "VARCHAR(50)"}

    columns_types = []

    for column, dtype in data.dtypes.items():
        sql_type = type_map[dtype.name] + " NOT NULL"
        column_name = column

        if column_name == "Id":
            sql_type = sql_type.replace("NOT NULL", "PRIMARY KEY")
        elif column_name == "Bmi":
            sql_type = sql_type.removesuffix(" NOT NULL")

        columns_types.append(f"`{column_name}` {sql_type}")

    cur.execute(
        f"""CREATE TABLE IF NOT EXISTS {table_name} ({", ".join(columns_types)})"""
    )

    column_names = ", ".join([f"`{column}`" for column in data.columns])
    placeholders = ", ".join(["%s"] * len(data.columns))

    insert = f"""INSERT IGNORE INTO {table_name}
        ({column_names})
        VALUES ({placeholders})"""

    rows_to_execute = [
        tuple(None if pd.isna(x) else x for x in row) for row in data.to_numpy()
    ]

    cur.executemany(insert, rows_to_execute)


def load_data_from_database(table_name, cnx):
    return pd.read_sql_query(f"SELECT * FROM {table_name}", cnx)


def encode_data(encoder, X, is_train, categorical_columns):
    if is_train:
        X_ohe = encoder.fit_transform(X[categorical_columns])
    else:
        X_ohe = encoder.transform(X[categorical_columns])

    data_ohe = pd.DataFrame(
        X_ohe,
        columns=encoder.get_feature_names_out(categorical_columns),
        index=X.index,
    )

    data_encoded = pd.concat([X, data_ohe], axis=1)

    return data_encoded.drop(columns=categorical_columns)


def scale_data(scaler, X, is_train, numerical_columns):
    if is_train:
        X_scaled = scaler.fit_transform(X[numerical_columns])
    else:
        X_scaled = scaler.transform(X[numerical_columns])

    data_scaled = pd.DataFrame(X_scaled, columns=numerical_columns, index=X.index)
    data_remaining = X.drop(columns=numerical_columns)

    return pd.concat([data_remaining, data_scaled], axis=1)


def save_model_results(
    cur, model_name, params, metrics, dropped_cols, train_size, threshold
):
    # params: dict of hyperparameters
    # metrics: dict (recall, pr_auc, tn, fp, fn, tp)
    cur.execute("""CREATE TABLE IF NOT EXISTS MODEL_RESULT (
                Id INT AUTO_INCREMENT PRIMARY KEY,
                Model_name VARCHAR(50),
                Params TEXT,
                Recall FLOAT,
                Pr_auc FLOAT,
                Pr_auc_baseline FLOAT,
                Accuracy FLOAT,
                True_neg INT,
                False_pos INT,
                False_neg INT,
                True_pos INT,
                Dropped_columns TEXT,
                Train_size INT,
                Threshold FLOAT
                )""")

    values = (
        model_name,
        params,
        metrics["recall"],
        metrics["pr_auc"],
        metrics["pr_auc_baseline"],
        metrics["accuracy"],
        metrics["tn"],
        metrics["fp"],
        metrics["fn"],
        metrics["tp"],
        ",".join(dropped_cols),
        train_size,
        threshold,
    )

    query = """INSERT INTO MODEL_RESULT
            (Model_name, Params, Recall, Pr_auc, Pr_auc_baseline, Accuracy, True_neg, False_pos, 
            False_neg, True_pos, Dropped_columns, Train_size, Threshold) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

    cur.execute(query, values)


def drop_no_stroke_patients(trainset, number_of_patients, seed):
    """
    Return trainset which has all stroke patients, but only number_of_patients of no stroke patients"""
    if number_of_patients is None:
        return trainset

    no_stroke_df = trainset[trainset["Stroke"] == 0]
    stroke_df = trainset[trainset["Stroke"] == 1]

    no_stroke_sampled = no_stroke_df.sample(n=number_of_patients, random_state=seed)

    return pd.concat([no_stroke_sampled, stroke_df], axis=0)


def run_random_forest(cur, trainset_whole, testset, config, thresholds, seed):
    trainset = drop_no_stroke_patients(trainset_whole, config["no_stroke_counts"], seed)
    y_train = trainset["Stroke"]
    X_train = trainset.drop(
        ["Id", "Stroke"] + config["dropped_cols"],
        axis=1,
    )
    y_test = testset["Stroke"]
    X_test = testset.drop(
        ["Id", "Stroke"] + config["dropped_cols"],
        axis=1,
    )

    pr_auc_baseline = y_test.sum() / len(y_test)

    param_grid = {
        "n_estimators": [50, 100, 200, 500],  # number of trees
        "max_depth": [5, 10, 20, None],  # maximum number of levels in tree
        "min_samples_leaf": [1, 2, 4],  # samples required at each leaf
        "min_samples_split": [2, 4, 6],  # samples required to split the node
        "max_features": [
            "sqrt",
            "log2",
        ],  # number of features to consider at every split
        "random_state": [seed],
        "class_weight": ["balanced"],
    }

    grid = list(ParameterGrid(param_grid))

    for params in tqdm(grid, desc="-> Params", leave=False):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        stroke_probabilities = model.predict_proba(X_test)[:, 1]

        for threshold in thresholds:
            y_pred = (stroke_probabilities >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel().tolist()

            metrics = {
                "recall": recall_score(
                    y_test, y_pred
                ),  # how many with stroke were caught
                "pr_auc": average_precision_score(
                    y_test, stroke_probabilities
                ),  # threshold independent, tells how good the model is at finding strokes while avoiding false alarms
                "pr_auc_baseline": pr_auc_baseline,
                "accuracy": accuracy_score(y_test, y_pred),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }

            save_model_results(
                cur,
                "RandomForest",
                params=json.dumps(params),
                metrics=metrics,
                dropped_cols=config["dropped_cols"],
                train_size=len(trainset),
                threshold=threshold,
            )
