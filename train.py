from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score, auc
from sklearn.model_selection import train_test_split
from azureml.core.run import Run, Dataset
from azureml.core.workspace import Workspace

run = Run.get_context()

def clean_data(data):
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    x_df.drop("CLIENTNUM", inplace=True, axis=1)
    gender = pd.get_dummies(x_df.Gender, prefix="Gender")
    x_df.drop("Gender", inplace=True, axis=1)
    education_level = pd.get_dummies(x_df.education_level, prefix="Education_Level")
    x_df.drop("Education_Level", inplace=True, axis=1)
    x_df = x_df.join(education_level)
    marital_status = pd.get_dummies(x_df.marital_status, prefix="Marital_Status")
    x_df.drop("Marital_Status", inplace=True, axis=1)
    x_df = x_df.join(marital_status)
    income_category = pd.get_dummies(x_df.income_category, prefix="Income_Category")
    x_df.drop("Income_Category", inplace=True, axis=1)
    x_df = x_df.join(income_category)
    card_category = pd.get_dummies(x_df.card_category, prefix="Card_Category")
    x_df.drop("Card_Category", inplace=True, axis=1)
    x_df = x_df.join(card_category)
    y_df = x_df.pop("Attrition_Flag").apply(lambda s: 1 if s == "Attrited Customer" else 0)

    return x_df, y_df

def retrieve_cleaned_data():
    ws = run.experiment.workspace
    ds_name = 'bankchurners'
    ds = Dataset.get_by_name(workspace = ws, name = ds_name)
    x, y = clean_data(ds)
    return train_test_split(x, y, test_size = 0.3, random_state = 0)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100, help="Number of trees in the forest")
    parser.add_argument('--max_depth', type=int, default=None, help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
    parser.add_argument('--min_samples_split', type=int, default=2, help="The minimum number of samples required to split an internal node.")
    parser.add_argument('--min_samples_leaf', type=int, default=1, help="The minimum number of samples required to be at a leaf node.")

    args = parser.parse_args()

    if args.max_depth == 0:
        max_depth = None
    else:
        max_depth = args.max_depth

    run.log("Num Estimators:", np.float(args.n_estimators))
    run.log("Max Depth:", max_depth)
    run.log("Min Samples Split:", np.int(args.min_samples_split))
    run.log("Min Samples Leaf:", np.int(args.min_samples_leaf))

    x_train, x_test, y_train, y_test = retrieve_cleaned_data()

    model = RandomForestClassifier(n_estimators=args.n_estimators,max_depth=max_depth,min_samples_split=args.min_samples_split,min_samples_leaf=args.min_samples_leaf).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    run.log("Accuracy", np.float(accuracy))
    run.log("F1 Score", np.float(f1))
    run.log("Precision", np.float(precision))
    run.log("Recall", np.float(recall))

if __name__ == '__main__':
    main()
