from sklearn.ensemble import RandomForestClassifier
import argparse
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
from azureml.core.workspace import Workspace
from azureml.data.dataset_factory import TabularDatasetFactory

run = Run.get_context()

def clean_data(data):
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    x_df.drop("CLIENTNUM", inplace=True, axis=1)
    gender = pd.get_dummies(x_df.Gender, prefix="Gender")
    x_df.drop("Gender", inplace=True, axis=1)
    education_level = pd.get_dummies(x_df.Education_Level, prefix="Education_Level")
    x_df.drop("Education_Level", inplace=True, axis=1)
    x_df = x_df.join(education_level)
    marital_status = pd.get_dummies(x_df.Marital_Status, prefix="Marital_Status")
    x_df.drop("Marital_Status", inplace=True, axis=1)
    x_df = x_df.join(marital_status)
    income_category = pd.get_dummies(x_df.Income_Category, prefix="Income_Category")
    x_df.drop("Income_Category", inplace=True, axis=1)
    x_df = x_df.join(income_category)
    card_category = pd.get_dummies(x_df.Card_Category, prefix="Card_Category")
    x_df.drop("Card_Category", inplace=True, axis=1)
    x_df = x_df.join(card_category)
    y_df = x_df.pop("Attrition_Flag").apply(lambda s: 1 if s == "Attrited Customer" else 0)

    return x_df, y_df

def retrieve_cleaned_data():
    # Create TabularDataset using TabularDatasetFactory
    url = "https://raw.githubusercontent.com/riokorb/nd00333-capstone/master/BankChurners.csv"
    ds = TabularDatasetFactory.from_delimited_files(path=url)
    # clean dataset
    x, y = clean_data(ds)
    return train_test_split(x, y, test_size = 0.3, random_state = 0)

run = Run.get_context()


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=str, default='linear',
                        help='Kernel type to be used in the algorithm')
    parser.add_argument('--penalty', type=float, default=1.0,
                        help='Penalty parameter of the error term')

    args = parser.parse_args()
    run.log('Kernel type', np.str(args.kernel))
    run.log('Penalty', np.float(args.penalty))

    x_train, x_test, y_train, y_test = retrieve_cleaned_data()

    # training a linear SVM classifier
    from sklearn.svm import SVC
    svm_model_linear = SVC(kernel=args.kernel, C=args.penalty).fit(x_train, y_train)
    svm_predictions = svm_model_linear.predict(x_test)

    # model accuracy for X_test
    accuracy = svm_model_linear.score(x_test, y_test)
    print('Accuracy of SVM classifier on test set: {:.2f}'.format(accuracy))
    run.log('Accuracy', np.float(accuracy))
    # creating a confusion matrix
    cm = confusion_matrix(y_test, svm_predictions)
    print(cm)

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(svm_model_linear, 'outputs/model.joblib')

if __name__ == '__main__':
    main()
