# +
# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

input_sample = pd.DataFrame({"CLIENTNUM": pd.Series(["768805383"], dtype="int64"), "Attrition_Flag": pd.Series(["Existing Customer"], dtype="object"), "Customer_Age": pd.Series(["45"], dtype="int64"), "Gender": pd.Series(["M"], dtype="object"), "Dependent_count": pd.Series(["3"], dtype="int64"), "Education_Level": pd.Series(["High School"], dtype="object"), "Marital_Status": pd.Series(["Married"], dtype="object"), "Card_Category": pd.Series(["Blue"], dtype="object"), "Months_on_book": pd.Series(["39"], dtype="int64"), "Total_Relationship_Count": pd.Series(["5"], dtype="int64"), "Months_Inactive_12_mon": pd.Series(["1"], dtype="int64"), "Contacts_Count_12_mon": pd.Series(["3"], dtype="int64"), "Credit_Limit": pd.Series(["12691.0"], dtype="float64"), "Total_Revolving_Bal": pd.Series(["777"], dtype="int64"), "Avg_Open_To_Buy": pd.Series(["11914.0"], dtype="float64"), "Total_Amt_Chng_Q4_Q1": pd.Series(["1.335"], dtype="float64"), "Total_Trans_Amt":
pd.Series(["1144"], dtype="int64"), "Total_Trans_Ct": pd.Series(["42"], dtype="int64"), "Total_Ct_Chng_Q4_Q1": pd.Series(["1.625"], dtype="float64"), "Avg_Utilization_Ratio": pd.Series(["0.061"], dtype="float64")})
output_sample = np.array([0])


# -

def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = Model.get_model_path("best-automl-model")
    print(model_path)
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})