# Capstone Project - Azure Machine Learning Engineer

## Introduction

In this capstone project as part of Udacity's Machine Learning Engineer with Microsoft Azure Nanodegree Program, we will be selecting a credit card customers problem to solve.

This information would be useful for credit card companies to proactively approach likely churning customers to provide them better services and turn customers' decisions in the opposite direction.

Two models will be created: one using Automated ML (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using HyperDrive. We will then compare the performance of both the models and deploy the best performing model.

## Dataset
### Overview
We will be using a dataset from Kaggle located [here](https://www.kaggle.com/sakshigoyal7/credit-card-customers), which comprises of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc.

### Task
The goal is to predict if customers are likely to churn from their credit card services.
The original dataset comprises 23 columns, but we have removed the last 2 columns (NAIVE BAYES CLAS...) which appear to be derived fields and do not gel with the other features in the dataset.
The remaining 21 columns are described below:

- CLIENTNUM: Client number, which is a unique identifier for the customer holding the account
- Attrition_Flag: Variable based on customer activity, if the account is closed then 1 else 0
- Customer_Age: Customer's Age in Years
- Gender: M=Male, F=Female
- Dependent_count: Number of dependents
- Education_Level: Educational Qualification of the account holder (example: high school, college graduate, etc.)
- Marital_Status: Married, Single, Divorced, Unknown
- Income_Category: Annual Income Category of the account holder (< $40K, $40K - 60K, $60K - $80K, $80K-$120K, > $120K)
- Card_Category: Type of Credit Card (Blue, Silver, Gold, Platinum)
- Months_on_book: Period of relationship with bank
- Total_Relationship_Count: Total no. of products held by the customer
- Months_Inactive_12_mon: No. of months inactive in the last 12 months
- Contacts_Count_12_mon: No. of Contacts in the last 12 months
- Credit_Limit: Credit Limit on the Credit Card
- Total_Revolving_Bal: Total Revolving Balance on the Credit Card
- Avg_Open_To_Buy: Open to Buy Credit Line (Average of last 12 months)
- Total_Amt_Chng_Q4_Q1: Change in Transaction Amount (Q4 over Q1)
- Total_Trans_Amt: Total Transaction Amount (Last 12 months)
- Total_Trans_Ct: Total Transaction Count (Last 12 months)
- Total_Ct_Chng_Q4_Q1: Change in Transaction Count (Q4 over Q1)
- Avg_Utilization_Ratio: Average Card Utilization Ratio

### Access

To access the data in our workspace, we downloaded the dataset from Kaggle and uploaded it in this Github rep.
Next, we register the CSV file as a dataset within Azure ML, so that it can be used by Jupyter notebook running on an Azure ML compute instance.

## Automated ML

### Settings:
- Primary metric is set to accuracy, which is the metric AutoML will try to optimize for.
- Early stopping is set to true, enabling this setting will terminate the training when the score is no longer improving.
- The maximum number of concurrent iterations is set to 5.

### Configuration:
- AutoML is configured to solve a classification task.
- The label column is 'Attrition_Flag', which can take on the value of either 'Existing Customer' or 'Attrited Customer'.
- There are 3 cross validations performed to evaluate the predictive model.
- The experiment is set to timeout in 20 minutes, which is the total time taken all iterations combined can take before the experiment terminates.

### Parameters
Consequently, the parameters are as follows:

- Type of Task `task`: classification
- Primary Metric `primary_metric`: accuracy
- Metric Operation `metric_operation`: maximise
- Training Data used `training_data`: dataset (refer to Bankchurners.csv)
- Name of the Label Column `label_column_name`: Attrition_Flag
- Compute Target to run the AutoML experiment on `compute_target`: demo-cluster (compute cluster created for this experiment)
- Sample Weight Column `weight_column_name`: null
- Number of Cross Validations `n_cross_validations`: 3
- Featurization Config `featurization`: auto
- Maximum number of threads to use for a given training iteration `max_cores_per_iteration`: 1
- Maximum number of iterations that would be executed in parallel `max_concurrent_iterations`: 5
- Maximum time in minutes that each iteration can run for before it terminates `iteration_timeout_minutes`: null
- Maximum time in minutes that all iterations combined can take before the experiment terminates`experiment_timeout_minutes`: 20
- Supported Models `supported_models`:
```
["KNN","SGD","AveragedPerceptronClassifier","LinearSVM","ExtremeRandomTrees","LightGBM","TensorFlowDNN","TensorFlowLinearClassifier","RandomForest","LogisticRegression","SVM","XGBoostClassifier","BernoulliNaiveBayes","GradientBoosting","DecisionTree","MultinomialNaiveBayes"]
```
- Whether to enable early termination if the score is not improving in the short term `enable_early_stopping`: True
- Number of iterations for the early stopping window `early_stopping_n_iters`: 10
- The verbosity level for writing to the log file `verbosity`: 20

### Results
36 models were tested and Voting Ensemble was the best performing model found using AutoML, achieving an accuracy of 97.06%.
The best performing Soft Voting Ensemble algorithm used a LightGBM Classifier with a MaxAbsScaler wrapper.

The other metrics are described below:
- AUC macro: 99.267%
- AUC micro: 99.602%
- AUC weighted: 99.267%

The additional characteristics of the best performing model are described below:
- Steps: MaxAbsScaler with copy = True and LightGBMClassifier with boosting_type = Gradient Boosting Decision Tree (GBDT)
- Flatten_transform: None
- Weights: 
```
[0.2857142857142857, 0.07142857142857142,
0.21428571428571427, 0.07142857142857142,
0.07142857142857142, 0.14285714285714285,
0.07142857142857142, 0.07142857142857142])
```

AzureML RunDetails Widget
![alt text](https://github.com/riokorb/nd00333-capstone/blob/master/img01.PNG?raw=true)

Best Model with Run ID
![alt text](https://github.com/riokorb/nd00333-capstone/blob/master/img02.PNG?raw=true)

Run Details
![alt text](https://github.com/riokorb/nd00333-capstone/blob/master/automlrun.PNG?raw=true)

## Hyperparameter Tuning

In this project, we created a Support Vector Machine (SVM) model from scikit-learn. 
It was selected for this project as it is a versatile supervised learning method that works well in high dimensional spaces, and can be applied to solve classification problems.

### Parameters

The parameters used were:
- Kernel type, where the choice of kernel affects the mapping of observations into the feature space.
- Regularization parameter C, which is a penalty parameter that tells the SVM how much we want to avoid misclassifying each training example eg. via overfitting.

The parameters were then tuned based on the choices below:
- Kernel type = choice('linear', 'rbf', 'poly', 'sigmoid'),
- C (represented by penalty) = choice(0.5, 1, 1.5)

### Results
The best model from the hyperdrive experiment achieved an accuracy of 88.48%.

The parameters are described below:
- Kernel type = Linear
- C (represented by penalty) = 1

HyperDrive RunDetails Widget
![alt text](https://github.com/riokorb/nd00333-capstone/blob/master/img03.PNG?raw=true)

Best Model with Run ID and Tuned Hyperparameters
![alt text](https://github.com/riokorb/nd00333-capstone/blob/master/img04.PNG?raw=true)

Run Details
![alt text](https://github.com/riokorb/nd00333-capstone/blob/master/hyperdriverun.PNG?raw=true)


## Future Improvements
The accuracy of the data might also be affected by class imbalance as the dataset only comprises 16.07% of customers who have churned.
Thus, it may be difficult to train our model to predict churning customers and can lead to a falsely perceived positive effect of a model's accuracy, because the input data is biased towards a certain class.
By collecting more data (especially of churned customers) to the dataset, the results could have been improved. 

For the AutoML experiment, we could also try incorporating deep learning methods to see if better results can be obtained.
For the Hyperdrive experiment, repeating the runs with Grid Sampling instead of Random Sampling might lead to better outcomes since it uses all the possible values from the search space. 

More extensive data cleaning and featurization efforts could allow us to achieve better accuracy scores for both experiments.
We could also revisit the sampling methods and consider custom cross-validation strategies to try and mitigate the impact of imbalanced data on performance metrics.

## Model Deployment
For this project, the Voting Ensemble model from our AutoML experiment achieved the best accuracy and will be selected for deployment as a webservice.

To do this, we register this best-performing model in our workspace and make use of the InferenceConfig from our score.py file for the deployment.

After deploying the REST endpoint for inference,  we can query the endpoint by making a request to the webservice.

Active Model Endpoint
![alt text](https://github.com/riokorb/nd00333-capstone/blob/master/img05.PNG?raw=true)

Value returned after sending sample input/payload
![alt text](https://github.com/riokorb/nd00333-capstone/blob/master/sampleinput.PNG?raw=true)

## Screen Recording

[Link to Screen Recording](https://drive.google.com/file/d/14P32IMIh9BowA4rOnNlXEJdxtWfZpdnx/view?usp=sharing)
