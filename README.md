# Machine-Learning---Heart-Disease

## Introduction:

    Early prediction of coronary heart disease (CHD) allows for preventive measures like diet and exercise changes.
    Timely diagnosis can lead to early treatment, avoiding more invasive procedures.

## Problem Statement:

    Objective: Predict whether a patient has a 10-year risk of future CHD.

## Data Description:

    Dataset: Over 4,000 records with 15 attributes.
    Attributes: Include demographic, behavioral, and medical risk factors.

    Demographic:
        Sex: Male or Female
        Age: Patient's age

    Behavioral:
        is_smoking: Current smoker (Yes or No)
        Cigs Per Day: Average daily cigarette consumption

    Medical(History):
        BP Meds: Blood pressure medication status
        Prevalent Stroke: History of stroke
        Prevalent Hyp: Hypertension status
        Diabetes: Diabetes status

    Medical(Current):
        Tot Chol: Total cholesterol level
        Sys BP: Systolic blood pressure
        Dia BP: Diastolic blood pressure
        BMI: Body Mass Index
        Heart Rate: Heart rate
        Glucose: Glucose level

    Target_
        CHD: 10-year risk of CHD (Binary: 1 for Yes, 0 for No)

Approach:

    Data Preparation: Removed null values
    Balance the dataset using SMOTE 
    Implement different ML approches (KNN, Random Forest, XGBoost) and compare recall, accuracy and loss 
    

## Results:

Only chosen models like KNN, RF and XGBoost were tested:

#### Model Performance Metrics
|         | Classification Model | Train Recall (%) | Test Recall (%) |
| ------- | -------------------- | ----------------- | --------------- |
|    1    | K Nearest Neighbors  | 84.88            | 75.17           |
|    2    | Random Forests       | 70.11            | 74.50           |
|    3    | XG Boost             | 80.11            | 87.25           |


|         | Classification Model | Train Accuracy (%) | Test Accuracy (%) |
| ------- | --------------------- | ------------------ | ----------------- |
|    1    | K Nearest Neighbors  | 71.26              | 53.00             |
|    2    | Random Forests       | 67.70              | 58.60             |
|    3    | XG Boost             | 77.18              | 40.41             |

A high recall value is important as it signifies the ability to accurately identify individuals at risk of coronary heart disease (CHD) with a high true positive rate, thereby enabling timely warnings and interventions.
However, it would be desirable to improve accuracy as well and achieve a lower false positive rate.


## Further Investigations: Attempts to Improve Accuracy

Analysis will be continued with neural networks to try to improve the accuracy using neural networks.






