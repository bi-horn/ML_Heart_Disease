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
    Implement different ML approches (KNN, Random Forest, ..) and compare recall, accuracy and loss 
    

Conclusion:
