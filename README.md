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
    

Results:

Only chosen models like KNN, RF and XGBoost were tested:

## Model Performance Metrics

# Recall
table_content = [
    ["Sl. No.", "Classification Model", "Train Accuracy (%)", "Test Accuracy (%)"],
    ["1", "K Nearest Neighbors", f"{knn_train_accuracy*100:.2f}", f"{knn_test_accuracy*100:.2f}"],
    ["2", "Random Forests", f"{rf_train_accuracy*100:.2f}", f"{rf_test_accuracy*100:.2f}"],
    ["3", "XG Boost", f"{xgb_train_accuracy*100:.2f}", f"{xgb_test_accuracy*100:.2f}"]
]

# Calculate the maximum width of each column
max_widths = [max(len(str(row[i])) for row in table_content) for i in range(len(table_content[0]))]

# Generate the Markdown table
markdown_table = "\n".join([" | ".join(cell.ljust(max_widths[i]) for i, cell in enumerate(row)) for row in table_content])

# Add Markdown table headers
markdown_table = markdown_table + "\n" + "|".join(["-" * width for width in max_widths])

# Print the Markdown table
print(markdown_table)




