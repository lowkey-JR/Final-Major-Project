# Library for Data Manipulation
import numpy as np
import pandas as pd

# Library for Statistical Modelling
from sklearn.preprocessing import LabelEncoder

# Library to Ignore Warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Load the dataset
employee_data = pd.read_csv(r'hospital_employee_dataset-Revised.csv')

# Print the shape of the DataFrame
print("The shape of data frame:", employee_data.shape)
# Print the length (number of rows) of the DataFrame
print("Number of Rows in the dataframe:", len(employee_data))
# Print the number of columns in the DataFrame
print("Number of Columns in the dataframe:", len(employee_data.columns))

print("Column labels in the dataset in column order:")
for column in employee_data.columns:
    print(column)

# Print a summary of the dataframe
print(employee_data.info(verbose=True))

# Display a sample of numerical columns with styling
employee_data.select_dtypes(np.number).sample(5).style.set_properties(**{'background-color': '#E9F6E2', 'color': 'black', 'border-color': '#8b8c8c'})

# Calculate the number of missing values in each column
missing_df = employee_data.isnull().sum().to_frame().rename(columns={0: "Total No. of Missing Values"})
missing_df["% of Missing Values"] = round((missing_df["Total No. of Missing Values"] / len(employee_data)) * 100, 2)
print(missing_df)

# Display statistical summary
print(employee_data.describe().T)

# Drop unnecessary columns (modify based on dataset)
columns_to_drop = ['EmployeeCount', 'EmployeeID', 'Over18', 'StandardHours']
employee_data.drop(columns=[col for col in columns_to_drop if col in employee_data.columns], axis="columns", inplace=True)

# Print the updated shape of the DataFrame
print("The shape of data frame after column drop:", employee_data.shape)

# Print updated column labels
print("Updated column labels in the dataset:")
for column in employee_data.columns:
    print(column)

# Display summary of categorical columns
print(employee_data.describe(include="O").T)

# Calculate the number of unique values in each column
for column in employee_data.columns:
    print(f"{column} - Number of unique values : {employee_data[column].nunique()}")
    print("=============================================================")

# Identify categorical features with <= 30 unique values
categorical_features = [column for column in employee_data.columns if employee_data[column].dtype == object and len(employee_data[column].unique()) <= 30]

for column in categorical_features:
    print(f"{column} : {employee_data[column].unique()}")
    print(employee_data[column].value_counts())
    print("====================================================================================")

if 'Attrition' in categorical_features:
    categorical_features.remove('attrition')

# Save the cleaned DataFrame to a new CSV file
employee_data.to_csv('hospital_employee_dataset-Cleaned.csv', index=False)

print("Processed dataset saved as: hospital_employee_dataset-Cleaned.csv")
