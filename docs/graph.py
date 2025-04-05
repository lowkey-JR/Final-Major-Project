import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Load the dataset
file_path = "watson_healthcare_modified-Cleaned.csv"
df = pd.read_csv(file_path)
df.dropna(inplace=True)

# Define selected columns
selected_columns = [
    "OverTime", "Age", "Gender", "WorkLifeBalance", "JobSatisfaction",
    "MaritalStatus", "TotalWorkingYears", "Shift", "Education", "JobRole",
    "JobLevel", "DistanceFromHome", "Department", "PercentSalaryHike", "TrainingTimesLastYear"
]

# Preprocessed dataset (Ensure all models use the same X_selected)
X_selected = df[selected_columns]
y = df["Attrition"]

# Normalize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_selected = scaler.fit_transform(X_selected)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Train Models
rf_clf = RandomForestClassifier(n_estimators=100, bootstrap=False, random_state=42).fit(X_train, y_train)
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42).fit(X_train, y_train)
lr_clf = LogisticRegression(solver="liblinear", penalty="l1", random_state=42).fit(X_train, y_train)
# Function to plot normal feature importance
def plot_feature_importance(importance, model_name):
    sorted_idx = np.argsort(importance)[::-1]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=importance[sorted_idx], y=np.array(selected_columns)[sorted_idx], palette="coolwarm")
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title(f"Feature Importance - {model_name}")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

# Random Forest Feature Importance
plot_feature_importance(rf_clf.feature_importances_, "Random Forest")

# XGBoost Feature Importance
plot_feature_importance(xgb_clf.feature_importances_, "XGBoost")

# Logistic Regression Feature Importance (absolute coefficients)
plot_feature_importance(np.abs(lr_clf.coef_).flatten(), "Logistic Regression")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define selected columns
selected_columns = [
    "OverTime", "Age", "Gender", "WorkLifeBalance", "JobSatisfaction",
    "MaritalStatus", "TotalWorkingYears", "Shift", "Education", "JobRole",
    "JobLevel", "DistanceFromHome", "Department", "PercentSalaryHike", "TrainingTimesLastYear"
]

# Select only the specified columns
X_selected = df[selected_columns]
y = df["Attrition"]

# Normalize features
scaler = StandardScaler()
X_selected = scaler.fit_transform(X_selected)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Train AdaBoost Classifier
ab_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
ab_clf.fit(X_train, y_train)

# Get feature importances
feature_importances = ab_clf.feature_importances_

# Sort features by importance
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_features = np.array(selected_columns)[sorted_idx]
sorted_importances = feature_importances[sorted_idx]

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=sorted_importances, y=sorted_features, palette="magma")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance (AdaBoostClassifier) - Selected Columns Only")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define selected columns
selected_columns = [
    "OverTime", "Age", "Gender", "WorkLifeBalance", "JobSatisfaction",
    "MaritalStatus", "TotalWorkingYears", "Shift", "Education", "JobRole",
    "JobLevel", "DistanceFromHome", "Department", "PercentSalaryHike", "TrainingTimesLastYear"
]

# Load the dataset  # load the dataframe
file_path = "watson_healthcare_modified-Cleaned.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
df_encoded = df.copy()
categorical_cols = ["OverTime", "Gender", "MaritalStatus", "Shift", "JobRole", "Department"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Convert Attrition to binary (Yes -> 1, No -> 0)
# Handle NaN or unexpected values in 'Attrition'
df_encoded["Attrition"] = df_encoded["Attrition"].map({"Yes": 1, "No": 0}).fillna(-1)  # Fill NaN with -1 or other suitable value
df_encoded = df_encoded[df_encoded["Attrition"] != -1]  # Remove rows with the fill value (-1)


# Select features and target variable after removing NaN rows
X_selected = df_encoded[selected_columns]
y = df_encoded["Attrition"]

# Normalize numerical features
scaler = StandardScaler()
X_selected = scaler.fit_transform(X_selected)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM with GridSearchCV for best hyperparameters
svm_clf = SVC(kernel="linear", random_state=42)
param_grid = {'C': [0.1, 1, 10, 100]}
search = GridSearchCV(svm_clf, param_grid=param_grid, scoring='roc_auc', cv=3, refit=True, verbose=0)
search.fit(X_train, y_train)

# Best SVM Model with linear kernel
svm_linear = SVC(kernel="linear", C=search.best_params_["C"], random_state=42)
svm_linear.fit(X_train, y_train)

# Get feature coefficients (impact on attrition prediction)
feature_importance = svm_linear.coef_.flatten()

# Sort features by impact
sorted_idx = np.argsort(feature_importance)
sorted_features = np.array(selected_columns)[sorted_idx]
sorted_importances = feature_importance[sorted_idx]

# Plot feature prediction impact
plt.figure(figsize=(10, 5))
plt.barh(sorted_features, sorted_importances, color="skyblue")
plt.axvline(x=0, color="gray", linestyle="--")
plt.xlabel("Change in Attrition Prediction")
plt.ylabel("Feature")
plt.title("Feature Prediction Impact on Attrition - SVM")
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Define selected columns
selected_columns = [
    "OverTime", "Age", "Gender", "WorkLifeBalance", "JobSatisfaction",
    "MaritalStatus", "TotalWorkingYears", "Shift", "Education", "JobRole",
    "JobLevel", "DistanceFromHome", "Department", "PercentSalaryHike", "TrainingTimesLastYear"
]

file_path = "watson_healthcare_modified-Cleaned.csv"
df = pd.read_csv(file_path)

# Select only the specified columns
X_selected = df[selected_columns]
y = df["Attrition"]

categorical_cols = ['OverTime', 'Gender', 'MaritalStatus', 'Shift', 'JobRole', 'Department']  # Add other categorical columns if needed
for col in categorical_cols:
    le = LabelEncoder()
    X_selected[col] = le.fit_transform(X_selected[col]) # Encode categorical columns before scaling


# Normalize features
scaler = StandardScaler()
X_selected = scaler.fit_transform(X_selected)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Train CatBoost Classifier
cb_clf = CatBoostClassifier(random_state=42, verbose=0)
cb_clf.fit(X_train, y_train)

# Get feature importances
feature_importances = cb_clf.get_feature_importance()

# Sort features by importance
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_features = np.array(selected_columns)[sorted_idx]
sorted_importances = feature_importances[sorted_idx]

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=sorted_importances, y=sorted_features, palette="plasma")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance (CatBoostClassifier) - Selected Columns Only")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()