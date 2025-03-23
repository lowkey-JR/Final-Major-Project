#Loading all Packages
print("==================== BLOCK 1 Started! ======================")

# Library for Data Manipulation
import numpy as np
import pandas as pd

#Library for Data Visualization.
import seaborn as sns
import matplotlib.pyplot as plt
import hvplot

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# Library for Statistical Modelling
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier

print("==================== Packages Loaded ======================")

# Library for Ignore the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

employee_data = pd.read_csv(r'hospital_employee_dataset-Cleaned.csv')

# Convert categorical variables into numerical form.
label = LabelEncoder()
employee_data["attrition"] = label.fit_transform(employee_data.attrition)
employee_data.info()

# Transform categorical data into dummies
dummy_col = [column for column in employee_data.drop('attrition', axis=1).columns if employee_data[column].nunique() < 20]
data = pd.get_dummies(employee_data, columns=dummy_col, drop_first=True, dtype='uint8')
data.info()

print(data.shape)

# Remove duplicate Features
data = data.T.drop_duplicates()
data = data.T

# Remove Duplicate Rows
data.drop_duplicates(inplace=True)

print(data.shape)

# Transform categorical data into dummies, including all object type columns.
dummy_col = [column for column in employee_data.drop('attrition', axis=1).columns if employee_data[column].dtype == 'object']
data = pd.get_dummies(employee_data, columns=dummy_col, drop_first=True, dtype='uint8')

# Now calculate the feature correlation.
feature_correlation = data.drop('attrition', axis=1).corrwith(data.attrition).sort_values()
model_col = feature_correlation[np.abs(feature_correlation) > 0.02].index
len(model_col)

X = data.drop('attrition', axis=1)
y = data.attrition

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
X_std = scaler.transform(X)

def feature_imp(df, model):
    fi = pd.DataFrame()
    fi["feature"] = df.columns
    fi["importance"] = model.feature_importances_
    return fi.sort_values(by="importance", ascending=False)

print(y_test.value_counts()[0] / y_test.shape[0])

stay = (y_train.value_counts()[0] / y_train.shape)[0]
leave = (y_train.value_counts()[1] / y_train.shape)[0]

print("===============TRAIN=================")
print(f"Staying Rate: {stay * 100:.2f}%")
print(f"Leaving Rate: {leave * 100 :.2f}%")

stay = (y_test.value_counts()[0] / y_test.shape)[0]
leave = (y_test.value_counts()[1] / y_test.shape)[0]

print("===============TEST=================")
print(f"Staying Rate: {stay * 100:.2f}%")
print(f"Leaving Rate: {leave * 100 :.2f}%")

def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    print("TRAINING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

    print("TESTING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

#RANDOM FOREST CLASSIFIER
rf_clf = RandomForestClassifier(n_estimators=100, bootstrap=False,
#                                      class_weight={0:stay, 1:leave}
   )
rf_clf.fit(X_train, y_train)
print("RANDOM FOREST CLASSIFIER")
evaluate(rf_clf, X_train, X_test, y_train, y_test)

#XGBOOST CLASSIFIER
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)
print("XGBOOST CLASSIFIER")
evaluate(xgb_clf, X_train, X_test, y_train, y_test)

# Get probability predictions from both models
rf_probs = rf_clf.predict_proba(X_test)[:, 1]  # Probabilities for class 1
xgb_probs = xgb_clf.predict_proba(X_test)[:, 1]

# Average the probabilities
hybrid_probs = (rf_probs + xgb_probs) / 2

# Convert probabilities to class labels (Threshold = 0.5)
hybrid_preds = (hybrid_probs >= 0.5).astype(int)

# Evaluate Hybrid Model
print("HYBRID MODEL XGBRF RESULTS: \n===============================")
clf_report = pd.DataFrame(classification_report(y_test, hybrid_preds, output_dict=True))
print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, hybrid_preds)}")
print(f"ACCURACY SCORE:\n{accuracy_score(y_test, hybrid_preds):.4f}")
print(f"CLASSIFICATION REPORT:\n{clf_report}")

# AUC Score
print(f"AUC Score: {roc_auc_score(y_test, hybrid_probs):.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, hybrid_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Hybrid Model (AUC = {roc_auc_score(y_test, hybrid_probs):.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Hybrid Model")
plt.legend()
plt.show()

# Train Logistic Regression
lr_clf = LogisticRegression(solver='liblinear', penalty='l1')
lr_clf.fit(X_train_std, y_train)

# Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, bootstrap=False)
rf_clf.fit(X_train, y_train)

# Evaluate individual models
evaluate(lr_clf, X_train_std, X_test_std, y_train, y_test)
evaluate(rf_clf, X_train, X_test, y_train, y_test)

# Store individual model scores
scores_dict = {
    'Logistic Regression': {
        'Train': roc_auc_score(y_train, lr_clf.predict(X_train_std)),
        'Test': roc_auc_score(y_test, lr_clf.predict(X_test_std)),
    },
    'Random Forest': {
        'Train': roc_auc_score(y_train, rf_clf.predict(X_train)),
        'Test': roc_auc_score(y_test, rf_clf.predict(X_test)),
    }
}

# Hybrid Model: Averaging Probabilities
lr_probs = lr_clf.predict_proba(X_test_std)[:, 1]  # Probabilities for class 1
rf_probs = rf_clf.predict_proba(X_test)[:, 1]

hybrid_probs = (lr_probs + rf_probs) / 2  # Average probabilities
hybrid_preds = (hybrid_probs >= 0.5).astype(int)  # Convert to class labels

# Evaluate Hybrid Model
print("HYBRID MODEL LOGISTIC AND RANDOM RESULTS: \n===============================")
clf_report = pd.DataFrame(classification_report(y_test, hybrid_preds, output_dict=True))
print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, hybrid_preds)}")
print(f"ACCURACY SCORE:\n{accuracy_score(y_test, hybrid_preds):.4f}")
print(f"CLASSIFICATION REPORT:\n{clf_report}")

# AUC Score
hybrid_auc = roc_auc_score(y_test, hybrid_probs)
print(f"AUC Score: {hybrid_auc:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, hybrid_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Hybrid Model (AUC = {hybrid_auc:.4f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Hybrid Model")
plt.legend()
plt.show()

# Add Hybrid Model Score
scores_dict['Hybrid Model (LR + RF)'] = {
    'Train': roc_auc_score(y_train, (lr_clf.predict_proba(X_train_std)[:, 1] + rf_clf.predict_proba(X_train)[:, 1]) / 2),
    'Test': hybrid_auc,
}

# Print Model Performance Summary
pd.DataFrame(scores_dict)
