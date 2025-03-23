# Library for Data Manipulation
import numpy as np
import pandas as pd

# Library for Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white",font_scale=1.5)
sns.set(rc={"axes.facecolor":"#FFFAF0","figure.facecolor":"#FFFAF0"})
sns.set_context("poster",font_scale = .7)

# Library to perform Statistical Analysis.
from scipy import stats
from scipy.stats import chi2
from scipy.stats import chi2_contingency

# Library for Ignore the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

employee_data = pd.read_csv(r'hospital_employee_dataset-Cleaned.csv')

columns_to_drop = ['attendance']
employee_data.drop(columns=[col for col in columns_to_drop if col in employee_data.columns], axis="columns", inplace=True)

num_cols = employee_data.select_dtypes(np.number).columns
new_df = employee_data.copy()

new_df["attrition"] = new_df["attrition"].replace({"No":0,"Yes":1})
f_scores = {}
p_values = {}

for column in num_cols:
    f_score, p_value = stats.f_oneway(new_df[column],new_df["attrition"])

    f_scores[column] = f_score
    p_values[column] = p_value

plt.figure(figsize=(12, 8))  # Adjusted figure size for better readability

sns.barplot(y=list(f_scores.keys()), x=list(f_scores.values()), palette="viridis")  # Horizontal barplot

plt.title("ANOVA Test F-Scores Comparison", fontweight="bold", fontsize=16, pad=15)
plt.xlabel("F-Score", fontsize=12)
plt.ylabel("Features", fontsize=12)

# Annotate bars with F-Score values
for index, value in enumerate(f_scores.values()):
    plt.text(value + 0.5, index, f"{int(value)}", ha="left", va="center", fontweight="bold", fontsize=10)

plt.tight_layout()  # Adjust layout to fit labels
plt.show()

# Convert results to DataFrame
test_df = pd.DataFrame({"Features": list(f_scores.keys()), "F_Score": list(f_scores.values())})
test_df["P_value"] = [format(p, '.20f') for p in list(p_values.values())]

test_df = pd.DataFrame({"Features": list(f_scores.keys()), "F_Score": list(f_scores.values())})
test_df["P_value"] = [format(p, '.20f') for p in list(p_values.values())]
print(test_df) 

employee_data.drop(['specializations', 'supervisor_name', 'employee_name'], axis="columns", inplace=True)
cat_cols = employee_data.select_dtypes(include="object").columns.tolist()
cat_cols.remove("attrition")
chi2_statistic = {}
p_values = {}

# Perform chi-square test for each column
for col in cat_cols:
    contingency_table = pd.crosstab(employee_data[col], employee_data['attrition'])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    chi2_statistic[col] = chi2
    p_values[col] = p_value

columns = list(chi2_statistic.keys())
values = list(chi2_statistic.values())

plt.figure(figsize=(12, 8))  # Adjusted figure size for better readability

sns.barplot(y=list(chi2_statistic.keys()), x=list(chi2_statistic.values()), palette="viridis")  # Horizontal barplot

plt.title("Chi-Square Statistic Values for Categorical Features", fontweight="bold", fontsize=16, pad=15)
plt.xlabel("Chi-Square Statistic", fontsize=12)
plt.ylabel("Features", fontsize=12)

# Annotate bars with Chi-Square values
for index, value in enumerate(chi2_statistic.values()):
    plt.text(value + 0.5, index, round(value, 2), ha="left", va="center", fontweight="bold", fontsize=10)

plt.tight_layout()  # Adjust layout to fit labels
plt.show()


test_df = pd.DataFrame({"Features":columns,"Chi_2 Statistic":values})
test_df["P_value"] =  [format(p, '.20f') for p in list(p_values.values())]
print(test_df)