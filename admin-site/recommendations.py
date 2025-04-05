import pandas as pd

# Load your dataset
df = pd.read_csv("admin-site/watson_healthcare_modified-Cleaned.csv")

# Make sure Attrition column is in 0/1 format
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

recommendations = []

# 1. Department with Highest Attrition Rate
dept_attrition = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False)
high_risk_depts = dept_attrition[dept_attrition > 0.15]  # Example threshold: 15%
for dept, rate in high_risk_depts.items():
    recommendations.append(f"⚠️ High attrition detected in the **{dept}** department (Rate: {rate:.2%}).")

# 2. OverTime Impact
overtime_rate = df.groupby('OverTime')['Attrition'].mean()
if overtime_rate.get('Yes', 0) > 2 * overtime_rate.get('No', 1e-5):  # Avoid divide-by-zero
    recommendations.append("⚠️ Employees working overtime have significantly higher attrition rates. Consider monitoring workloads.")

# 3. Job Satisfaction
low_js = df[df['JobSatisfaction'] <= 2]['Attrition'].mean()
high_js = df[df['JobSatisfaction'] >= 4]['Attrition'].mean()
if low_js > high_js * 2:
    recommendations.append("⚠️ Low job satisfaction correlates with higher attrition. Focus on employee engagement.")

# 4. WorkLifeBalance
wlb_group = df.groupby('WorkLifeBalance')['Attrition'].mean()
low_wlb = wlb_group.get(1, 0)
if low_wlb > 0.25:
    recommendations.append("⚠️ Poor work-life balance is a major factor. Consider promoting flexible work policies.")

# 5. Age Groups
df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 60], labels=["18-25", "26-35", "36-45", "46-60"])
age_attrition = df.groupby('AgeGroup')['Attrition'].mean()
young_high = age_attrition[age_attrition > 0.20]
for age_group, rate in young_high.items():
    recommendations.append(f"⚠️ Young employees in age group **{age_group}** have high attrition (Rate: {rate:.2%}). Offer career growth opportunities.")

# 6. Gender Differences
gender_attrition = df.groupby('Gender')['Attrition'].mean()
if abs(gender_attrition['Male'] - gender_attrition['Female']) > 0.05:
    higher_gender = 'Male' if gender_attrition['Male'] > gender_attrition['Female'] else 'Female'
    recommendations.append(f"⚠️ Attrition is higher among **{higher_gender}** employees. Consider investigating gender-specific concerns.")

# Print or export recommendations
print("==== RECOMMENDATIONS TO REDUCE ATTRITION ====")
for rec in recommendations:
    print("- " + rec)

# Optional: Save to a .txt file or JSON to load on your HTML page
with open("recommendations.txt", "w", encoding="utf-8") as f:
    for rec in recommendations:
        f.write(rec + "\n")