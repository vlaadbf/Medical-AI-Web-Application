import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('diabetes.csv')

# 2. General information
print("General Information:")
print(df.info())

# 3. First 5 rows
print("\nFirst 5 rows:")
print(df.head())

# 4. Descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# 5. Check for missing values (0s in places where it's not logical: e.g., glucose = 0)
print("\nSuspicious values (possible missing data):")
print((df == 0).sum())

# Histograms
df.hist(bins=20, figsize=(12, 10))
plt.suptitle("Variable Distributions")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Variables")
plt.show()

# Label distribution (0 = healthy, 1 = diabetic)
sns.countplot(x='Outcome', data=df)
plt.title("Distribution of Patients With/Without Diabetes")
plt.xticks([0, 1], ['Without Diabetes', 'With Diabetes'])
plt.show()

import numpy as np

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

print("\nMissing values after replacement:")
print(df.isnull().sum())
