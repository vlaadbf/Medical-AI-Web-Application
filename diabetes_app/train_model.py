import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Încarcă datele
df = pd.read_csv('diabetes.csv')

# 2. Înlocuiește valorile 0 cu NaN în coloanele relevante
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# 3. Separă X și y
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 4. Imputare cu mediana
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 5. Scălare
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 6. Împărțire în train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Antrenare model: Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# 8. Alternativ: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 9. Evaluare
print("Logistic Regression:")
print(classification_report(y_test, lr.predict(X_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr.predict(X_test)))

print("\nRandom Forest:")
print(classification_report(y_test, rf.predict(X_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf.predict(X_test)))

# 10. Salvare model Random Forest (cel mai bun)
joblib.dump(rf, 'model/diabetes_model.pkl')
joblib.dump(imputer, 'model/imputer.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
