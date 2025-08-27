# === 1. Importuri ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
import joblib
import os

# === 2. Încarcă datele și înlocuiește 0 cu NaN ===
df = pd.read_csv('diabetes.csv')
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# === 3. Feature engineering ===
df['AgeCategory'] = pd.cut(df['Age'], bins=[20, 30, 50, 100], labels=['Young', 'Adult', 'Senior'])
df['IsObese'] = (df['BMI'] > 30).astype(int)
df['Glucose_BMI_ratio'] = df['Glucose'] / df['BMI'].replace(0, np.nan)
df['HighGlucose'] = (df['Glucose'] > 140).astype(int)
df['Has_Insulin'] = df['Insulin'].notna().astype(int)
df = pd.get_dummies(df, columns=['AgeCategory'], drop_first=True)

# === 4. Separare X și y ===
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# === 5. Imputare și scalare ===
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# === 6. Train/Test split + SMOTE ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# === 7. Grid Search Random Forest ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_resampled, y_resampled)
best_model = grid_search.best_estimator_

# === 8. Predictii si evaluare standard ===
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("Cel mai bun model:", grid_search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_proba))

# === 9. Plot Confusion Matrix standard ===
fig1, ax1 = plt.subplots()
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax1)
ax1.set_title("Matrice Confuzie (threshold = 0.5)")
plt.show()

# === 10. Evaluare cu threshold custom ===
threshold = 0.4
y_pred_custom = (y_proba >= threshold).astype(int)

print(f"\n=== Evaluare cu threshold {threshold} ===")
print("Recall (clasa 1 - diabet):", recall_score(y_test, y_pred_custom))
print("Precision:", precision_score(y_test, y_pred_custom))
print("F1-score:", f1_score(y_test, y_pred_custom))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_custom))

# === 11. Plot Confusion Matrix custom ===
fig2, ax2 = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_custom, ax=ax2)
ax2.set_title(f"Matrice Confuzie (threshold = {threshold})")
plt.show()

# === 12. ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_proba))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend()
plt.grid(True)
plt.show()

# === 13. Importanța variabilelor ===
feature_names = X.columns
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title("Importanța Variabilelor")
plt.xlabel("Importanță")
plt.ylabel("Atribute")
plt.tight_layout()
plt.show()

# === 14. Salvare model și preprocesori ===
os.makedirs('model', exist_ok=True)
joblib.dump(best_model, 'model/diabetes_model.pkl')
joblib.dump(imputer, 'model/imputer.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
