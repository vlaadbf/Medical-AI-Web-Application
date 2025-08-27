import joblib
import pandas as pd

model = joblib.load('model/diabetes_model.pkl')
imputer = joblib.load('model/imputer.pkl')
scaler = joblib.load('model/scaler.pkl')

# Trei pacienți
test_cases = [
    {
        'Nume': 'Sănătos',
        'Pregnancies': 0, 'Glucose': 85, 'BloodPressure': 70, 'SkinThickness': 20, 'Insulin': 80,
        'BMI': 22.5, 'DiabetesPedigreeFunction': 0.2, 'Age': 25,
        'IsObese': 0, 'Glucose_BMI_ratio': 85/22.5, 'HighGlucose': 0, 'Has_Insulin': 1,
        'AgeCategory_Adult': 0, 'AgeCategory_Senior': 0
    },
    {
        'Nume': 'Mediu',
        'Pregnancies': 2, 'Glucose': 120, 'BloodPressure': 74, 'SkinThickness': 32, 'Insulin': 100,
        'BMI': 28.0, 'DiabetesPedigreeFunction': 0.4, 'Age': 38,
        'IsObese': 0, 'Glucose_BMI_ratio': 120/28.0, 'HighGlucose': 0, 'Has_Insulin': 1,
        'AgeCategory_Adult': 1, 'AgeCategory_Senior': 0
    },
    {
        'Nume': 'Cu Diabet',
        'Pregnancies': 4, 'Glucose': 165, 'BloodPressure': 60, 'SkinThickness': 35, 'Insulin': 0,
        'BMI': 35.0, 'DiabetesPedigreeFunction': 0.8, 'Age': 55,
        'IsObese': 1, 'Glucose_BMI_ratio': 165/35.0, 'HighGlucose': 1, 'Has_Insulin': 0,
        'AgeCategory_Adult': 0, 'AgeCategory_Senior': 1
    }
]

for case in test_cases:
    name = case.pop('Nume')
    df = pd.DataFrame([case])
    transformed = scaler.transform(imputer.transform(df))
    prob = model.predict_proba(transformed)[0][1]
    result = 'Are diabet' if prob >= 0.4 else 'Nu are diabet'
    print(f"{name}: {prob:.2f} → {result}")
