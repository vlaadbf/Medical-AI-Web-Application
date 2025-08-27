import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# Config
model_path = "mobilenet_lung_model.h5"
class_names = ["Benign", "Malignant"]
image_size = (224, 224)

# Folders cu pacienți
root_dirs = {
    "Benign": "bun",
    "Malignant": "rau"
}

model = load_model(model_path)

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize(image_size)
    img_array = np.array(img)
    img_preprocessed = preprocess_input(img_array)
    img_expanded = np.expand_dims(img_preprocessed, axis=0)
    prediction = model.predict(img_expanded)[0]
    return prediction

results = []

for true_label, root in root_dirs.items():
    for patient_id in os.listdir(root):
        patient_path = os.path.join(root, patient_id)
        if not os.path.isdir(patient_path):
            continue

        patient_preds = []

        for img_file in os.listdir(patient_path):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(patient_path, img_file)
            pred = predict_image(img_path)
            pred_label = class_names[np.argmax(pred)]
            confidence = float(np.max(pred))
            results.append({
                "patient_id": patient_id,
                "image": img_file,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": confidence
            })
            patient_preds.append(pred)

        if patient_preds:
            mean_pred = np.mean(patient_preds, axis=0)
            final_label = class_names[np.argmax(mean_pred)]
            print(f"📊 {patient_id} -> Predicție medie: {final_label} ({np.max(mean_pred)*100:.2f}%)")

# Salvăm toate rezultatele
df = pd.DataFrame(results)
df.to_csv("predictii_pacienti_test.csv", index=False)
print("✅ Rezultatele au fost salvate în predictii_pacienti_test.csv")
