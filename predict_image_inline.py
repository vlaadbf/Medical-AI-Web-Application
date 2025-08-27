import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

def predict_image(img_path, model_path="mobilenet_lung_model.h5"):
    # Încarcă modelul
    model = load_model(model_path)
    class_names = ["Benign", "Malignant"]

    # Încarcă și preprocesează imaginea
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_preprocessed = preprocess_input(img_array)
    img_expanded = np.expand_dims(img_preprocessed, axis=0)

    # Prezicere
    prediction = model.predict(img_expanded)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = prediction[np.argmax(prediction)]

    print(f"🧠 Predicție pentru {img_path}: {predicted_class} ({confidence*100:.2f}%)")

if __name__ == "__main__":
    # Modifică aici calea către imaginea ta PNG:
    img_path = r"C:\Users\capri\Desktop\dizertatie\The IQ-OTHNCCD lung cancer dataset\Bengin cases/Bengin case (1).jpg"
    predict_image(img_path)
