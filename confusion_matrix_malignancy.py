import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Setări
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "output_images_windowed"
MODEL_PATH = "mobilenet_lung_model.h5"

# Încărcare model
model = load_model(MODEL_PATH)

# Încărcare date
val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    label_mode="categorical",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = val_ds.class_names  # ['Benign', 'Malignant']

# Preprocesare imagini (mobilenet)
val_ds = val_ds.map(lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y))

# Predictii
y_true = []
y_pred = []

for batch in val_ds:
    images, labels = batch
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confuzie")
plt.savefig("confusion_matrix.png")
plt.show()

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("🔍 Raport de clasificare:")
print(report)
