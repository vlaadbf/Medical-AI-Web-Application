import os
import pydicom
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

def is_dicom_file(path):
    try:
        pydicom.dcmread(path, stop_before_pixels=True)
        return True
    except:
        return False

def preprocess_image(img):
    img = img.convert("L")

    # Doar normalizare contrast, fără histogram equalization (opțional)
    # img = ImageOps.equalize(img)  # Poți comenta asta complet

    # Contrast și luminozitate foarte moderate
    img = ImageEnhance.Contrast(img).enhance(1.05)
    img = ImageEnhance.Brightness(img).enhance(1.02)
    
    return img

def convert_dicoms_to_png_flat(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    found_files = False
    count = 1  # număr pentru fișiere

    for root, _, files in os.walk(input_folder):
        for file in files:
            full_path = os.path.join(root, file)

            if is_dicom_file(full_path):
                found_files = True
                try:
                    ds = pydicom.dcmread(full_path)
                    pixel_array = ds.pixel_array

                    # Normalizare 0–255
                    image = ((pixel_array - pixel_array.min()) / pixel_array.ptp() * 255.0).astype(np.uint8)
                    img = Image.fromarray(image)

                    # Prelucrare imagine
                    processed_img = preprocess_image(img)

                    # Salvare direct în folderul de output, numerotat
                    output_path = os.path.join(output_folder, f"{count}.png")
                    processed_img.save(output_path)
                    print(f"✅ Salvat: {output_path}")
                    count += 1

                except Exception as e:
                    print(f"❌ Eroare la {full_path}: {e}")

    if not found_files:
        print("⚠️ Nu s-au găsit fișiere DICOM valide.")

if __name__ == "__main__":
    input_folder = r"C:\Users\capri\Desktop\dizertatie\DOAMNA ANDREEA\DICOM"
    output_folder = r"C:\Users\capri\Desktop\dizertatie\DOAMNA ANDREEA\PNG_PROCESAT"
    convert_dicoms_to_png_flat(input_folder, output_folder)
