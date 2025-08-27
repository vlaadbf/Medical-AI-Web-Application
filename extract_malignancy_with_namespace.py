import os
import xml.etree.ElementTree as ET
import pandas as pd

NS = {'lidc': 'http://www.nih.gov'}

def extract_scores_with_namespace(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        scores = []

        for session in root.findall(".//lidc:readingSession", NS):
            for nodule in session.findall(".//lidc:unblindedReadNodule", NS):
                nid = nodule.find("lidc:noduleID", NS)
                if nid is None:
                    continue  # fără ID => nu procesăm

                characteristics = nodule.find("lidc:characteristics", NS)
                if characteristics is not None:
                    malignancy = characteristics.find("lidc:malignancy", NS)
                    if malignancy is not None and malignancy.text and malignancy.text.isdigit():
                        scores.append(int(malignancy.text))

        return scores

    except Exception as e:
        print(f"Eroare la {xml_path}: {e}")
        return []

def get_patient_id(path):
    parts = os.path.normpath(path).split(os.sep)
    for p in reversed(parts):
        if p.startswith("LIDC-IDRI-"):
            return p
    return "UNKNOWN"

def scan_lidc_namespace(base_dir, output_csv="malignancy_labels_namespace.csv"):
    data = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".xml"):
                xml_path = os.path.join(root, file)
                scores = extract_scores_with_namespace(xml_path)
                if not scores:
                    continue

                avg = sum(scores) / len(scores)
                if avg >= 4:
                    label = "Malignant"
                elif avg <= 2:
                    label = "Benign"
                else:
                    label = "Unknown"

                data.append({
                    "xml_file": xml_path,
                    "patient_id": get_patient_id(xml_path),
                    "label": label,
                    "malignancy_scores": scores
                })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Salvat {len(df)} pacienți în {output_csv}")
    if not df.empty:
        print("Distribuție etichete:", df["label"].value_counts(dropna=False))
    else:
        print("⚠️ Nicio etichetă validă găsită.")

if __name__ == "__main__":
    scan_lidc_namespace("D:/dataset-cancer/manifest-1600709154662/LIDC-IDRI")
