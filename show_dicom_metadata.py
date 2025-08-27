import pydicom

def show_dicom_metadata(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    print("📄 Metadate pentru fișierul DICOM:")
    print("-" * 50)
    for elem in ds:
        print(f"{elem.tag} : {elem.name} = {elem.value}")

if __name__ == "__main__":
    # Înlocuiește cu calea către un fișier .dcm real
    dicom_path = r"D:\dataset-cancer\manifest-1600709154662\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-NA-NA-30178\3000566.000000-NA-03192/1-001.dcm"
    show_dicom_metadata(dicom_path)
##################################################
📄 Metadate pentru fișierul DICOM:
--------------------------------------------------
(0008,0005) : Specific Character Set = ISO_IR 100
(0008,0008) : Image Type = ['ORIGINAL', 'PRIMARY', 'AXIAL']
(0008,0016) : SOP Class UID = 1.2.840.10008.5.1.4.1.1.2
(0008,0018) : SOP Instance UID = 1.3.6.1.4.1.14519.5.2.1.6279.6001.262721256650280657946440242654
(0008,0020) : Study Date = 20000101
(0008,0021) : Series Date = 20000101
(0008,0022) : Acquisition Date = 20000101
(0008,0023) : Content Date = 20000101
(0008,0024) : Overlay Date = 20000101
(0008,0025) : Curve Date = 20000101
(0008,002A) : Acquisition DateTime = 20000101
(0008,0030) : Study Time = 
(0008,0032) : Acquisition Time = 
(0008,0033) : Content Time = 
(0008,0050) : Accession Number = 
(0008,0060) : Modality = CT
(0008,0070) : Manufacturer = GE MEDICAL SYSTEMS
(0008,0090) : Referring Physician's Name = 
(0008,1090) : Manufacturer's Model Name = LightSpeed Plus
(0008,1155) : Referenced SOP Instance UID = 1.3.6.1.4.1.14519.5.2.1.6279.6001.167780047448237579267150010168
(0010,0010) : Patient's Name = 
(0010,0020) : Patient ID = LIDC-IDRI-0001
(0010,0030) : Patient's Birth Date = 
(0010,0040) : Patient's Sex = 
(0010,1010) : Patient's Age = 
(0010,21D0) : Last Menstrual Date = 20000101
(0012,0062) : Patient Identity Removed = YES
(0012,0063) : De-identification Method = DCM:113100/113105/113107/113108/113109/113111
(0013,0010) : Private Creator = CTP
(0013,1010) : Private tag data = LIDC-IDRI
(0013,1013) : Private tag data = 62796001
(0018,0010) : Contrast/Bolus Agent = IV
(0018,0015) : Body Part Examined = CHEST
(0018,0022) : Scan Options = HELICAL MODE
(0018,0050) : Slice Thickness = 2.500000
(0018,0060) : KVP = 120
(0018,0090) : Data Collection Diameter = 500.000000
(0018,1020) : Software Versions = LightSpeedApps2.4.2_H2.4M5
(0018,1100) : Reconstruction Diameter = 360.000000
(0018,1110) : Distance Source to Detector = 949.075012
(0018,1111) : Distance Source to Patient = 541.000000
(0018,1120) : Gantry/Detector Tilt = 0.000000
(0018,1130) : Table Height = 144.399994
(0018,1140) : Rotation Direction = CW
(0018,1150) : Exposure Time = 570
(0018,1151) : X-Ray Tube Current = 400
(0018,1152) : Exposure = 4684
(0018,1160) : Filter Type = BODY FILTER
(0018,1170) : Generator Power = 48000
(0018,1190) : Focal Spot(s) = 1.200000
(0018,1210) : Convolution Kernel = STANDARD
(0018,5100) : Patient Position = FFS
(0020,000D) : Study Instance UID = 1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178
(0020,000E) : Series Instance UID = 1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192
(0020,0010) : Study ID = 
(0020,0011) : Series Number = 3000566
(0020,0013) : Instance Number = 1
(0020,0032) : Image Position (Patient) = [-166.000000, -171.699997, -10.000000]
(0020,0037) : Image Orientation (Patient) = [1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
(0020,0052) : Frame of Reference UID = 1.3.6.1.4.1.14519.5.2.1.6279.6001.229925374658226729607867499499
(0020,1040) : Position Reference Indicator = SN
(0020,1041) : Slice Location = -10.000000
(0028,0002) : Samples per Pixel = 1
(0028,0004) : Photometric Interpretation = MONOCHROME2
(0028,0010) : Rows = 512
(0028,0011) : Columns = 512
(0028,0030) : Pixel Spacing = [0.703125, 0.703125]
(0028,0100) : Bits Allocated = 16
(0028,0101) : Bits Stored = 16
(0028,0102) : High Bit = 15
(0028,0103) : Pixel Representation = 1
(0028,0120) : Pixel Padding Value = 63536
(0028,0303) : Longitudinal Temporal Information Modified = MODIFIED
(0028,1050) : Window Center = -600
(0028,1051) : Window Width = 1600
(0028,1052) : Rescale Intercept = -1024
(0028,1053) : Rescale Slope = 1
(0038,0020) : Admitting Date = 20000101
(0040,0002) : Scheduled Procedure Step Start Date = 20000101
(0040,0004) : Scheduled Procedure Step End Date = 20000101
(0040,0244) : Performed Procedure Step Start Date = 20000101
(0040,2016) : Placer Order Number / Imaging Service Request = 
(0040,2017) : Filler Order Number / Imaging Service Request = 
(0040,A075) : Verifying Observer Name = Removed by CTP
(0040,A123) : Person Name = Removed by CTP
(0040,A124) : UID = 1.3.6.1.4.1.14519.5.2.1.6279.6001.242033371867591328384552261733
(0070,0084) : Content Creator's Name = 
(0088,0140) : Storage Media File-set UID = 1.3.6.1.4.1.14519.5.2.1.6279.6001.286658939037720062103202200222