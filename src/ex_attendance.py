import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from typing import List
import csv
import sys

global unique_recognized_count

import warnings
warnings.filterwarnings("ignore")

# SUB = "ex_atn"
# CLASS = ""
# TEST = "Minor Project/Facial Recognition Attendance Sysytem/final_dataset/" + CLASS + "/present"
# BIODATA = "Minor Project/Facial Recognition Attendance Sysytem/final_dataset/" + CLASS + "/listed_100_biodata.csv"

# SUB = "ex_atn"
# TEST = ""
# BIODATA = ""

from src.model_development import FaceRecognitionModel

class TestDataset(Dataset):
    def __init__(self, test_dir: str, transform: transforms.Compose):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self) -> List[str]:
        return [os.path.join(self.test_dir, f) for f in os.listdir(self.test_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, os.path.basename(img_path)

def load_model(model_path: str, num_classes: int, architecture: str, device: str):
    model = FaceRecognitionModel(num_classes=num_classes, architecture=architecture)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str, full_name_list: List[str]):
    predictions = {}

    with torch.no_grad():
        for images, image_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            predicted_names = [full_name_list[p] for p in predicted.cpu().tolist()]

            for name, pred in zip(image_names, predicted_names):
                predictions[name] = pred

    return predictions

def update_attendance(predictions: dict, biodata_path: str, sub:str, unique_recognized_count:int):
    df = pd.read_csv(biodata_path)

    updated_names = set()

    for _, name in predictions.items():
        if name not in updated_names:
            df.loc[df['name'] == name, sub] += 1
            
            unique_recognized_count += 1

            updated_names.add(name)
    print("       = ", unique_recognized_count)
    df.to_csv(biodata_path, index=False)

    return unique_recognized_count

def model_testing(sub:str, class_name:str):
    # global TEST, BIODATA

    global unique_recognized_count
    unique_recognized_count = 0

    # Configuration
    test_direc = f"C:\\A Drive\\Machine Learning\\Minor Project\\Facial Recognition Attendance Sysytem\\final_dataset\\{class_name}\\present"
    biodata_direc = f"C:\\A Drive\\Machine Learning\\Minor Project\\Facial Recognition Attendance Sysytem\\final_dataset\\{class_name}\\listed_100_biodata.csv"
    model_path_1 = f"C:\\A Drive\\Machine Learning\\Minor Project\\Facial Recognition Attendance Sysytem\\models\\{class_name}\\resnet50_face_recognition_model.pth"
    print(test_direc)
    CONFIG = {
        'test_dir': test_direc,
        'biodata_file': biodata_direc,
        'model_path': model_path_1,
        'batch_size': 32,
        'architecture': 'resnet50',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("this works" , CONFIG['biodata_file'], type(CONFIG['biodata_file']) )
    biodata_file_path = os.path.abspath(CONFIG['biodata_file'])

    print(f"Loading biodata from: {biodata_file_path}")

    biodata = pd.read_csv(CONFIG['biodata_file'])
    print("this doesnr works")

    full_name_list = biodata['name'].tolist()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = TestDataset(CONFIG['test_dir'], test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    model = load_model(CONFIG['model_path'], len(full_name_list), CONFIG['architecture'], CONFIG['device'])

    predictions = evaluate_model(model, test_loader, CONFIG['device'], full_name_list)

    print("Working unitll here ------------")

    unique_recognized_count = update_attendance(predictions, CONFIG['biodata_file'], sub, unique_recognized_count)

    print("------------ Working unitll here ?")

    print(f"Evaluation complete. Attendance updated in {CONFIG['biodata_file']}")

    print("Total Present students : ", unique_recognized_count)

    return(unique_recognized_count)

def mytry():
    print("djuisegadiu")

# if __name__ == "__main__":
#     sub = sys.argv[1]
#     class_name = sys.argv[2]
#     mymain(sub, class_name)