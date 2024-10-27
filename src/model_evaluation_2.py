import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import warnings

warnings.filterwarnings("ignore")

# Import the necessary classes from your training script
from model_development import LFWDataset, FaceRecognitionModel

class TestDataset(Dataset):
    def __init__(self, data: pd.DataFrame, lfw_dir: str, transform: transforms.Compose, full_name_list: List[str]):
        self.data = data
        self.lfw_dir = lfw_dir
        self.transform = transform
        self.image_paths = self._create_image_paths()
        self.full_name_list = full_name_list
        self.name_to_idx = {name: idx for idx, name in enumerate(full_name_list)}

    def _create_image_paths(self) -> List[Tuple[str, str]]:
        image_paths = []
        for _, row in self.data.iterrows():
            name = row['name']
            person_dir = os.path.join(self.lfw_dir, name)
            if os.path.exists(person_dir):
                for img_name in os.listdir(person_dir):
                    if img_name.endswith('.jpg'):
                        image_paths.append((os.path.join(person_dir, img_name), name))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, name = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.name_to_idx[name]

def load_model(model_path: str, num_classes: int, architecture: str, device: str):
    model = FaceRecognitionModel(num_classes=num_classes, architecture=architecture)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str, full_name_list: List[str]):
    all_predictions = []
    all_labels = []
    image_counts = {}

    with torch.no_grad():
        for images, label_indices in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            predicted = predicted.cpu().tolist()
            label_indices = label_indices.cpu().tolist()
            
            predicted_names = [full_name_list[p] for p in predicted]
            true_names = [full_name_list[l] for l in label_indices]
            
            all_predictions.extend(predicted_names)
            all_labels.extend(true_names)
            
            for name in true_names:
                image_counts[name] = image_counts.get(name, 0) + 1

    return all_predictions, all_labels, image_counts

def calculate_metrics(true_labels: List[str], predicted_labels: List[str]):
    accuracy = accuracy_score(true_labels, predicted_labels)
    error_rate = 1 - accuracy
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    return {
        'Accuracy': accuracy,
        'Error Rate': error_rate,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

def plot_confusion_matrix(true_labels: List[str], predicted_labels: List[str], save_path: str):
    # Get unique labels
    unique_labels = sorted(set(true_labels + predicted_labels))
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_per_person(true_labels: List[str], predicted_labels: List[str], save_path: str):
    df = pd.DataFrame({'True': true_labels, 'Predicted': predicted_labels})
    accuracy_per_person = df.groupby('True').apply(lambda x: (x['True'] == x['Predicted']).mean())
    
    plt.figure(figsize=(12, 6))
    accuracy_per_person.sort_values().plot(kind='bar')
    plt.title('Accuracy per Person')
    plt.xlabel('Person')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Configuration
    CONFIG = {
        'train_file': 'Minor Project/Facial Recognition Attendance Sysytem/Dataset/peopleDevTrain.csv',
        'lfw_dir': 'Minor Project/Facial Recognition Attendance Sysytem/Dataset/lfw_deepfunneled',
        'model_path': 'Minor Project/Facial Recognition Attendance Sysytem/models/face_recognition_model.pth',
        'results_dir': 'Minor Project/Facial Recognition Attendance Sysytem/results',
        'num_test_people': 100,
        'batch_size': 32,
        'architecture': 'resnet50',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Create results directory
    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    # Load and prepare data
    full_data = pd.read_csv(CONFIG['train_file'])
    unique_names = full_data['name'].unique().tolist()
    test_names = random.sample(unique_names, CONFIG['num_test_people'])
    test_data = full_data[full_data['name'].isin(test_names)]

    # Create test dataset and loader
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_dataset = TestDataset(test_data, CONFIG['lfw_dir'], test_transform, unique_names)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Print some information about the dataset and model
    print(f"Number of unique names in full dataset: {len(unique_names)}")
    print(f"Number of test names: {len(test_names)}")
    print(f"Number of images in test dataset: {len(test_dataset)}")
    print(f"Model architecture: {CONFIG['architecture']}")

    # Load model
    model = load_model(CONFIG['model_path'], len(unique_names), CONFIG['architecture'], CONFIG['device'])

    # Evaluate model
    predicted_labels, true_labels, image_counts = evaluate_model(model, test_loader, CONFIG['device'], unique_names)

    # Calculate metrics
    metrics = calculate_metrics(true_labels, predicted_labels)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'Actual Name': true_labels,
        'Predicted Name': predicted_labels,
        'Images Used': [image_counts[name] for name in true_labels],
        'Correct': [true == pred for true, pred in zip(true_labels, predicted_labels)]
    })

    # Save results to CSV
    results_df.to_csv(os.path.join(CONFIG['results_dir'], 'evaluation_results.csv'), index=False)

    # Print metrics
    print("\nModel Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot and save confusion matrix
    plot_confusion_matrix(true_labels, predicted_labels, 
                          os.path.join(CONFIG['results_dir'], 'confusion_matrix.png'))

    # Plot and save accuracy per person
    plot_accuracy_per_person(true_labels, predicted_labels, 
                             os.path.join(CONFIG['results_dir'], 'accuracy_per_person.png'))

if __name__ == "__main__":
    main()