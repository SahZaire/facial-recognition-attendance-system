import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import os
from tqdm import tqdm
import random

from model_development import LFWDataset, FaceRecognitionModel

class UniquePersonDataset(Dataset):
    def __init__(self, full_dataset, selected_people):
        self.full_dataset = full_dataset
        self.selected_indices = []
        for person in selected_people:
            person_indices = [i for i, (_, label) in enumerate(full_dataset.image_paths) 
                              if full_dataset.label_encoder.inverse_transform([label])[0] == person]
            self.selected_indices.append(random.choice(person_indices))
        
    def __len__(self):
        return len(self.selected_indices)
    
    def __getitem__(self, idx):
        return self.full_dataset[self.selected_indices[idx]]

def load_model(model_path, architecture, num_classes):
    model = FaceRecognitionModel(num_classes=num_classes, architecture=architecture)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model, test_loader, device, label_encoder):
    model.eval()
    all_preds = []
    all_labels = []
    all_names = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_names.extend(label_encoder.inverse_transform(labels.numpy()))
    
    return all_preds, all_labels, all_names

def save_results(predictions, true_labels, names, output_path):
    df = pd.DataFrame({
        'Predicted': [names[i] for i in predictions],
        'Actual': [names[i] for i in true_labels],
        'Correct': (np.array(predictions) == np.array(true_labels))
    })
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def plot_confusion_matrix(true_labels, predictions, class_names, output_path):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Configuration
    CONFIG = {
        'train_file': 'Minor Project/Facial Recognition Attendance Sysytem/Dataset/peopleDevTrain.csv',
        'lfw_dir': 'Minor Project/Facial Recognition Attendance Sysytem/Dataset/lfw_deepfunneled',
        'model_path': 'Minor Project/Facial Recognition Attendance Sysytem/models/face_recognition_model.pth',
        'results_dir': 'Minor Project/Facial Recognition Attendance Sysytem/results',
        'batch_size': 60,  # Changed to 60 to process all images at once
        'num_test_people': 60,
        'architecture': 'resnet50',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Create results directory
    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    # Load the full dataset
    full_dataset = LFWDataset(CONFIG['train_file'], CONFIG['lfw_dir'], is_training=False)

    # Randomly select 60 unique people
    unique_people = list(set(full_dataset.data['name']))
    test_people = random.sample(unique_people, CONFIG['num_test_people'])

    # Create a dataset with one image per selected person
    test_dataset = UniquePersonDataset(full_dataset, test_people)

    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Load the model
    model = load_model(CONFIG['model_path'], CONFIG['architecture'], full_dataset.num_classes)
    model = model.to(CONFIG['device'])

    # Evaluate the model
    predictions, true_labels, names = evaluate_model(model, test_loader, CONFIG['device'], full_dataset.label_encoder)

    # Save results
    save_results(predictions, true_labels, names, 
                 os.path.join(CONFIG['results_dir'], 'evaluation_results.csv'))

    # Generate and save visualizations
    test_class_names = sorted(set(names))
    plot_confusion_matrix(true_labels, predictions, test_class_names, 
                          os.path.join(CONFIG['results_dir'], 'confusion_matrix.png'))

    # Print classification report and additional metrics
    print(classification_report(true_labels, predictions, target_names=test_class_names))

    # Calculate and print additional metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()