import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class LFWDataset(Dataset):
    def __init__(self, 
                 train_file: str, 
                 lfw_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 is_training: bool = True):
        """
        Args:
            train_file: Path to peopleDevTrain.csv
            lfw_dir: Path to lfw_deepfunneled directory
            transform: Optional transform to be applied to images
            is_training: Whether this is for training (enables augmentation)
        """
        self.data = pd.read_csv(train_file)
        self.lfw_dir = lfw_dir
        self.is_training = is_training
        self.transform = transform or self._get_transforms()
        
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data['name'])
        self.num_classes = len(self.label_encoder.classes_)
        
        self.image_paths = self._create_image_paths()
    
    def _get_transforms(self):
        """Enhanced transformations with augmentation for training"""
        if self.is_training:
            return transforms.Compose([
                transforms.Resize((256, 256)),  # Larger initial size for random cropping
                transforms.RandomCrop(224),     # Random crop for variation
                transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
                transforms.RandomApply([        # Color jittering
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.1
                    )
                ], p=0.3),
                transforms.RandomApply([        # Gaussian blur for robustness
                    transforms.GaussianBlur(kernel_size=3)
                ], p=0.1),
                transforms.RandomAffine(        # Random rotation and translation
                    degrees=15, 
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    # # For Inception_v3 model
    # def _get_transforms(self):
    #     """Enhanced transformations with augmentation for training, optimized for Inception v3"""
    #     if self.is_training:
    #         return transforms.Compose([
    #             transforms.Resize((320, 320)),  # Larger initial size for random cropping
    #             transforms.RandomCrop(299),     # Random crop to Inception v3 input size
    #             transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
    #             transforms.RandomApply([        # Color jittering
    #                 transforms.ColorJitter(
    #                     brightness=0.2,
    #                     contrast=0.2,
    #                     saturation=0.2,
    #                     hue=0.1
    #                 )
    #             ], p=0.3),
    #             transforms.RandomApply([        # Gaussian blur for robustness
    #                 transforms.GaussianBlur(kernel_size=3)
    #             ], p=0.1),
    #             transforms.RandomAffine(        # Random rotation and translation
    #                 degrees=15, 
    #                 translate=(0.1, 0.1),
    #                 scale=(0.9, 1.1)
    #             ),
    #             transforms.ToTensor(),
    #             transforms.Normalize(
    #                 mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225]
    #             )
    #         ])
    #     else:
    #         return transforms.Compose([
    #             transforms.Resize((299, 299)),  # Resize to Inception v3 input size
    #             transforms.ToTensor(),
    #             transforms.Normalize(
    #                 mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225]
    #             )
    #         ])

    def _create_image_paths(self) -> List[Tuple[str, int]]:
        """Create list of (image_path, label) tuples"""
        image_paths = []
        for idx, row in self.data.iterrows():
            name = row['name']
            person_dir = os.path.join(self.lfw_dir, name)
            if os.path.exists(person_dir):
                for img_name in os.listdir(person_dir):
                    if img_name.endswith('.jpg'):
                        image_paths.append((
                            os.path.join(person_dir, img_name),
                            self.labels[idx]
                        ))
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer"""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FaceRecognitionModel(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 architecture: str = 'resnet50',
                 pretrained: bool = True):
        super(FaceRecognitionModel, self).__init__()
        
        self.architecture = architecture
        
        if architecture == 'vgg16':
            self.base = models.vgg16(pretrained=pretrained)
            num_ftrs = self.base.classifier[6].in_features
            self.base.classifier[6] = nn.Linear(num_ftrs, num_classes)
            
        elif architecture == 'resnet50':
            self.base = models.resnet50(pretrained=pretrained)
            num_ftrs = self.base.fc.in_features
            self.base.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        elif architecture == 'senet50':
            self.base = models.resnet50(pretrained=pretrained)
            # Add SE layers to ResNet50
            for m in self.base.modules():
                if isinstance(m, models.resnet.Bottleneck):
                    m.se = SELayer(m.conv3.out_channels)
            num_ftrs = self.base.fc.in_features
            self.base.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        
        elif architecture == 'inception_v3':
            self.base = models.inception_v3(pretrained=pretrained)
            # Handle auxiliary outputs
            self.base.aux_logits = False
            # Replace final fully connected layer
            num_ftrs = self.base.fc.in_features
            self.base.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        return self.base(x)

class ModelTrainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 learning_rate: float = 0.0001,
                 device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Use Label Smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Use AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Cosine Annealing scheduler with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Reset every 10 epochs
            T_mult=2  # Double the reset interval after each restart
        )
    
    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                val_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = val_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, 
              num_epochs: int,
              save_path):
        """Train the model"""
        best_val_loss = float('inf')
        
        # Create the models directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.train_epoch()
            self.scheduler.step()
            
            if self.val_loader:
                val_loss, val_accuracy = self.validate()
                
                print(f'Epoch: {epoch+1}/{num_epochs}')
                print(f'Train Loss: {train_loss:.4f}')
                print(f'Train Accuracy: {train_accuracy:.4f}')
                print(f'Val Loss: {val_loss:.4f}')
                print(f'Val Accuracy: {val_accuracy:.4f}')
                print(f'Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}')
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': best_val_loss,
                    }, save_path)
            else:
                print(f'Epoch: {epoch+1}/{num_epochs}')
                print(f'Train Loss: {train_loss:.4f}')
                print(f'Train Accuracy: {train_accuracy:.4f}')
                print(f'Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}')

def model_development(class_name:str):
    test_direc = f"C:\\A Drive\\Machine Learning\\Minor Project\\Facial Recognition Attendance Sysytem\\final_dataset\\{class_name}\\present"
    lfw_direc = f"C:\\A Drive\\Machine Learning\\Minor Project\\Facial Recognition Attendance Sysytem\\final_dataset\\{class_name}\\listed_100"
    train_direc = f"C:\\A Drive\\Machine Learning\\Minor Project\\Facial Recognition Attendance Sysytem\\final_dataset\\{class_name}\\listed_100_names.csv"
    biodata_direc = f"C:\\A Drive\\Machine Learning\\Minor Project\\Facial Recognition Attendance Sysytem\\final_dataset\\{class_name}\\listed_100_biodata.csv"
    model_save_path = f"C:\\A Drive\\Machine Learning\\Minor Project\\Facial Recognition Attendance Sysytem\\models\\{class_name}\\resnet50_face_recognition_model.pth"
    
    print("Starting Model Deployment")

    # Configuration
    CONFIG = {
        'train_file': train_direc,
        'test_file': test_direc,
        'lfw_dir': lfw_direc,
        'batch_size': 16,  # Increased batch size
        'num_epochs': 5,  # Increased number of epochs
        'learning_rate': 0.0001,
        'architecture': 'resnet50',  # Changed default to ResNet50
        'device': 'cuda'
    }
    
    # Create datasets with different transforms for train and val
    train_dataset = LFWDataset(CONFIG['train_file'], CONFIG['lfw_dir'], is_training=True)
    val_dataset = LFWDataset(CONFIG['train_file'], CONFIG['lfw_dir'], is_training=False)
    print("Any error ?")
    # Split dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True  # Added for faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = FaceRecognitionModel(
        num_classes=train_dataset.dataset.num_classes,
        architecture=CONFIG['architecture']
    )
    print("ulitmate goal")
    # Create trainer and train
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=CONFIG['learning_rate'],
        device=CONFIG['device']
    )
    
    trainer.train(CONFIG['num_epochs'], save_path=model_save_path)

    print("Finished Model Deployment")

# if __name__ == "__main__":
#     main()