#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm

# ==================== PART 1: WikiArt Fine-tuning ====================

class WikiArtDataset(Dataset):
    """
    WikiArt Dataset Loader for HuggingFace datasets
    Works with: ds = load_dataset("huggan/wikiart")
    """
    def __init__(self, hf_dataset, transform=None, target='style'):
        """
        Args:
            hf_dataset: HuggingFace dataset split (e.g., ds['train'])
            transform: torchvision transforms
            target: which label to use - 'style', 'artist', or 'genre'
        """
        self.dataset = hf_dataset
        self.transform = transform
        self.target = target
        
        # Get number of classes based on target
        if target == 'style':
            self.num_classes = 27
            self.class_names = ["Abstract_Expressionism", "Action_painting", "Analytical_Cubism", 
                               "Art_Nouveau", "Baroque", "Color_Field_Painting", "Contemporary_Realism", 
                               "Cubism", "Early_Renaissance", "Expressionism", "Fauvism", "High_Renaissance", 
                               "Impressionism", "Mannerism_Late_Renaissance", "Minimalism", 
                               "Naive_Art_Primitivism", "New_Realism", "Northern_Renaissance", 
                               "Pointillism", "Pop_Art", "Post_Impressionism", "Realism", "Rococo", 
                               "Romanticism", "Symbolism", "Synthetic_Cubism", "Ukiyo_e"]
        elif target == 'artist':
            self.num_classes = 129
            self.class_names = None  # Too many to list here
        elif target == 'genre':
            self.num_classes = 11
            self.class_names = ["abstract_painting", "cityscape", "genre_painting", "illustration", 
                               "landscape", "nude_painting", "portrait", "religious_painting", 
                               "sketch_and_study", "still_life", "Unknown Genre"]
        else:
            raise ValueError(f"target must be 'style', 'artist', or 'genre', got {target}")
        
        print(f"Loaded {len(self.dataset)} images for {target} classification ({self.num_classes} classes)")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get PIL image
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        
        # Get label based on target
        label = item[self.target]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, label


class EfficientNetFineTuner:
    """Fine-tune EfficientNet-B3 on WikiArt dataset"""
    
    def __init__(self, num_classes, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 freeze_backbone=False):
        self.device = device
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.model = self._build_model()
        
    def _build_model(self):
        """Load EfficientNet-B3 and modify classifier"""
        model = models.efficientnet_b3(pretrained=True)
        
        # Optionally freeze backbone for faster training
        if self.freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
            print("Backbone frozen - only training classifier")
        
        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, self.num_classes)
        )
        
        return model.to(self.device)
    
    def get_transforms(self, minimal_augmentation=True):
        """
        Data augmentation and normalization
        
        Args:
            minimal_augmentation: If True, use minimal augmentation for initial training
                                 If False, use more aggressive augmentation
        """
        if minimal_augmentation:
            # Minimal augmentation for initial fine-tuning
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # More aggressive augmentation
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def train(self, train_loader, val_loader, epochs=10, lr=0.001):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=2)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'loss': train_loss/len(train_loader), 
                                 'acc': 100.*train_correct/train_total})
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            print(f'\nEpoch {epoch+1}: Train Acc: {100.*train_correct/train_total:.2f}%, '
                  f'Val Acc: {val_acc:.2f}%, Val Loss: {avg_val_loss:.4f}')
            
            scheduler.step(avg_val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'efficientnet_b3_wikiart_best.pth')
                print(f'Saved best model with Val Acc: {val_acc:.2f}%')
        
        print(f'\nTraining complete. Best Val Acc: {best_val_acc:.2f}%')
        return self.model


# ==================== PART 2: NST with Fine-tuned Model ====================

class FineTunedStyleTransfer:
    """Neural Style Transfer using fine-tuned EfficientNet-B3"""
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self._load_model(model_path)
        
        # Feature extraction layers
        self.content_layers = ['blocks.4']
        self.style_layers = ['blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5']
        
        # Loss weights
        self.content_weight = 1e0
        self.style_weight = 1e6
        
    def _load_model(self, model_path):
        """Load fine-tuned or pretrained model"""
        model = models.efficientnet_b3(pretrained=False)
        
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            # Load state dict, but only the features part
            state_dict = torch.load(model_path, map_location=self.device)
            # Filter out classifier weights
            feature_state_dict = {k: v for k, v in state_dict.items() 
                                 if k.startswith('features')}
            model.load_state_dict(feature_state_dict, strict=False)
        else:
            print("Using pretrained ImageNet weights")
            model = models.efficientnet_b3(pretrained=True)
        
        model = model.to(self.device)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
            
        return model
    
    def _get_features(self, image, layers):
        """Extract features from specified layers"""
        features = {}
        x = image
        
        x = self.model.features[0](x)
        
        for idx, block in enumerate(self.model.features[1:], 1):
            x = block(x)
            layer_name = f'blocks.{idx}'
            if layer_name in layers:
                features[layer_name] = x
                
        return features
    
    def gram_matrix(self, tensor):
        """Compute Gram matrix for style representation"""
        b, c, h, w = tensor.size()
        features = tensor.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        return gram.div(b * c * h * w)
    
    def content_loss(self, target_features, content_features):
        """Compute content loss"""
        loss = 0
        for layer in self.content_layers:
            loss += torch.mean((target_features[layer] - content_features[layer]) ** 2)
        return loss
    
    def style_loss(self, target_features, style_features):
        """Compute style loss"""
        loss = 0
        for layer in self.style_layers:
            target_gram = self.gram_matrix(target_features[layer])
            style_gram = self.gram_matrix(style_features[layer])
            loss += torch.mean((target_gram - style_gram) ** 2)
        return loss
    
    def load_image(self, img_path, max_size=512):
        """Load and preprocess image"""
        image = Image.open(img_path).convert('RGB')
        
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)
            
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = transform(image).unsqueeze(0)
        return image.to(self.device)
    
    def deprocess_image(self, tensor):
        """Convert tensor back to image"""
        image = tensor.clone().detach().cpu()
        image = image.squeeze(0)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        
        transform = transforms.ToPILImage()
        return transform(image)
    
    def transfer(self, content_img_path, style_img_path, num_steps=500, 
                 learning_rate=0.01, content_weight=None, style_weight=None):
        """Perform neural style transfer"""
        if content_weight is not None:
            self.content_weight = content_weight
        if style_weight is not None:
            self.style_weight = style_weight
            
        content_img = self.load_image(content_img_path)
        style_img = self.load_image(style_img_path)
        target_img = content_img.clone().requires_grad_(True)
        
        content_features = self._get_features(content_img, self.content_layers)
        style_features = self._get_features(style_img, self.style_layers)
        
        optimizer = optim.Adam([target_img], lr=learning_rate)
        
        print("Starting style transfer with fine-tuned model...")
        for step in range(num_steps):
            target_features = self._get_features(target_img, 
                                                self.content_layers + self.style_layers)
            
            c_loss = self.content_loss(target_features, content_features)
            s_loss = self.style_loss(target_features, style_features)
            total_loss = self.content_weight * c_loss + self.style_weight * s_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if (step + 1) % 50 == 0:
                print(f"Step [{step+1}/{num_steps}], "
                      f"Content Loss: {c_loss.item():.4f}, "
                      f"Style Loss: {s_loss.item():.4f}, "
                      f"Total Loss: {total_loss.item():.4f}")
        
        print("Style transfer complete!")
        return self.deprocess_image(target_img)


# ==================== USAGE EXAMPLE ====================
#%%
from datasets import load_dataset

# Load WikiArt dataset from HuggingFace
print("Loading WikiArt dataset from HuggingFace...")
ds = load_dataset("huggan/wikiart")

print(f"Total samples in train split: {len(ds['train'])}")

# STEP 1: Split train into train/val (90/10 split)
print("\n" + "="*50)
print("Creating Train/Val Split")
print("="*50)

# Split the train dataset into train and validation
train_val_split = ds['train'].train_test_split(test_size=0.1, seed=42)
train_split = train_val_split['train']
val_split = train_val_split['test']

print(f"Training samples: {len(train_split)}")
print(f"Validation samples: {len(val_split)}")

# STEP 2: Fine-tune on WikiArt
print("\n" + "="*50)
print("STEP 2: Fine-tuning EfficientNet-B3 on WikiArt")
print("="*50)

# Choose what to predict: 'style', 'artist', or 'genre'
target = 'style'  # Change to 'artist' or 'genre' if desired

# Create datasets with minimal augmentation for now
train_transform, val_transform = EfficientNetFineTuner(num_classes=27).get_transforms(
    minimal_augmentation=True  # Set to False later for more augmentation
)

train_dataset = WikiArtDataset(train_split, transform=train_transform, target=target)
val_dataset = WikiArtDataset(val_split, transform=val_transform, target=target)

print(f"Number of {target} classes: {train_dataset.num_classes}")

# Create dataloaders
# Adjust batch_size based on your GPU memory (32, 64, or 128)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=4, pin_memory=True)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Initialize fine-tuner
# Set freeze_backbone=True for faster training (only trains classifier)
# Set freeze_backbone=False for full fine-tuning (better results, slower)
finetuner = EfficientNetFineTuner(
    num_classes=train_dataset.num_classes,
    freeze_backbone=False  # Change to True for faster training
)

# Train
print("\nStarting training...")
print("Note: With ~73k training samples, each epoch will take some time.")
print("Consider using a GPU for faster training.\n")

finetuned_model = finetuner.train(
    train_loader, 
    val_loader, 
    epochs=10,      # Start with 10 epochs
    lr=0.0001       # Lower learning rate for full fine-tuning
)

print("\nFine-tuning complete! Model saved as 'efficientnet_b3_wikiart_best.pth'")

#%%
# STEP 3: Use fine-tuned model for NST
print("\n" + "="*50)
print("STEP 3: Neural Style Transfer with Fine-tuned Model")
print("="*50)

# Load fine-tuned model for NST
nst = FineTunedStyleTransfer(model_path='efficientnet_b3_wikiart_best.pth')

# Perform style transfer
# Make sure you have content.jpg and style.jpg in your directory
result = nst.transfer(
    content_img_path='/mnt/d/gitclones/nst/NST-ArtFusion/data/content-images/c6.jpg',
    style_img_path=  '/mnt/d/gitclones/nst/NST-ArtFusion/data/style-images/s3.jpg',
    num_steps=500,
    learning_rate=0.01
)

result.save('stylized_finetuned_output.jpg')
print("\nStyle transfer complete! Saved to 'stylized_finetuned_output.jpg'")