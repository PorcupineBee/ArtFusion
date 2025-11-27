#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

# ==================== AdaIN Layer ====================

class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    
    def __init__(self):
        super(AdaIN, self).__init__()
    
    def forward(self, content_feat, style_feat):
        """
        Adaptive Instance Normalization
        
        Args:
            content_feat: Content features (B, C, H, W)
            style_feat: Style features (B, C, H, W)
        
        Returns:
            Normalized features with style statistics
        """
        assert content_feat.size()[:2] == style_feat.size()[:2], \
            "Content and style features must have same batch size and channels"
        
        batch_size, num_channels = content_feat.size()[:2]
        
        # Calculate mean and std for content
        content_mean = content_feat.view(batch_size, num_channels, -1).mean(dim=2)
        content_std = content_feat.view(batch_size, num_channels, -1).std(dim=2)
        
        # Calculate mean and std for style
        style_mean = style_feat.view(batch_size, num_channels, -1).mean(dim=2)
        style_std = style_feat.view(batch_size, num_channels, -1).std(dim=2)
        
        # Normalize content features
        content_mean = content_mean.view(batch_size, num_channels, 1, 1)
        content_std = content_std.view(batch_size, num_channels, 1, 1)
        normalized = (content_feat - content_mean) / (content_std + 1e-5)
        
        # Apply style statistics
        style_mean = style_mean.view(batch_size, num_channels, 1, 1)
        style_std = style_std.view(batch_size, num_channels, 1, 1)
        stylized = normalized * style_std + style_mean
        
        return stylized


# ==================== Encoder ====================

class EfficientNetEncoder(nn.Module):
    """EfficientNet-B3 Encoder for feature extraction"""
    
    def __init__(self, pretrained=True, output_layer='blocks.4'):
        super(EfficientNetEncoder, self).__init__()
        efficientnet = models.efficientnet_b3(pretrained=pretrained)
        
        self.output_layer = output_layer
        
        # Extract features up to specified layer
        self.features = nn.Sequential()
        
        # Add stem
        self.features.add_module('stem', efficientnet.features[0])
        
        # Add blocks up to output_layer
        layer_idx = int(output_layer.split('.')[1])
        for idx in range(1, layer_idx + 1):
            self.features.add_module(f'block_{idx}', efficientnet.features[idx])
        
        # Freeze encoder
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.features(x)


# ==================== Decoder ====================

class Decoder(nn.Module):
    """Decoder network to reconstruct image from features"""
    
    def __init__(self, in_channels=232):  # EfficientNet-B3 block 4 output channels
        super(Decoder, self).__init__()
        
        # Mirror the encoder architecture
        self.decoder = nn.Sequential(
            # Upsample and reduce channels
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        return self.decoder(x)


# ==================== AdaIN Style Transfer Network ====================

class AdaINStyleTransfer(nn.Module):
    """Complete AdaIN-based Style Transfer Network"""
    
    def __init__(self, encoder_layer='blocks.4', decoder_path=None):
        super(AdaINStyleTransfer, self).__init__()
        
        self.encoder = EfficientNetEncoder(pretrained=True, output_layer=encoder_layer)
        self.adain = AdaIN()
        
        # Get encoder output channels
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            encoder_out = self.encoder(dummy_input)
            in_channels = encoder_out.size(1)
        
        self.decoder = Decoder(in_channels=in_channels)
        
        # Load pretrained decoder if available
        if decoder_path and os.path.exists(decoder_path):
            self.decoder.load_state_dict(torch.load(decoder_path))
            print(f"Loaded decoder weights from {decoder_path}")
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, content, style, alpha=1.0):
        """
        Forward pass for style transfer
        
        Args:
            content: Content image (B, 3, H, W)
            style: Style image (B, 3, H, W)
            alpha: Style strength (0.0 to 1.0)
        
        Returns:
            Stylized image
        """
        # Encode
        content_feat = self.encode(content)
        style_feat = self.encode(style)
        
        # AdaIN
        stylized_feat = self.adain(content_feat, style_feat)
        
        # Blend with original content features (alpha control)
        if alpha < 1.0:
            stylized_feat = alpha * stylized_feat + (1 - alpha) * content_feat
        
        # Decode
        output = self.decode(stylized_feat)
        
        return output


# ==================== Training Module ====================

class AdaINTrainer:
    """Trainer for AdaIN network"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
    def content_loss(self, output_feat, target_feat):
        """MSE loss between features"""
        return F.mse_loss(output_feat, target_feat)
    
    def style_loss(self, output_feat, style_feat):
        """MSE loss between mean and std"""
        batch_size, num_channels = output_feat.size()[:2]
        
        # Output statistics
        output_mean = output_feat.view(batch_size, num_channels, -1).mean(dim=2)
        output_std = output_feat.view(batch_size, num_channels, -1).std(dim=2)
        
        # Style statistics
        style_mean = style_feat.view(batch_size, num_channels, -1).mean(dim=2)
        style_std = style_feat.view(batch_size, num_channels, -1).std(dim=2)
        
        loss_mean = F.mse_loss(output_mean, style_mean)
        loss_std = F.mse_loss(output_std, style_std)
        
        return loss_mean + loss_std
    
    def train_step(self, content_images, style_images, optimizer, 
                   content_weight=1.0, style_weight=10.0):
        """Single training step"""
        self.model.train()
        
        content_images = content_images.to(self.device)
        style_images = style_images.to(self.device)
        
        # Forward pass
        stylized_images = self.model(content_images, style_images)
        
        # Encode outputs and targets
        stylized_feat = self.model.encode(stylized_images)
        content_feat = self.model.encode(content_images)
        style_feat = self.model.encode(style_images)
        
        # Calculate losses
        loss_c = self.content_loss(stylized_feat, content_feat)
        loss_s = self.style_loss(stylized_feat, style_feat)
        
        total_loss = content_weight * loss_c + style_weight * loss_s
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'content_loss': loss_c.item(),
            'style_loss': loss_s.item()
        }


# ==================== Inference Module ====================

class AdaINInference:
    """Inference module for trained AdaIN network"""
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = AdaINStyleTransfer(decoder_path=model_path).to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_image(self, img_path, size=512):
        """Load and preprocess image"""
        image = Image.open(img_path).convert('RGB')
        
        # Resize
        if isinstance(size, int):
            image = transforms.Resize(size)(image)
        
        # Transform
        image = self.transform(image).unsqueeze(0)
        return image.to(self.device)
    
    def deprocess_image(self, tensor):
        """Convert tensor back to PIL image"""
        image = tensor.clone().detach().cpu().squeeze(0)
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        
        # Clip and convert
        image = torch.clamp(image, 0, 1)
        return transforms.ToPILImage()(image)
    
    def transfer(self, content_path, style_path, alpha=1.0, output_size=None):
        """
        Perform style transfer
        
        Args:
            content_path: Path to content image
            style_path: Path to style image
            alpha: Style strength (0.0 to 1.0)
            output_size: Output image size (default: same as content)
        
        Returns:
            Stylized PIL image
        """
        # Load images
        content_img = self.load_image(content_path, size=output_size or 512)
        style_img = self.load_image(style_path, size=output_size or 512)
        
        # Style transfer
        with torch.no_grad():
            output = self.model(content_img, style_img, alpha=alpha)
        
        # Deprocess
        return self.deprocess_image(output)


# ==================== Usage Example ====================

# if __name__ == "__main__":

#%%
print("="*60)
print("AdaIN Style Transfer with EfficientNet-B3")
print("="*60)

# ===== INFERENCE MODE (Pretrained/Untrained) =====
print("\n1. INFERENCE MODE - Style Transfer")
print("-"*60)

# Initialize inference
adain_inference = AdaINInference(model_path=None)  # Use trained decoder if available

# Perform style transfer
print("Note: Without training, the decoder produces random outputs.")
print("You need to train the decoder first for good results.\n")

# Uncomment to run inference:

result = adain_inference.transfer(
    content_path='/mnt/d/gitclones/nst/NST-ArtFusion/data/content-images/c3.jpg',
    style_path='/mnt/d/gitclones/nst/NST-ArtFusion/data/style-images/s3.jpg',
    alpha=1.0,  # Style strength (0.0 to 1.0)
    output_size=512
)
result.save('adain_output_c3_s3.jpg')
print("Stylized image saved to 'adain_output.jpg'")
#%%
# ===== TRAINING MODE =====
print("\n2. TRAINING MODE - Train Decoder")
print("-"*60)
print("To train the decoder, you need:")
print("- Content images (e.g., MS-COCO dataset)")
print("- Style images (e.g., WikiArt dataset)")
print("\nTraining loop example:")


# Example training code:
from torch.utils.data import DataLoader

# Initialize model and trainer
model = AdaINStyleTransfer()
trainer = AdaINTrainer(model)
optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)

# Create dataloaders (you need to implement these)
content_loader = DataLoader(content_dataset, batch_size=8, shuffle=True)
style_loader = DataLoader(style_dataset, batch_size=8, shuffle=True)

# Training loop
for epoch in range(10):
    for content_batch, style_batch in zip(content_loader, style_loader):
        losses = trainer.train_step(
            content_batch, 
            style_batch, 
            optimizer,
            content_weight=1.0,
            style_weight=10.0
        )
        print(f"Epoch {epoch}, Loss: {losses['total_loss']:.4f}")
    
    # Save decoder
    torch.save(model.decoder.state_dict(), f'decoder_epoch_{epoch}.pth')


# ===== COMPARISON WITH OPTIMIZATION-BASED NST =====
print("\n3. COMPARISON")
print("-"*60)
print("AdaIN vs Gatys et al.:")
print("  ✓ AdaIN: Real-time (~0.02s per image)")
print("  ✓ Gatys: High quality but slow (~minutes per image)")
print("\nAdaIN Advantages:")
print("  - Fast inference (single forward pass)")
print("  - Can process videos in real-time")
print("  - Good for interactive applications")
print("\nAdaIN Requirements:")
print("  - Needs decoder training (~10 epochs on COCO+WikiArt)")
print("  - Requires large training dataset")
print("  - Slightly lower quality than optimization-based")

# ===== KEY PARAMETERS =====
print("\n4. KEY PARAMETERS")
print("-"*60)
print("alpha: Style strength")
print("  - 0.0: Pure content (no style)")
print("  - 0.5: Balanced")
print("  - 1.0: Full style transfer")
print("\nEncoder layer: Feature extraction depth")
print("  - 'blocks.3': Earlier features (more structural)")
print("  - 'blocks.4': Mid-level features (balanced)")
print("  - 'blocks.5': Deeper features (more abstract)")

print("\n" + "="*60)
print("Setup complete! Train the decoder to use AdaIN style transfer.")
print("="*60)