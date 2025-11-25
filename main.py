import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import numpy as np

class ImprovedStyleTransfer:
    """
    Neural Style Transfer with patch reduction techniques
    """
    
    def __init__(self, model_type='efficientnet', 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 block_idx=4,
                 style_indices=None):
        self.device = device
        self.model_type = model_type
        self.model = self._get_model()
        
        # Layer configuration
        if model_type == 'vgg19':
            self.content_layers = ['conv4_2']
            self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        elif model_type == 'efficientnet':
            self.content_layers = [f'blocks.{block_idx}']
            self.style_layers = ['blocks.1', 'blocks.2' , 'blocks.3' , 'blocks.4' , 'blocks.5']
        
        # Loss weights (CRITICAL for reducing patches)
        self.content_weight = 1e0
        self.style_weight = 1e5  # Reduced from 1e6
        self.tv_weight = 1e-3    # Total Variation for smoothness
        
    def _get_model(self):
        """Load pretrained model"""
        if self.model_type == 'vgg19':
            vgg = models.vgg19(pretrained=True).features
            model = vgg.to(self.device).eval()
        elif self.model_type == 'efficientnet':
            efficientnet = models.efficientnet_b7(pretrained=True)
            model = efficientnet.to(self.device).eval()
        # elif self.model_type == 'efficientnet':
        #     efficientnet = models.efficientnet_b5(pretrained=True)
        #     model = efficientnet.to(self.device).eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _get_features(self, image, layers):
        """Extract features from specified layers"""
        features = {}
        
        if self.model_type == 'vgg19':
            x = image
            layer_names = {
                '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
                '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'
            }
            for name, layer in self.model._modules.items():
                x = layer(x)
                if name in layer_names and layer_names[name] in layers:
                    features[layer_names[name]] = x
                    
        elif self.model_type == 'efficientnet':
            x = image
            x = self.model.features[0](x)
            for idx, block in enumerate(self.model.features[1:], 1):
                x = block(x)
                layer_name = f'blocks.{idx}'
                if layer_name in layers:
                    features[layer_name] = x
        
        return features
    
    def gram_matrix(self, tensor):
        """Compute Gram matrix"""
        b, c, h, w = tensor.size()
        features = tensor.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        return gram.div(b * c * h * w)
    
    def content_loss(self, target_features, content_features):
        """Content loss"""
        loss = 0
        for layer in self.content_layers:
            loss += F.mse_loss(target_features[layer], content_features[layer])
        return loss
    
    def style_loss(self, target_features, style_features):
        """Style loss with weighted layers"""
        loss = 0
        
        # Weight earlier layers more to reduce patchiness
        layer_weights = {
            self.style_layers[0]: 1.0,  # Early layer - more weight
            self.style_layers[1]: 0.8,
            self.style_layers[2]: 0.6,
            self.style_layers[3]: 0.4,
            self.style_layers[4]: 0.2,  # Deep layer - less weight
        }
        
        for layer in self.style_layers:
            target_gram = self.gram_matrix(target_features[layer])
            style_gram = self.gram_matrix(style_features[layer])
            
            weight = layer_weights.get(layer, 1.0)
            loss += weight * F.mse_loss(target_gram, style_gram)
        
        return loss
    
    def total_variation_loss(self, img):
        """
        Total Variation Loss for smoothness
        Penalizes large differences between adjacent pixels
        """
        # Horizontal variation
        tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).sum()
        
        # Vertical variation
        tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).sum()
        
        return tv_h + tv_w
    
    def color_preservation_loss(self, output, content):
        """
        Preserve color distribution from content image
        Helps reduce color patches
        """
        # Match color statistics
        output_mean = output.mean(dim=[2, 3], keepdim=True)
        content_mean = content.mean(dim=[2, 3], keepdim=True)
        
        output_std = output.std(dim=[2, 3], keepdim=True)
        content_std = content.std(dim=[2, 3], keepdim=True)
        
        return F.mse_loss(output_mean, content_mean) + F.mse_loss(output_std, content_std)
    
    def load_image(self, img_path, max_size=512):
        """Load and preprocess image"""
        image = Image.open(img_path).convert('RGB')
        
        # Resize
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
        """Convert tensor to PIL image"""
        image = tensor.clone().detach().cpu().squeeze(0)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        
        return transforms.ToPILImage()(image)
    
    def transfer(self, content_path, style_path, num_steps=500, 
                 learning_rate=0.01, tv_weight=None, style_weight=None,
                 use_l_bfgs=False):
        """
        Improved style transfer with smoothness
        """
        # Update weights if provided
        if tv_weight is not None:
            self.tv_weight = tv_weight
        if style_weight is not None:
            self.style_weight = style_weight
        
        # Load images
        content_img = self.load_image(content_path)
        style_img = self.load_image(style_path)
        
        # Initialize target (IMPORTANT: Use content + noise for better convergence)
        target_img = content_img.clone()
        noise = torch.randn_like(target_img) * 0.1
        target_img = target_img + noise
        target_img.requires_grad_(True)
        
        # Extract features
        content_features = self._get_features(content_img, self.content_layers)
        style_features = self._get_features(style_img, self.style_layers)
        
        # Choose optimizer
        if use_l_bfgs:
            optimizer = optim.LBFGS([target_img], lr=1, max_iter=20)
            num_steps = num_steps // 20  # L-BFGS uses multiple iterations per step
        else:
            optimizer = optim.Adam([target_img], lr=learning_rate)
        
        print(f"Starting improved style transfer ({self.model_type})...")
        print(f"Content weight: {self.content_weight}")
        print(f"Style weight: {self.style_weight}")
        print(f"TV weight: {self.tv_weight}")
        
        for step in range(num_steps):
            def closure():
                # Clamp target to valid range
                with torch.no_grad():
                    target_img.clamp_(-2.5, 2.5)  # Prevent extreme values
                
                optimizer.zero_grad()
                
                # Extract target features
                target_features = self._get_features(
                    target_img, 
                    self.content_layers + self.style_layers
                )
                
                # Calculate losses
                c_loss = self.content_loss(target_features, content_features)
                s_loss = self.style_loss(target_features, style_features)
                tv_loss = self.total_variation_loss(target_img)
                
                # Total loss
                total_loss = (self.content_weight * c_loss + 
                            self.style_weight * s_loss + 
                            self.tv_weight * tv_loss)
                
                total_loss.backward()
                
                return total_loss
            
            if use_l_bfgs:
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()
            
            # Print progress
            if (step + 1) % 50 == 0 or step == 0:
                with torch.no_grad():
                    target_features = self._get_features(
                        target_img,
                        self.content_layers + self.style_layers
                    )
                    c_loss = self.content_loss(target_features, content_features)
                    s_loss = self.style_loss(target_features, style_features)
                    tv_loss = self.total_variation_loss(target_img)
                    
                    print(f"Step [{step+1}/{num_steps}], "
                          f"Content: {c_loss.item():.4f}, "
                          f"Style: {s_loss.item():.4f}, "
                          f"TV: {tv_loss.item():.4f}")
        
        print("Style transfer complete!")
        
        # Final clamping
        with torch.no_grad():
            target_img.clamp_(-2.5, 2.5)
        
        return self.deprocess_image(target_img)


# ==================== Post-processing for further smoothing ====================

def post_process_smoothing(image, method='bilateral'):
    """
    Apply post-processing to reduce remaining patches
    
    Args:
        image: PIL Image
        method: 'bilateral', 'gaussian', or 'median'
    """
    import cv2
    
    # Convert to numpy
    img_array = np.array(image)
    
    if method == 'bilateral':
        # Bilateral filter: smooth while preserving edges
        smoothed = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == 'gaussian':
        # Gaussian blur
        smoothed = cv2.GaussianBlur(img_array, (5, 5), 0)
    elif method == 'median':
        # Median filter: good for removing salt-and-pepper noise
        smoothed = cv2.medianBlur(img_array, 5)
    else:
        smoothed = img_array
    
    return Image.fromarray(smoothed)


# ==================== Usage Examples ====================
import os
if __name__ == "__main__":
    content_image_pth = 'data/content-images/Tuebingen_Neckarfront.jpg'
    style_image_pth = 'data/style-images/vangogh_starry_night.jpg'
    op_fldr = "data/styled_images"
    if not os.path.exists(op_fldr):
        os.makedirs(op_fldr)
    block_idx = 3
    # style_indices = [1, 2, 3, 4]  # EfficientNet blocks to use for style
    print("="*70)
    print("IMPROVED STYLE TRANSFER - Patch Reduction")
    print("="*70)
    
    # Example 1: Basic improved transfer
    print("\n1. BASIC IMPROVED TRANSFER")
    print("-"*70)
    
    nst = ImprovedStyleTransfer(model_type='vgg19', block_idx=block_idx)
    
    result = nst.transfer(
        content_path=content_image_pth,
        style_path=style_image_pth,
        num_steps=500,
        learning_rate=0.01,
        tv_weight=1e-3,      # Smoothness - increase for more smoothing
        style_weight=1e5     # Reduced style weight to prevent over-stylization
    )
    result.save(f'{op_fldr}/improved_output_{block_idx}.jpg')
    print("Saved: imp_op_imgs/improved_output.jpg")
    
    # Example 2: High-quality with L-BFGS
    print("\n2. HIGH-QUALITY (L-BFGS optimizer)")
    print("-"*70)
    
    result_hq = nst.transfer(
        content_path=content_image_pth,
        style_path=style_image_pth,
        num_steps=100,       # Fewer steps needed with L-BFGS
        use_l_bfgs=True,     # Better optimizer
        tv_weight=1e-3,
        style_weight=1e5
    )
    result_hq.save(f'{op_fldr}/high_quality_output_{block_idx}.jpg')
    print("Saved: imp_op_imgs/high_quality_output.jpg")