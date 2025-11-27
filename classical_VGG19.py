# requirements: make conda env with environment_py37.yml
#  python version 3.7.16 cuDNN : ~8.2 ; wsl; 
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as transforms
from PIL import Image
from collections import OrderedDict

# --- paths ---
Content_image_path = r"data/content-images/Tuebingen_neckarfront.jpg"
Style_image_dir = r"data/style-images/vangogh_starry_night.jpg"
output_dir = r"data/styled_images"
model_dir = os.path.join(os.getcwd(), "Models")
vgg_path = r"pretrain_vgg_model/vgg_conv.pth" 

# --- device ---
# region model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- VGG definition that returns intermediate activations by requested keys ---
class VGG(nn.Module):
    def __init__(self, pool="max"):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # pooling layers: choose max or avg
        Pool = nn.MaxPool2d if pool == "max" else nn.AvgPool2d
        self.pool_1 = Pool(kernel_size=2, stride=2)
        self.pool_2 = Pool(kernel_size=2, stride=2)
        self.pool_3 = Pool(kernel_size=2, stride=2)
        self.pool_4 = Pool(kernel_size=2, stride=2)
        self.pool_5 = Pool(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out["conv1_1"] = F.relu(self.conv1_1(x))
        out["conv1_2"] = F.relu(self.conv1_2(out["conv1_1"]))
        out["pool_1"] = self.pool_1(out["conv1_2"])

        out["conv2_1"] = F.relu(self.conv2_1(out["pool_1"]))
        out["conv2_2"] = F.relu(self.conv2_2(out["conv2_1"]))
        out["pool_2"] = self.pool_2(out["conv2_2"])

        out["conv3_1"] = F.relu(self.conv3_1(out["pool_2"]))
        out["conv3_2"] = F.relu(self.conv3_2(out["conv3_1"]))
        out["conv3_3"] = F.relu(self.conv3_3(out["conv3_2"]))
        out["conv3_4"] = F.relu(self.conv3_4(out["conv3_3"]))
        out["pool_3"] = self.pool_3(out["conv3_4"])

        out["conv4_1"] = F.relu(self.conv4_1(out["pool_3"]))
        out["conv4_2"] = F.relu(self.conv4_2(out["conv4_1"]))
        out["conv4_3"] = F.relu(self.conv4_3(out["conv4_2"]))
        out["conv4_4"] = F.relu(self.conv4_4(out["conv4_3"]))
        out["pool_4"] = self.pool_4(out["conv4_4"])

        out["conv5_1"] = F.relu(self.conv5_1(out["pool_4"]))
        out["conv5_2"] = F.relu(self.conv5_2(out["conv5_1"]))
        out["conv5_3"] = F.relu(self.conv5_3(out["conv5_2"]))
        out["conv5_4"] = F.relu(self.conv5_4(out["conv5_3"]))
        out["pool_5"] = self.pool_5(out["conv5_4"])

        return [out[k] for k in out_keys]

# --- Gram matrix and Gram MSE loss ---
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        Fflat = input.view(b, c, h * w)
        G = torch.bmm(Fflat, Fflat.transpose(1, 2))
        G = G.div(h * w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        gram_input = GramMatrix()(input)
        return nn.MSELoss()(gram_input, target)
#endregion
# region Imgepreprocessing / postprocessing

img_size = 512
prep = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),  # RGB -> BGR
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),
    transforms.Lambda(lambda x: x.mul_(255.0)),
])

postpOperations = transforms.Compose([
    transforms.Lambda(lambda x: x.mul_(1.0/255.0)),
    transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1,1,1]),
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),  # BGR -> RGB
])


def postpropcess(tensor):
    t = postpOperations(tensor)
    t = torch.clamp(t, 0.0, 1.0)
    img = transforms.ToPILImage()(t)
    return img

# endregion

# region Low-resolution pass

vgg = VGG(pool="max").to(device)
vgg.load_state_dict(torch.load(vgg_path, map_location=device))
for param in vgg.parameters():
    param.requires_grad = False

# --- load images ---
img_files = [Style_image_dir, Content_image_path]
imgs = [Image.open(f).convert("RGB") for f in img_files]
imgs_t = [prep(img).unsqueeze(0).to(device) for img in imgs]  # shape: (1,C,H,W)
style_image, content_image = imgs_t

# initialize optimization image (start from content)
opt_img = content_image.clone().detach().requires_grad_(True)

# --- define layers, losses and weights ---
style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
content_layers = ["conv4_2"]
loss_layers = style_layers + content_layers

# create loss functions list in same order
loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
loss_fns = [lf if not hasattr(lf, "to") else lf.to(device) for lf in loss_fns]

# good weights
style_weights = [1e3 / n**2 for n in [64, 128, 256, 512, 512]]
content_weights = [1.0]
weights = style_weights + content_weights

# compute targets
with torch.no_grad():
    style_targets = [GramMatrix().to(device)(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
targets = style_targets + content_targets

# quick env check
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())

# --- optimization loop (low-res) ---
max_iter = 500
show_iter = 50
optimizer = optim.LBFGS([opt_img], max_iter=20)
iteration = [0]  # mutable counter inside closure

print("Starting low-res style transfer...")
while iteration[0] <= max_iter:
    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, loss_layers)
        layer_losses = [weights[i] * loss_fns[i](A, targets[i]) for i, A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        iteration[0] += 1
        if iteration[0] % show_iter == (show_iter - 1):
            print("Iteration: %d, loss: %f" % (iteration[0] + 1, loss.item()))
        return loss
    optimizer.step(closure)
    if iteration[0] >= max_iter:
        break

# show low-res result
out_img = postpropcess(opt_img.detach().cpu().squeeze())
try:
    out_img.show() 
except Exception:
    pass

# endregion

# region High-resolution pass
# --- High-resolution pass ---
img_size_hr = 800
prep_hr = transforms.Compose([
    transforms.Resize(img_size_hr),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),  # RGB -> BGR
    transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1,1,1]),
    transforms.Lambda(lambda x: x.mul_(255.0)),
])

# prepare hr inputs
imgs_t_hr = [prep_hr(img).unsqueeze(0).to(device) for img in imgs]
style_image_hr, content_image_hr = imgs_t_hr

# initialize with upsampled low-res result
opt_img_hr = prep_hr(out_img).unsqueeze(0).to(device)
opt_img_hr = opt_img_hr.clone().detach().type_as(content_image_hr).requires_grad_(True)

# recompute high-res targets
with torch.no_grad():
    style_targets = [GramMatrix().to(device)(A).detach() for A in vgg(style_image_hr, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image_hr, content_layers)]
targets = style_targets + content_targets

# optimize high-res
max_iter_hr = 500
optimizer = optim.LBFGS([opt_img_hr], max_iter=20)
iteration = [0]
print("Starting high-res style transfer...")
while iteration[0] <= max_iter_hr:
    def closure_hr():
        optimizer.zero_grad()
        out = vgg(opt_img_hr, loss_layers)
        layer_losses = [weights[i] * loss_fns[i](A, targets[i]) for i, A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        iteration[0] += 1
        if iteration[0] % show_iter == (show_iter - 1):
            print("HR Iteration: %d, loss: %f" % (iteration[0] + 1, loss.item()))
        return loss
    optimizer.step(closure_hr)
    if iteration[0] >= max_iter_hr:
        break

# show high-res result
out_img_hr = postpropcess(opt_img_hr.detach().cpu().squeeze())
try:
    out_img_hr.show()
except Exception:
    pass

# optionally save results
out_img_hr.save(os.path.join(output_dir, "nst_result_highres.png"))
print("Saved high-res result to:", os.path.join(output_dir, "nst_result_highres.png"))
# endregion