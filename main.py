import os
import zfpy
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms

class Auto_Encoder(nn.Module):
    def __init__(self, input_size=100, latent_size=16):
        super(Auto_Encoder, self).__init__()
        nc = 256
        nc4 = int(nc / 4)
        self.input_size = input_size
        self.latent_size = latent_size

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, nc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nc),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((latent_size, latent_size)),
            nn.Conv2d(nc, nc4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nc4),
            nn.SiLU(inplace=True),
            nn.Conv2d(nc4, 3, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.Conv2d(3, nc4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nc4),
            nn.SiLU(inplace=True),
            nn.Conv2d(nc4, nc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nc),
            nn.SiLU(inplace=True),
            nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=True),
            nn.Conv2d(nc, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.enc(x)
        decoded = self.dec(encoded)
        return decoded

def compress_and_save_latents(model_class, model_path, img_folder, patch_size=100, latent_save_path='./latents'):
    device = torch.device("cpu")
    model = model_class().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()

    os.makedirs(latent_save_path, exist_ok=True)

    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        original_img = Image.open(img_path).convert('RGB')
        original_img = original_img.resize((2000, 2000))
        patches = []
        
        for i in range(0, 2000, patch_size):
            for j in range(0, 2000, patch_size):
                patch = original_img.crop((j, i, j + patch_size, i + patch_size))
                patch = transforms.ToTensor()(patch).unsqueeze(0).to(device)
                patches.append(patch)
        
        latent_list = []
        for patch in patches:
            with torch.no_grad():
                encoded_patch = model.enc(patch)
                latent_list.append(encoded_patch)

        all_latents = torch.cat(latent_list, dim=0).cpu().numpy()

        tolerance = 30
        compressed_data = zfpy.compress_numpy(all_latents, tolerance=tolerance)

        latent_file_path = os.path.join(latent_save_path, f"{os.path.splitext(img_name)[0]}.zfp")
        with open(latent_file_path, 'wb') as f:
            f.write(compressed_data)
        
        print(f"Saved compressed latent for {img_name} at {latent_file_path}")

# Usage
compress_and_save_latents(Auto_Encoder, 'checkpoint_epoch_7.pth', 'input_images', patch_size=100, latent_save_path='latents')
