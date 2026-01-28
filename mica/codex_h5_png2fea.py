from pathlib import Path
import os
from os.path import join
import numpy as np
from PIL import Image
from tqdm import tqdm
import h5py
import openslide

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def check_mag(wsi):
    try:
        currMPP = float(wsi.properties['aperio.MPP'])
    except:
        try:
            currMPP = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])
        except:
            currMPP = 0.25
    if currMPP < 0.2:
        return 80
    elif 0.2 <= currMPP < 0.3:
        return 40
    elif 0.4 <= currMPP < 0.6:
        return 20
    else:
        return 40

slide_ext = '.svs'
patch_size = 224

# ✅ 지정해야 하는 것들
he_path = Path("/data-hdd/home/shared/TCGA/WSI/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/TCGA-MN-A4N4-01Z-00-DX2.9550732D-8FB1-43D9-B094-7C0CD310E9C0.svs")
h5_file = Path('/home/26w_kyr/hex_2.h5')              # 파일 하나
npy_dir = Path('/home/26w_kyr')     # npy 저장 위치
fea_dir = Path('/home/26w_kyr')        # features.h5 저장 위치

npy_dir.mkdir(parents=True, exist_ok=True)
fea_dir.mkdir(parents=True, exist_ok=True)

# --- 1) h5 -> npy ---
he_id = h5_file.stem
wsi = openslide.open_slide(str(he_path))
mag = check_mag(wsi)

scale_down_factor = int(patch_size / (40/mag))
width = wsi.dimensions[0] // scale_down_factor + 1
height = wsi.dimensions[1] // scale_down_factor + 1
wsi.close()

with h5py.File(str(h5_file), 'r') as f:
    codex_prediction = f['codex_prediction'][:]   # (N, C)
    coords = f['coords'][:]                       # (N, 2)

C = codex_prediction.shape[1]
codex_image = np.zeros((height, width, C), dtype=np.float16)

for i in range(len(coords)):
    x, y = coords[i]
    x, y = int(x/scale_down_factor), int(y/scale_down_factor)
    if 0 <= x < width and 0 <= y < height:
        codex_image[y, x] = codex_prediction[i]

save_path = join(str(npy_dir), f'{he_id}.npy')
np.save(save_path, codex_image)

# --- 2) npy -> DINO features.h5 ---
class ImageChannelDataset(Dataset):
    def __init__(self, img_dir, num_channels, transform=None):
        self.img_dir = Path(img_dir)
        self.img_paths = list(self.img_dir.glob('*.npy'))
        self.transform = transform
        self.num_channels = num_channels

    def __len__(self):
        return len(self.img_paths) * self.num_channels

    def __getitem__(self, idx):
        img_idx = idx // self.num_channels
        channel_idx = idx % self.num_channels

        img_path = self.img_paths[img_idx]
        img_name = img_path.stem
        img = np.load(img_path)  # [H,W,C]

        channel = img[:, :, channel_idx].astype(np.float32)

        # robust min-max (0~99.5 퍼센타일 사용)
        lo = np.percentile(channel, 0.5)
        hi = np.percentile(channel, 99.5)
        if hi <= lo:
            channel_u8 = np.zeros_like(channel, dtype=np.uint8)
        else:
            channel_norm = (channel - lo) / (hi - lo)
            channel_norm = np.clip(channel_norm, 0, 1)
            channel_u8 = (channel_norm * 255).astype(np.uint8)

        channel_stacked = np.stack([channel_u8, channel_u8, channel_u8], axis=2)
        channel_pil = Image.fromarray(channel_stacked, mode="RGB")

        if self.transform:
            channel_pil = self.transform(channel_pil)

        return {'image': channel_pil, 'name': img_name, 'channel_idx': channel_idx}

transform_val = transforms.Compose([
    transforms.Resize((224,224), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

batch_size = 128  
num_workers = 8

dataset = ImageChannelDataset(npy_dir, num_channels=C, transform=transform_val)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
model.cuda()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.eval()

h5_out = fea_dir / 'features.h5'
features_dict = {}

import h5py
with h5py.File(h5_out, 'w') as out:
    for batch in tqdm(dataloader, desc='Extracting features'):
        images = batch['image'].cuda()
        names = batch['name']
        channel_indices = batch['channel_idx']

        with torch.no_grad():
            feats = model(images).cpu().numpy()

        for j, (name, ch) in enumerate(zip(names, channel_indices)):
            if name not in features_dict:
                features_dict[name] = np.zeros((C, feats.shape[1]), dtype=np.float32)
            features_dict[name][int(ch)] = feats[j]

    for name, feats in features_dict.items():
        out.create_dataset(name, data=feats)

print("Saved:", h5_out)
