# export_wsi_thumbnail.py
import os
import openslide
from PIL import Image

SVS_PATH = "/data-hdd/home/shared/TCGA/WSI/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/TCGA-MN-A4N4-01Z-00-DX2.9550732D-8FB1-43D9-B094-7C0CD310E9C0.svs"
OUT_PNG  = "./wsi_thumbnail.png"

# Desired Downsampling Ratio
TARGET_DOWNSAMPLE = 16

slide = openslide.OpenSlide(SVS_PATH)

# Select the closest level
downs = slide.level_downsamples
level = min(range(len(downs)), key=lambda i: abs(downs[i] - TARGET_DOWNSAMPLE))

print(f"Using level {level}, downsample={downs[level]}, size={slide.level_dimensions[level]}")

img = slide.read_region(
    (0, 0),                
    level,                  
    slide.level_dimensions[level]
).convert("RGB")

img.save(OUT_PNG)
slide.close()

print(f"✅ saved {OUT_PNG}")
