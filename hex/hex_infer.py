import os
import glob
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from hex.hex_architecture import CustomModel

# Biomarker names (same mapping as your script)
biomarker_names = {
    1: "DAPI",
    2: "CD8",
    3: "Pan-Cytokeratin",
    4: "CD3e",
    5: "CD163",
    6: "CD20",
    7: "CD4",
    8: "FAP",
    9: "CD138",
    10: "CD11c",
    11: "CD66b",
    12: "aSMA",
    13: "CD68",
    14: "Ki67",
    15: "CD31",
    16: "Collagen IV",
    17: "Granzyme B",
    18: "MMP9",
    19: "PD-1",
    20: "CD44",
    21: "PD-L1",
    22: "E-cadherin",
    23: "LAG3",
    24: "Mac2/Galectin-3",
    25: "FOXP3",
    26: "CD14",
    27: "EpCAM",
    28: "CD21",
    29: "CD45",
    30: "MPO",
    31: "TCF-1",
    32: "ICOS",
    33: "Bcl-2",
    34: "HLA-E",
    35: "CD45RO",
    36: "VISTA",
    37: "HIF1A",
    38: "CD39",
    39: "CD40",
    40: "HLA-DR"
}

def load_model(checkpoint_path: str) -> nn.Module:
    model = CustomModel(visual_output_dim=1024, num_outputs=40)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = nn.DataParallel(model).cuda()
    model.eval()
    # keep consistent with your original code
    try:
        model.module.training_status = False
    except Exception:
        pass
    return model


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        return img, p


def infer_patches(
    patch_root: str,
    checkpoint_path: str,
    input_csv: str,   # ✅ 기존 CSV (slide_index, x, y가 있는 파일)
    out_csv: str,
    batch_size: int = 128,
    num_workers: int = 8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA를 사용할 수 없어. GPU 환경에서 실행해줘.")

    # ===== 1) 기존 CSV 로드해서 (slide_index -> x,y) 맵 만들기 =====
    base_df = pd.read_csv(input_csv)

    required_cols = {"slide_index", "x", "y"}
    if not required_cols.issubset(set(base_df.columns)):
        raise ValueError(
            f"input_csv에는 {required_cols} 컬럼이 필요해. 현재 컬럼: {set(base_df.columns)}"
        )

    base_df = base_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["slide_index", "x", "y"]).copy()
    base_df["slide_index"] = base_df["slide_index"].astype(int)
    base_df["x"] = base_df["x"].astype(int)
    base_df["y"] = base_df["y"].astype(int)

    xy_map = base_df.set_index("slide_index")[["x", "y"]]

    # ===== 2) patch 이미지들 수집 =====
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(patch_root, "**", ext), recursive=True))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found under: {patch_root}")

    print(f"Found {len(image_paths)} patches under {patch_root}")

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])

    ds = ImagePathDataset(image_paths, transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ===== 3) 모델 로드 & 추론 =====
    model = load_model(checkpoint_path)

    all_preds = []
    all_paths = []

    print("Starting inference...")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for imgs, paths in tqdm(dl, total=len(dl)):
            imgs = imgs.to(device, non_blocking=True)

            # training forward signature 유지용 dummy labels
            dummy_labels = torch.zeros((imgs.size(0), 40), device=device, dtype=torch.float16)

            outputs, _ = model(imgs, dummy_labels, 0)  # (B, 40)
            all_preds.append(outputs.detach().float().cpu().numpy())
            all_paths.extend(paths)

    all_preds = np.concatenate(all_preds, axis=0)

    # ===== 4) output CSV 구성 (기존 CSV의 x,y를 그대로 포함) =====
    rows = []
    missing_xy = 0

    for i, p in enumerate(all_paths):
        p = str(p)
        patch_name = Path(p).name

        # patch 파일명이 "{slide_index}.png" 라는 전제
        try:
            slide_index = int(Path(p).stem)
        except Exception:
            slide_index = None

        if slide_index is not None and slide_index in xy_map.index:
            x = int(xy_map.loc[slide_index, "x"])
            y = int(xy_map.loc[slide_index, "y"])
        else:
            x = np.nan
            y = np.nan
            missing_xy += 1

        row = {
            "image_path": p,
            "patch_name": patch_name,
            "slide_index": slide_index,
            "x": x,
            "y": y,
        }
        for j in range(40):
            biomarker = biomarker_names[j + 1]
            row[f"{biomarker}_pred"] = float(all_preds[i, j])
        rows.append(row)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"✅ Saved predictions to: {out_csv}")
    if missing_xy > 0:
        print(f"⚠️ Warning: {missing_xy} patches had no matching x,y in input_csv (left as NaN).")


if __name__ == "__main__":
    # ====== 너 환경에 맞게 여기만 바꿔서 실행 ======
    PATCH_ROOT = "/home/26w_kyr/TCGA-MN-A4N4-01Z-00-DX2.9550732D-8FB1-43D9-B094-7C0CD310E9C0"  # patch 폴더
    CHECKPOINT = "/home/26w_kyr/HEX/hex/checkpoint.pth"  # HEX checkpoint

    # ✅ x,y 있는 "기존 좌표 CSV" (extract_coordinate 결과)
    INPUT_CSV  = "/home/26w_kyr/TCGA-MN-A4N4-01Z-00-DX2.9550732D-8FB1-43D9-B094-7C0CD310E9C0.csv"

    # ✅ 예측 + (x,y) 포함해서 저장될 CSV
    OUT_CSV    = "./hex_2_with_xy.csv"

    infer_patches(
        patch_root=PATCH_ROOT,
        checkpoint_path=CHECKPOINT,
        input_csv=INPUT_CSV,
        out_csv=OUT_CSV,
        batch_size=64,
        num_workers=8
    )
