# extract_coordinate.py
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import openslide
from PIL import Image

from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.morphology import (
    binary_opening,
    remove_small_objects,
    remove_small_holes,
    disk
)


# -----------------------------
# Tissue mask (thumbnail-based)
# -----------------------------
def make_tissue_mask(
    slide: openslide.OpenSlide,
    mask_level: int = None,
    min_object_size: int = 128,
    min_hole_size: int = 128
):
    """
    Create tissue (foreground) mask on a low-resolution level.
    Returns:
      mask_bool (H, W),
      used_level,
      downsample (level0 -> mask_level)
    """
    if mask_level is None:
        mask_level = slide.level_count - 1  # smallest level for speed

    w, h = slide.level_dimensions[mask_level]
    thumb = slide.read_region((0, 0), mask_level, (w, h)).convert("RGB")
    img = np.asarray(thumb).astype(np.float32) / 255.0

    # RGB -> HSV
    hsv = rgb2hsv(img)
    sat = hsv[..., 1]   # saturation
    val = hsv[..., 2]   # value (brightness)

    # Otsu thresholds
    try:
        thr_s = threshold_otsu(sat)
    except Exception:
        thr_s = float(np.mean(sat))

    try:
        thr_v = threshold_otsu(val)
    except Exception:
        thr_v = float(np.mean(val))

    # Tissue tends to be:
    #  - higher saturation
    #  - or not extremely bright
    mask = (sat > thr_s) | (val < thr_v)

    # Morphological cleanup
    mask = binary_opening(mask, footprint=disk(2))
    mask = remove_small_objects(mask, min_size=min_object_size)
    mask = remove_small_holes(mask, area_threshold=min_hole_size)

    downsample = float(slide.level_downsamples[mask_level])
    return mask.astype(bool), mask_level, downsample


def tissue_fraction_from_mask(
    mask_bool: np.ndarray,
    downsample: float,
    x0: int,
    y0: int,
    patch_size: int
) -> float:
    """
    Estimate tissue fraction of a level-0 patch using the mask.
    """
    xs = int(x0 / downsample)
    ys = int(y0 / downsample)
    ps = max(1, int(patch_size / downsample))

    H, W = mask_bool.shape
    xe = min(W, xs + ps)
    ye = min(H, ys + ps)
    xs = max(0, xs)
    ys = max(0, ys)

    if xe <= xs or ye <= ys:
        return 0.0

    return float(mask_bool[ys:ye, xs:xe].mean())


def extract_coords_from_wsi(
    wsi_path: str,
    out_csv_path: str,
    patch_size: int,
    step_size: int,
    mask_level: int,
    min_tissue_frac: float
):
    slide_id = os.path.splitext(os.path.basename(wsi_path))[0]
    slide = openslide.OpenSlide(wsi_path)

    tissue_mask, used_level, downsample = make_tissue_mask(
        slide,
        mask_level=mask_level
    )

    W0, H0 = slide.level_dimensions[0]
    x_max = W0 - patch_size
    y_max = H0 - patch_size

    rows = []
    idx = 0

    if x_max > 0 and y_max > 0:
        for y in range(0, y_max + 1, step_size):
            for x in range(0, x_max + 1, step_size):
                tf = tissue_fraction_from_mask(
                    tissue_mask, downsample, x, y, patch_size
                )
                if tf >= min_tissue_frac:
                    rows.append({
                        "slide_id": slide_id,
                        "slide_index": idx,
                        "x": int(x),
                        "y": int(y)
                    })
                    idx += 1

    slide.close()

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df = pd.DataFrame(rows, columns=["slide_id", "slide_index", "x", "y"])
    df.to_csv(out_csv_path, index=False)


# -----------------------------
# CLI
# -----------------------------
def main():
    # =============================
    # CONFIG (여기서 직접 설정)
    # =============================
    wsi_dir = "/data-hdd/home/shared/TCGA/WSI/939eae7b-afc7-437a-81e3-1e9615897027"
    out_csv_dir = "/data-hdd/home/shared/TCGA/WSI/939eae7b-afc7-437a-81e3-1e9615897027"

    patch_size = 224
    step_size = 224
    mask_level = None          # None이면 자동으로 마지막 레벨
    min_tissue_frac = 0.3
    # =============================

    os.makedirs(out_csv_dir, exist_ok=True)

    wsi_files = sorted(
        [f for f in os.listdir(wsi_dir) if f.lower().endswith(".svs")]
    )

    for f in tqdm(wsi_files, desc="WSI"):
        wsi_path = os.path.join(wsi_dir, f)
        slide_id = os.path.splitext(f)[0]
        out_csv_path = os.path.join(out_csv_dir, f"{slide_id}.csv")

        try:
            extract_coords_from_wsi(
                wsi_path=wsi_path,
                out_csv_path=out_csv_path,
                patch_size=patch_size,
                step_size=step_size,
                mask_level=mask_level,
                min_tissue_frac=min_tissue_frac
            )
        except Exception as e:
            print(f"[ERROR] {slide_id}: {e}")
            pd.DataFrame(
                columns=["slide_id", "slide_index", "x", "y"]
            ).to_csv(out_csv_path, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
