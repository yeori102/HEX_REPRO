#!/usr/bin/env python3
# vc.py
# CSV (per-tile protein expression predictions) -> Virtual CODEX channel maps (npz) + channel PNGs
# + Determine grid (H, W) from full SVS level0 dimensions
# + Save tile occupancy overlay visualization on SVS thumbnail
#
# - Build (H, W, C) map on a stride-based grid
# - Sanitize filenames so characters like '/' can be safely saved

import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import openslide

# =====================
# Configuration
# =====================
CSV_PATH   = "/home/26w_kyr/hex_2_with_xy.csv"   # Prediction CSV path
SVS_PATH   = "/data-hdd/home/shared/TCGA/WSI/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/TCGA-MN-A4N4-01Z-00-DX2.9550732D-8FB1-43D9-B094-7C0CD310E9C0.svs"                             # Set this to your single slide SVS path
OUT_DIR    = "./virtual_codex_2"

STRIDE     = 224   # Tile spacing (=STEP_SIZE); must match the step_size used for coordinate extraction
PATCH_SIZE = 224   # For reference only (not directly needed for map construction)

# Save options
SAVE_NPZ   = True
SAVE_PNGS  = True

# PNG scaling
# - "per_channel_minmax": per-channel min~max normalization then scale to 0~255
# - "global_minmax": shared min~max across all channels (falls back to per-channel here)
# - "clip_0_1": assumes values are in 0~1 range, clips then scales to 0~255
PNG_SCALE_MODE = "per_channel_minmax"
EPS = 1e-8

# =====================
# Utilities
# =====================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def sanitize_filename(name: str) -> str:
    """
    Convert to a safe filename:
    - Remove/replace path separators like '/' and '\'
    - Replace spaces with '_'
    - Replace other unsafe characters with '_'
    """
    name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    # Allow: alphanumeric, underscore, dash, dot, parentheses, plus
    name = re.sub(r"[^0-9A-Za-z_\-.\(\)\+]", "_", name)
    # Collapse consecutive '_'
    name = re.sub(r"_+", "_", name).strip("_")
    if len(name) == 0:
        name = "channel"
    return name

def to_uint8_map(m: np.ndarray, mode: str) -> np.ndarray:
    """
    m: (H, W) float map
    return uint8 (H, W)
    """
    if mode == "clip_0_1":
        x = np.clip(m, 0.0, 1.0)
        return (x * 255.0).round().astype(np.uint8)

    if mode == "global_minmax":
        # global_minmax safely falls back to per-channel here
        mode = "per_channel_minmax"

    # per_channel_minmax
    mn = np.nanmin(m)
    mx = np.nanmax(m)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < EPS:
        return np.zeros_like(m, dtype=np.uint8)
    x = (m - mn) / (mx - mn + EPS)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)

# =====================
# Main
# =====================
def main():
    ensure_dir(OUT_DIR)

    # 0) File check
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not os.path.exists(SVS_PATH):
        raise FileNotFoundError(f"SVS not found: {SVS_PATH}")

    # 1) Load CSV
    df = pd.read_csv(CSV_PATH)

    # Required columns check
    required = ["x", "y"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV requires column '{c}'. Current columns: {list(df.columns)[:20]} ...")

    # 1-1) Handle x,y NaN/inf (rows with NaN coordinates cannot be mapped to grid -> drop)
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=["x", "y"]).copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"⚠️ Dropped {dropped} rows due to NaN/inf in x/y")

    # 2) Find prediction channel columns
    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    if len(pred_cols) == 0:
        raise ValueError("No prediction columns ending with '*_pred' found. Please check the CSV header.")

    # Assume x,y are tile top-left coordinates
    df["x"] = df["x"].astype(int)
    df["y"] = df["y"].astype(int)
    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()

    # 3) Determine grid size based on SVS level0 dimensions
    slide = openslide.OpenSlide(SVS_PATH)
    W0, H0 = slide.level_dimensions[0]  # level0 pixel size
    slide.close()

    W = (W0 - 1) // STRIDE + 1
    H = (H0 - 1) // STRIDE + 1

    print(f"SVS level0 size: W0={W0}, H0={H0}")
    print(f"Grid size: H={H}, W={W} (stride={STRIDE})")
    print(f"Total tiles in CSV: {len(df)} | channels: {len(pred_cols)}")

    # 4) Create (H, W, C) map (float32)
    # Empty cells (no tissue) are NaN; replaced with 0 when saving PNGs
    C = len(pred_cols)
    maps = np.full((H, W, C), np.nan, dtype=np.float32)

    # Map each row to grid index
    gx = (xs // STRIDE).astype(int)
    gy = (ys // STRIDE).astype(int)

    # Bounds check (guard against outliers)
    valid = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    if not np.all(valid):
        bad = int(np.count_nonzero(~valid))
        print(f"⚠️ {bad} coordinates out of grid bounds; excluding them.")
        df_valid = df.loc[valid].reset_index(drop=True)
        gx = gx[valid]
        gy = gy[valid]
    else:
        df_valid = df

    # Channel value array
    vals = df_valid[pred_cols].to_numpy(dtype=np.float32)  # (N, C)

    # Fill map (duplicate (gx,gy) entries are overwritten with the last value)
    maps[gy, gx, :] = vals
    

    # 5) Save npz
    if SAVE_NPZ:
        npz_path = os.path.join(OUT_DIR, "virtual_codex_maps_float32.npz")
        np.savez_compressed(
            npz_path,
            maps=maps,                         # (H,W,C)
            pred_cols=np.array(pred_cols),     # Original channel names
            stride=np.int32(STRIDE),
            H=np.int32(H),
            W=np.int32(W),
            level0_W=np.int32(W0),
            level0_H=np.int32(H0),
            svs_path=np.array([SVS_PATH]),
        )
        print(f"Saved numeric maps: {npz_path}")

    # 6) Save per-channel PNGs
    if SAVE_PNGS:
        png_dir = os.path.join(OUT_DIR, "png_channels")
        ensure_dir(png_dir)

        for ci, col in enumerate(pred_cols):
            m = maps[:, :, ci]

            # Replace NaN with 0 (empty tiles become black)
            m0 = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

            img_u8 = to_uint8_map(m0, PNG_SCALE_MODE)

            safe_col = sanitize_filename(col)
            out_path = os.path.join(png_dir, f"{safe_col}.png")

            # Save as grayscale
            Image.fromarray(img_u8, mode="L").save(out_path)

        print(f"Saved channel PNGs: {png_dir} ({len(pred_cols)} files)")

    # =====================
    # 7) Visualization: occupancy (tile presence) + SVS thumbnail overlay
    # =====================

    # Occupancy map: 255 only where tiles exist in CSV
    occ = np.zeros((H, W), dtype=np.uint8)
    occ[gy, gx] = 255

    occ_path = os.path.join(OUT_DIR, "occupancy_tiles.png")
    Image.fromarray(occ, mode="L").save(occ_path)
    print(f"Saved occupancy map: {occ_path}")

    # Overlay on SVS thumbnail (using the smallest level)
    slide = openslide.OpenSlide(SVS_PATH)
    thumb_level = slide.level_count - 1
    tw, th = slide.level_dimensions[thumb_level]
    thumb = slide.read_region((0, 0), thumb_level, (tw, th)).convert("RGB")
    slide.close()

    # Resize occupancy to thumbnail size and paint in red
    occ_img = Image.fromarray(occ, mode="L").resize((tw, th), resample=Image.NEAREST)

    thumb_np = np.array(thumb).astype(np.uint8)
    occ_np = np.array(occ_img)
    overlay = thumb_np.copy()

    mask = occ_np > 0
    overlay[mask, 0] = 255  # R
    overlay[mask, 1] = 0    # G
    overlay[mask, 2] = 0    # B

    overlay_path = os.path.join(OUT_DIR, "occupancy_overlay_on_thumbnail.png")
    Image.fromarray(overlay).save(overlay_path)
    print(f"Saved overlay thumbnail: {overlay_path}")

    print("✅ Done.")

if __name__ == "__main__":
    main()
