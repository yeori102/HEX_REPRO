# vc_render.py
import os
import re
import numpy as np
from PIL import Image

# =====================
# Input/Output Settings
# =====================
NPZ_PATH = "/home/26w_kyr/virtual_codex_2/virtual_codex_maps_float32.npz"
OUT_DIR  = "./virtual_codex_2_rendered"

# Save pseudo-color PNGs by channel
SAVE_PSEUDOCOLOR = True
PSEUDOCOLOR_DIR = os.path.join(OUT_DIR, "png_pseudocolor")

# If you also want to save grayscale PNGs per channel, set to True
SAVE_GRAYSCALE = False
GRAYSCALE_DIR = os.path.join(OUT_DIR, "png_grayscale")

# composite save
COMPOSITE_DIR = os.path.join(OUT_DIR, "composites")

# Normalization method: Scale to 0–1 after percentile clipping
P_LOW  = 1.0
P_HIGH = 99.0

# ✅ Important: When the background is predominantly zero, exclude zeros from percentile calculations.
EXCLUDE_ZERO_FOR_NORM = True

# Composite channel intensity weights (adjust as needed)
DEFAULT_WEIGHT = 1.0

# =====================
# Utility
# =====================
def safe_name(s: str) -> str:
    s = s.strip()
    s = s.replace(os.sep, "_")
    s = s.replace("/", "_")
    s = re.sub(r'[\\:*?"<>|]', "_", s)
    return s

def norm01(a: np.ndarray, p_low=P_LOW, p_high=P_HIGH, eps=1e-8, exclude_zero=False) -> np.ndarray:
    """
    percentile clip 후 0~1 정규화
    - exclude_zero=True면 a==0을 percentile 계산에서 제외 (조직 타일만 강조하는 용도)
    """
    a = np.asarray(a, dtype=np.float32)

    # NaN/inf 제거
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.zeros_like(a, dtype=np.float32)

    if exclude_zero:
        mask = finite & (a != 0)
    else:
        mask = finite

    if not np.any(mask):
        # 전부 0이거나(또는 전부 비어있거나)인 케이스
        return np.zeros_like(a, dtype=np.float32)

    v = a[mask]
    lo = np.percentile(v, p_low)
    hi = np.percentile(v, p_high)

    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi - lo < eps):
        return np.zeros_like(a, dtype=np.float32)

    x = (a - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0).astype(np.float32)

    # finite 아닌 곳은 0으로
    x[~finite] = 0.0
    return x

def to_u8(x01: np.ndarray) -> np.ndarray:
    return (np.clip(x01, 0, 1) * 255.0 + 0.5).astype(np.uint8)

def pseudocolor(gray01: np.ndarray, rgb: tuple[int,int,int]) -> np.ndarray:
    """gray01 (H,W) in [0,1] -> color image (H,W,3)"""
    r, g, b = rgb
    out = np.zeros((gray01.shape[0], gray01.shape[1], 3), dtype=np.float32)
    out[..., 0] = gray01 * (r / 255.0)
    out[..., 1] = gray01 * (g / 255.0)
    out[..., 2] = gray01 * (b / 255.0)
    return to_u8(out)

def additive_composite(ch_list: list[np.ndarray],
                       color_list: list[tuple[int,int,int]],
                       weight_list=None,
                       exclude_zero_for_norm=False) -> np.ndarray:
    """
    ch_list: list of (H,W) float32 raw
    returns: (H,W,3) uint8
    """
    if weight_list is None:
        weight_list = [DEFAULT_WEIGHT] * len(ch_list)

    H, W = ch_list[0].shape
    acc = np.zeros((H, W, 3), dtype=np.float32)

    for ch, col, w in zip(ch_list, color_list, weight_list):
        ch01 = norm01(ch, exclude_zero=exclude_zero_for_norm)  # ✅ 핵심 변경
        r, g, b = col
        acc[..., 0] += w * ch01 * (r / 255.0)
        acc[..., 1] += w * ch01 * (g / 255.0)
        acc[..., 2] += w * ch01 * (b / 255.0)

    acc = np.clip(acc, 0.0, 1.0)
    return to_u8(acc)

# =====================
# 관례 색상 팔레트
# =====================
PALETTE = {
    "DAPI_pred": (  0, 120, 255),

    "CD3e_pred": (  0, 255,   0),
    "CD8_pred":  (  0, 255, 128),
    "CD4_pred":  (  0, 220, 255),
    "FOXP3_pred":(  0, 200, 200),
    "PD-1_pred": (  0, 160, 255),
    "ICOS_pred": (  0, 180, 120),
    "TCF-1_pred":( 80, 255, 180),
    "CD45RO_pred":( 60, 255,  60),
    "LAG3_pred": ( 60, 180, 255),
    "VISTA_pred":( 70, 130, 255),

    "Pan-Cytokeratin_pred": (255,  60,  60),
    "EpCAM_pred":           (255,   0, 255),
    "E-cadherin_pred":      (255,  80, 180),
    "PD-L1_pred":           (255, 150,  60),

    "CD20_pred": (255, 255,   0),
    "CD138_pred":(255, 210,   0),

    "CD68_pred": (255, 160,   0),
    "CD163_pred":(255, 120,   0),
    "CD14_pred": (255, 200,  80),
    "CD11c_pred":(255, 255, 120),
    "MPO_pred":  (255, 255, 255),
    "CD66b_pred":(255, 230, 150),
    "Mac2/Galectin-3_pred":(255, 180,  80),

    "CD31_pred": (180,  80, 255),
    "aSMA_pred": (200, 200, 200),
    "FAP_pred":  (255,   0, 180),
    "Collagen IV_pred": (180,   0, 255),
    "MMP9_pred": (255,  80,   0),

    "Ki67_pred": (255, 255, 255),
    "HIF1A_pred":(150, 150, 255),
    "HLA-DR_pred":(120, 255, 255),
    "HLA-E_pred": (120, 120, 255),
    "CD39_pred":  (120, 255, 120),
    "CD40_pred":  (255, 120, 255),
    "CD44_pred":  (255,  80, 255),
    "Bcl-2_pred": (255, 255, 200),
    "Granzyme B_pred": (0, 255, 255),
}

def get_color_for_channel(name: str) -> tuple[int,int,int]:
    return PALETTE.get(name, (255, 255, 255))

# =====================
# Composite 정의
# =====================
COMPOSITES = {
    "DAPI_CD8_PanCK": [
        ("DAPI_pred", (0,120,255), 1.0),
        ("CD8_pred",  (0,255,128), 1.0),
        ("Pan-Cytokeratin_pred", (255,60,60), 1.0),
    ],
    "DAPI_CD3_CD8": [
        ("DAPI_pred", (0,120,255), 1.0),
        ("CD3e_pred", (0,255,0),   1.0),
        ("CD8_pred",  (0,255,128), 1.0),
    ],
    "DAPI_PD1_PDL1": [
        ("DAPI_pred", (0,120,255), 1.0),
        ("PD-1_pred", (0,160,255), 1.0),
        ("PD-L1_pred",(255,150,60),1.0),
    ],
}

# =====================
# 메인
# =====================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if SAVE_PSEUDOCOLOR:
        os.makedirs(PSEUDOCOLOR_DIR, exist_ok=True)
    if SAVE_GRAYSCALE:
        os.makedirs(GRAYSCALE_DIR, exist_ok=True)
    os.makedirs(COMPOSITE_DIR, exist_ok=True)

    data = np.load(NPZ_PATH, allow_pickle=True)
    maps = data["maps"]  # (H,W,C) float32
    pred_cols = [str(x) for x in list(data["pred_cols"])]
    H, W, C = maps.shape
    print(f"Loaded maps: {H}x{W}x{C}")

    # ✅ 안전: NaN/inf는 0으로 (렌더링용)
    maps = np.nan_to_num(maps, nan=0.0, posinf=0.0, neginf=0.0)

    # 1) 채널별 PNG 저장
    for ci, col in enumerate(pred_cols):
        ch = maps[:, :, ci]

        ch01 = norm01(ch, exclude_zero=EXCLUDE_ZERO_FOR_NORM)
        ch_u8 = to_u8(ch01)

        if SAVE_GRAYSCALE:
            Image.fromarray(ch_u8, mode="L").save(
                os.path.join(GRAYSCALE_DIR, f"{safe_name(col)}.png")
            )

        if SAVE_PSEUDOCOLOR:
            rgb = get_color_for_channel(col)
            pc = pseudocolor(ch01, rgb)
            Image.fromarray(pc, mode="RGB").save(
                os.path.join(PSEUDOCOLOR_DIR, f"{safe_name(col)}.png")
            )

    print("Saved per-channel images.")

    # 2) composite 저장
    name_to_idx = {c: i for i, c in enumerate(pred_cols)}

    for comp_name, items in COMPOSITES.items():
        ch_list = []
        color_list = []
        weight_list = []
        missing = []

        for ch_name, rgb, w in items:
            if ch_name not in name_to_idx:
                missing.append(ch_name)
                continue
            ch_list.append(maps[:, :, name_to_idx[ch_name]])
            color_list.append(rgb)
            weight_list.append(w)

        if missing:
            print(f"[WARN] composite {comp_name}: missing channels {missing} -> skip")
            continue

        comp = additive_composite(
            ch_list, color_list, weight_list,
            exclude_zero_for_norm=EXCLUDE_ZERO_FOR_NORM
        )
        out_path = os.path.join(COMPOSITE_DIR, f"{safe_name(comp_name)}.png")
        Image.fromarray(comp, mode="RGB").save(out_path)

    print("Saved composites.")
    print(f"✅ Done. Outputs in: {OUT_DIR}")

if __name__ == "__main__":
    main()
