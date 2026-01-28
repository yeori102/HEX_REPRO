import pandas as pd
import numpy as np
import h5py

# ======================
# CONFIG
# ======================
csv_path = "/home/26w_kyr/hex_2.csv"            # 입력 CSV
output_file = "./hex_2.h5"     # 출력 h5

# ======================
# LOAD CSV
# ======================
df = pd.read_csv(csv_path)

# ----------------------
# marker 컬럼 자동 추출
# ----------------------
marker_cols = [c for c in df.columns if c.endswith("_pred")]
print(f"[INFO] Found {len(marker_cols)} marker columns")

# 좌표 컬럼
coord_cols = ["x", "y"]
assert all(c in df.columns for c in coord_cols), "x,y 컬럼이 없습니다"

# ======================
# BUILD H5 CONTENT
# ======================
coords = df[coord_cols].values.astype(np.int32)           # (N, 2)
codex_pred = df[marker_cols].values.astype(np.float16)    # (N, C)

print(f"[INFO] coords shape: {coords.shape}")
print(f"[INFO] codex_prediction shape: {codex_pred.shape}")

# ======================
# SAVE H5
# ======================
with h5py.File(output_file, "w") as f:
    f.create_dataset("coords", data=coords)
    f.create_dataset("codex_prediction", data=codex_pred)

print(f"✅ Saved h5 file: {output_file}")
