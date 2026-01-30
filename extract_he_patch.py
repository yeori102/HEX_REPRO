import os
import pandas as pd
import openslide
from tqdm import tqdm
from PIL import Image


def process_one_csv(csv_path, he_dir, output_dir, patch_size):
    label_df = pd.read_csv(csv_path)

    # Use the CSV filename (stem) as the slide ID
    he_id = os.path.splitext(os.path.basename(csv_path))[0]
    he_path = os.path.join(he_dir, f"{he_id}.svs")

    if not os.path.exists(he_path):
        print(f"[ERROR] H&E image not found: {he_path}")
        return

    print(f"Processing {he_id}")
    slide = openslide.OpenSlide(he_path)

    slide_output_dir = os.path.join(output_dir, he_id)

    # ⚠️ 패치 전부 다시 만들 거면 기존 폴더 삭제 권장
    if os.path.exists(slide_output_dir):
        print(f"[INFO] Removing existing patch dir: {slide_output_dir}")
        for f in os.listdir(slide_output_dir):
            os.remove(os.path.join(slide_output_dir, f))
    else:
        os.makedirs(slide_output_dir, exist_ok=True)

    has_slide_index = "slide_index" in label_df.columns

    for i, row in tqdm(label_df.iterrows(),
                       total=len(label_df),
                       desc="Saving patches"):

        x = int(row["x"])
        y = int(row["y"])

        patch_id = int(row["slide_index"]) if has_slide_index else i
        patch_path = os.path.join(slide_output_dir, f"{patch_id}.png")
        tmp_path = patch_path + ".tmp"

        try:
            patch = slide.read_region(
                (x, y),
                0,
                (patch_size, patch_size)
            ).convert("RGB")

            # 임시 파일에 저장
            patch.save(tmp_path, format="PNG", compress_level=3)

            # 무결성 체크
            Image.open(tmp_path).verify()

            # atomic replace
            os.replace(tmp_path, patch_path)

        except Exception as e:
            print(f"[ERROR] patch {patch_id} failed: {e}")

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    slide.close()
    print("Done:", slide_output_dir)


def main():
    csv_path = "/data-hdd/home/shared/TCGA/WSI/a1b3946e-e734-46a8-ae64-43ec9a1bea74/TCGA-49-4505-01Z-00-DX4.623c4278-fc3e-4c80-bb4d-000e24fbb1c2.csv"

    he_dir = "/data-hdd/home/shared/TCGA/WSI/a1b3946e-e734-46a8-ae64-43ec9a1bea74"

    output_dir = "/data-hdd/home/shared/TCGA/WSI/a1b3946e-e734-46a8-ae64-43ec9a1bea74"

    patch_size = 224

    process_one_csv(csv_path, he_dir, output_dir, patch_size)


if __name__ == "__main__":
    main()
