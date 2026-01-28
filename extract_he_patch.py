import os
import pandas as pd
import openslide
from tqdm import tqdm

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
    os.makedirs(slide_output_dir, exist_ok=True)

    # If slide_index is missing in CSV, substitute with index
    has_slide_index = "slide_index" in label_df.columns

    for i, row in tqdm(label_df.iterrows(), total=len(label_df), desc="Saving patches"):
        x = int(row["x"])
        y = int(row["y"])

        patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")

        patch_id = int(row["slide_index"]) if has_slide_index else i
        patch_path = os.path.join(slide_output_dir, f"{patch_id}.png")
        patch.save(patch_path)

    slide.close()
    print("Done:", slide_output_dir)

def main():
    csv_path = "./TCGA-MN-A4N4-01Z-00-DX2.9550732D-8FB1-43D9-B094-7C0CD310E9C0.csv" # input extract_coordinate.py result file path
    he_dir = "/data-hdd/home/shared/TCGA/WSI/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c" # input svs directory name
    output_dir = "."  # Create a folder named SlideID at the current location
    patch_size = 224

    process_one_csv(csv_path, he_dir, output_dir, patch_size)

if __name__ == "__main__":
    main()
