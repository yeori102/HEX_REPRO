# HEX_REPRO

This repository reproduces the HEX pipeline for predicting protein expression from H&E whole-slide images (WSIs) and generating Virtual CODEX visualizations.

## How to run
1. **`extract_wsi_thumbnail.py`**
   * Extracts a low-resolution thumbnail image from a WSI (`.svs`)
   * Used for slide-level visualization and sanity checks
2. **`extract_coordinate.py`**
   * Identifies tissue-containing regions in the WSI
   * Extracts top-left patch coordinates (`x, y`) and saves them to a CSV file
3. **`extract_he_patch.py`**
   * Uses the coordinate CSV to extract H&E patch images
   * All patches are extracted at **level 0 (full resolution)** from the WSI
4. **`hex/hex_infer.py`**
   * Runs the HEX model on the extracted H&E patches
   * Predicts patch-level protein expression
   * Preserves the original `x, y` coordinates in the output CSV
5. **`hex/vc.py`**
   * Converts HEX prediction results into **Virtual CODEX-style channel-wise numeric maps (`.npz`)**
6. **`hex/vc_render.py`**
   * Renders the Virtual CODEX maps into visual outputs
   * Generates per-channel grayscale images, pseudo-colored images, and composite visualizations
7. **`hex/csv_to_codex_h5.py`**
8. **`mica/codex_h5_png2fea.py`**