import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli import ensure_dir, parse_args_with_config, require_existing_file


def compute_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture_contrast, energy, homogeneity = compute_glcm_features(gray)
    texture_complexity = 1 - homogeneity
    freq_energy = compute_frequency_energy(gray)
    return lap_var, texture_contrast, freq_energy, texture_complexity


def compute_glcm_features(gray):
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        symmetric=True,
        normed=True
    )
    contrast = graycoprops(glcm, 'contrast').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    return contrast, energy, homogeneity


def compute_frequency_energy(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) ** 2
    h, w = gray.shape

    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_thresh = min(h, w) * 0.25
    high_mask = radius > r_thresh

    eps = 1e-8
    energy_high = np.mean(np.log(magnitude[high_mask] + eps))
    energy_low = np.mean(np.log(magnitude[~high_mask] + eps))
    return energy_high - energy_low


def main(args):
    input_dir = args.input_dir
    output_high = ensure_dir(args.output_high)
    output_low = ensure_dir(args.output_low)
    resize_shape = (args.image_size, args.image_size)

    features = np.load(str(require_existing_file(args.features_file, "Feature statistics file")), allow_pickle=True).item()
    thresholds = {k: np.percentile(v, args.percentile) for k, v in features.items()}

    sharp_thr = thresholds["Sharpness"]
    contrast_thr = thresholds["Local Contrast"]
    texture_thr = thresholds["Texture Complexity"]
    freq_thr = thresholds["High-Freq Energy"]

    print("============== Thresholds in use ==============")
    print(f"Percentile = P{args.percentile}")
    print(f"Sharpness ≥ {sharp_thr:.2f}")
    print(f"Local Contrast ≥ {contrast_thr:.2f}")
    print(f"Texture Complexity ≥ {texture_thr:.2f}")
    print(f"High-Freq Energy ≥ {freq_thr:.2f}")

    high_count, low_count = 0, 0

    for root, _, files in os.walk(input_dir):
        for fname in tqdm(files, desc="Processing"):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                continue

            fpath = os.path.join(root, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue

            img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)
            lap_var, local_contrast, freq_energy, texture_complexity = compute_features(img)

            failed = 0
            if lap_var < sharp_thr:
                failed += 1
            if local_contrast < contrast_thr:
                failed += 1
            if texture_complexity < texture_thr:
                failed += 1
            if freq_energy < freq_thr:
                failed += 1

            is_high = failed <= args.max_failed_metrics
            rel_path = os.path.relpath(fpath, input_dir)
            base, ext = os.path.splitext(rel_path)

            fname_new = (
                f"{base}_S{lap_var:.1f}_C{local_contrast:.1f}"
                f"_H{freq_energy:.1f}_T{texture_complexity:.3f}_{ext}"
            )

            if is_high:
                save_path = output_high / fname_new
                high_count += 1
            else:
                save_path = output_low / fname_new
                low_count += 1

            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), img)

    print(f"\nFiltering done: {high_count} high-quality, {low_count} low-quality images")
    print(f"High-quality images saved to: {output_high}")
    print(f"Low-quality images saved to: {output_low}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split images into high and low quality subsets.")
    parser.add_argument("--input_dir", type=str, default=None, help="Source ROI image directory.")
    parser.add_argument("--output_high", type=str, default="outputs/ROI_high_quality")
    parser.add_argument("--output_low", type=str, default="outputs/ROI_low_quality")
    parser.add_argument("--features_file", type=str, default=None, help="Path to all_features.npy generated earlier.")
    parser.add_argument("--percentile", type=float, default=10.0, help="Percentile used as the filtering threshold.")
    parser.add_argument("--max_failed_metrics", type=int, default=1, help="Maximum number of metrics allowed below threshold.")
    parser.add_argument("--image_size", type=int, default=256)
    args = parse_args_with_config(parser)
    missing = [
        name
        for name, val in (("--input_dir", args.input_dir), ("--features_file", args.features_file))
        if val is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
    main(args)
