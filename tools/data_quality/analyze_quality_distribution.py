import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli import ensure_dir, parse_args_with_config


def compute_glcm_features(gray):
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        symmetric=True,
        normed=True,
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


def compute_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture_contrast, energy, homogeneity = compute_glcm_features(gray)
    freq_energy = compute_frequency_energy(gray)
    texture_complexity = 1 - homogeneity
    return lap_var, texture_contrast, freq_energy, texture_complexity


def analyze_distribution(args):
    input_dir = args.input_dir
    output_dir = ensure_dir(args.output_dir)
    resize_shape = (args.image_size, args.image_size)

    sharpness, contrast, freq_energy, complexity = [], [], [], []

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                continue
            fpath = os.path.join(root, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)
            lap_var, contr, freq, comp = compute_features(img)
            sharpness.append(lap_var)
            contrast.append(contr)
            freq_energy.append(freq)
            complexity.append(comp)

    features = {
        "Sharpness": np.array(sharpness),
        "Local Contrast": np.array(contrast),
        "High-Freq Energy": np.array(freq_energy),
        "Texture Complexity": np.array(complexity),
    }

    print(f"\nSummarized {len(sharpness)} images.")
    print("============== Feature statistics and percentile thresholds (50/60/70/80%) ==============")

    percentiles = [50, 60, 70, 80]
    colors = ['green', 'orange', 'purple', 'red']

    np.save(output_dir / "all_features.npy", features, allow_pickle=True)

    plt.figure(figsize=(12, 6))
    thresholds_dict = {}

    for i, (name, vals) in enumerate(features.items()):
        mean = np.mean(vals)
        std = np.std(vals)
        p_values = {p: np.percentile(vals, p) for p in percentiles}
        thresholds_dict[name] = p_values

        print(f"\n{name:20s}: mean = {mean:.3f}, std = {std:.3f}")
        for p, v in p_values.items():
            print(f"    {p}% percentile = {v:.5f}")

        plt.subplot(2, 2, i + 1)
        plt.hist(vals, bins=60, color='gray', edgecolor='black')
        plt.axvline(mean, color='blue', linestyle='--', label=f"Mean={mean:.2f}")
        for c, p in zip(colors, percentiles):
            plt.axvline(p_values[p], color=c, linestyle='-', label=f"P{p}={p_values[p]:.2f}")
        plt.title(name)
        plt.legend(fontsize=7)
        plt.grid(alpha=0.3)
        plt.xlabel(name)
        plt.ylabel("Count")

        np.save(output_dir / f"{name.replace(' ', '_')}_values.npy", vals)
        np.save(output_dir / f"{name.replace(' ', '_')}_thresholds.npy", p_values)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_distribution_multi_thresholds.png", dpi=300)
    plt.show()

    np.save(output_dir / "all_thresholds.npy", thresholds_dict, allow_pickle=True)
    print(f"\nSaved feature distributions and thresholds to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze feature distributions for FlowPalm quality filtering.")
    parser.add_argument("--input_dir", type=str, default=None, help="Root directory of source ROI images.")
    parser.add_argument("--output_dir", type=str, default="outputs/feature_stats", help="Directory for saved statistics.")
    parser.add_argument("--image_size", type=int, default=256, help="Resize images to a square size before analysis.")
    args = parse_args_with_config(parser)
    if args.input_dir is None:
        parser.error("the following arguments are required: --input_dir")
    analyze_distribution(args)
