import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli import parse_args_with_config, require_existing_file


def main(args):
    output_dir = args.output_dir
    features = np.load(str(require_existing_file(os.path.join(output_dir, "all_features.npy"), "all_features.npy")), allow_pickle=True).item()

    selected_features = ['Sharpness', 'Local Contrast', 'High-Freq Energy', 'Texture Complexity']
    features = {k: features[k] for k in selected_features}

    data_dict = {}
    for name, vals in features.items():
        data_dict[name] = vals

    df = pd.DataFrame(data_dict)
    corr_pearson = df.corr(method='pearson')
    print("=" * 60)
    print("Pearson Correlation Matrix:")
    print("=" * 60)
    print(corr_pearson.round(4))
    print()

    corr_spearman = df.corr(method='spearman')
    print("=" * 60)
    print("Spearman Correlation Matrix:")
    print("=" * 60)
    print(corr_spearman.round(4))
    print()

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 14,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.unicode_minus': False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('white')

    selected_features_wrapped = [
        'Sharpness',
        'Local\nContrast',
        'High-Freq\nEnergy',
        'Texture\nComplexity'
    ]

    ax1 = axes[0]
    im1 = ax1.imshow(corr_pearson, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(np.arange(len(selected_features_wrapped)))
    ax1.set_yticks(np.arange(len(selected_features_wrapped)))
    ax1.set_xticklabels(selected_features_wrapped, rotation=0, ha='center')
    ax1.set_yticklabels(selected_features_wrapped)
    ax1.set_title('Pearson Correlation', fontweight='bold', pad=10)

    for i in range(len(selected_features_wrapped)):
        for j in range(len(selected_features_wrapped)):
            val = corr_pearson.iloc[i, j]
            ax1.text(
                j, i, f'{val:.3f}',
                ha="center", va="center",
                color="white" if abs(val) > 0.5 else "black",
                fontsize=9, fontweight='bold',
            )

    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Correlation Coefficient', rotation=270, labelpad=20)

    ax2 = axes[1]
    im2 = ax2.imshow(corr_spearman, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(np.arange(len(selected_features_wrapped)))
    ax2.set_yticks(np.arange(len(selected_features_wrapped)))
    ax2.set_xticklabels(selected_features_wrapped, rotation=0, ha='center')
    ax2.set_yticklabels(selected_features_wrapped)
    ax2.set_title('Spearman Correlation', fontweight='bold', pad=10)

    for i in range(len(selected_features_wrapped)):
        for j in range(len(selected_features_wrapped)):
            val = corr_spearman.iloc[i, j]
            ax2.text(
                j, i, f'{val:.3f}',
                ha="center", va="center",
                color="white" if abs(val) > 0.5 else "black",
                fontsize=9, fontweight='bold',
            )

    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Correlation Coefficient', rotation=270, labelpad=20)

    plt.tight_layout()
    save_path_png = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(save_path_png, dpi=400, bbox_inches='tight', facecolor='white')
    save_path_pdf = os.path.join(output_dir, "correlation_heatmap.pdf")
    plt.savefig(save_path_pdf, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot correlation heatmaps from saved feature statistics.")
    parser.add_argument("--output_dir", type=str, default="outputs/feature_stats", help="Directory containing all_features.npy.")
    main(parse_args_with_config(parser))
