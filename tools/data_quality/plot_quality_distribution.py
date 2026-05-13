import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli import parse_args_with_config, require_existing_file


def main(args):
    output_dir = args.output_dir
    features = np.load(str(require_existing_file(os.path.join(output_dir, "all_features.npy"), "all_features.npy")), allow_pickle=True).item()

    selected_features = ['Sharpness', 'Local Contrast', 'High-Freq Energy', 'Texture Complexity']
    features = {k: features[k] for k in selected_features}

    clip_percentile = {
        'Sharpness': 90,
        'Local Contrast': 95,
        'High-Freq Energy': 100,
        'Texture Complexity': 100
    }

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'axes.unicode_minus': False,
        'text.usetex': False,
    })

    fig = plt.figure(figsize=(7.2, 5.4))
    fig.patch.set_facecolor('white')

    for i, (name, vals) in enumerate(features.items()):
        ax = plt.subplot(2, 2, i + 1)
        pmax = np.percentile(vals, clip_percentile[name])
        vals_clip = vals[vals <= pmax]
        p20 = np.percentile(vals_clip, 10)

        ax.hist(
            vals_clip,
            bins=60,
            color='#4A90E2',
            edgecolor='white',
            alpha=0.85,
            density=True,
            linewidth=0.3,
        )
        ax.axvline(
            p20,
            color='#E74C3C',
            linestyle='--',
            linewidth=2.0,
            label=f'$P_{{10}}$ = {p20:.2f}',
            alpha=0.9,
            zorder=10,
        )
        ax.axvspan(vals_clip.min(), p20, alpha=0.15, facecolor='#E74C3C', label='Low Quality')
        ax.set_title(name, fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel("Value", fontsize=10, labelpad=4)

        legend = ax.legend(
            fontsize=7.5,
            frameon=True,
            loc='upper right',
            framealpha=1.0,
            edgecolor='#BDC3C7',
            fancybox=False,
        )
        legend.get_frame().set_linewidth(0.8)

        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, color='gray')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(1.0)
        ax.tick_params(axis='x', direction='in', length=4, width=1.0, colors='black')
        ax.set_yticks([])
        ax.set_yticklabels([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(output_dir, "feature_distribution_scientific.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')

    save_path_pdf = os.path.join(output_dir, "feature_distribution_scientific.pdf")
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')

    plt.close()

    print("Saved publication-style figures:")
    print(f"   PNG (600 DPI): {save_path}")
    print(f"   PDF (vector): {save_path_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot publication-style feature distributions from saved numpy stats.")
    parser.add_argument("--output_dir", type=str, default="outputs/feature_stats", help="Directory containing all_features.npy.")
    main(parse_args_with_config(parser))
