import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.optical_flow import RaftOpticalFlow
from models.recognition.mobile_face_net import MobileFaceNet
from utils.cli import ensure_dir, parse_args_with_config, require_existing_dir, require_existing_file, str2bool


rec_transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def demo_flow(args):
    with torch.no_grad():
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        optical_flow = RaftOpticalFlow(device)
        rec_net = MobileFaceNet().to(device)
        reg_ckpt = torch.load(str(require_existing_file(args.recognize_ckpt, "Recognition checkpoint")), map_location=device)
        rec_net.load_state_dict(reg_ckpt['net'])
        rec_net.eval()

        root_dir = require_existing_dir(args.input_dir, "Input image directory")
        save_dir = ensure_dir(args.output_dir)
        img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        subdirs = sorted([dirpath for dirpath, _, _ in os.walk(root_dir) if dirpath != str(root_dir)])
        sub_idx = 0

        for _, subdir in enumerate(subdirs, 1):
            if args.max_identities is not None and sub_idx >= args.max_identities:
                print(f"Reached max_identities={args.max_identities}, stopping.")
                break
            images = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.lower().endswith(img_exts)]

            if len(images) > 5:
                num_pairs = min(args.max_pairs, len(images) * (len(images) - 1) // 2)
                chosen_pairs = set()

                print(f"\n[{sub_idx}] folder: {subdir}, images: {len(images)}, planned pairs: {num_pairs}")

                while len(chosen_pairs) < num_pairs:
                    path1, path2 = random.sample(images, 2)
                    pair = tuple(sorted([path1, path2]))
                    if pair not in chosen_pairs:
                        chosen_pairs.add(pair)

                flows = []
                for path1, path2 in chosen_pairs:
                    from_image = Image.open(path1).convert("RGB")
                    to_image = Image.open(path2).convert("RGB")

                    flow = optical_flow.get_flow_map(from_image, to_image)
                    warped_tensor, _ = optical_flow.warp_image(from_image, flow)
                    flow_np = flow.cpu().numpy()

                    to_tensor = rec_transformer(to_image).unsqueeze(0).to(device)
                    feat_to = rec_net(to_tensor)

                    warped_resized = F.interpolate(warped_tensor, size=(224, 224), mode='bilinear', align_corners=False)
                    warped_resized = (warped_resized - 0.5) / 0.5
                    feat_warp = rec_net(warped_resized)

                    cos_sim = F.cosine_similarity(feat_to, feat_warp).item()
                    abnormal_ratio = optical_flow.check_flow_continuity(flow, threshold=5.0)

                    if cos_sim < args.cosine_threshold or abnormal_ratio > args.flow_threshold:
                        print(f"Bad optical flow (cos_sim={cos_sim:.2f}, abnormal_ratio={abnormal_ratio:.3f}); reset to zero flow")
                        flow_np = np.zeros_like(flow_np)

                    flows.append(flow_np)

                stacked = np.stack(flows, axis=0)
                dtype = np.float16 if args.flow_dtype == "float16" else np.float32
                stacked = stacked.astype(dtype, copy=False)

                if args.flow_compress:
                    save_path = save_dir / f"flows_{sub_idx:04d}.npz"
                    np.savez_compressed(save_path, flows=stacked)
                else:
                    save_path = save_dir / f"flows_{sub_idx:04d}.npy"
                    np.save(save_path, stacked)
                print(f"Saved {len(flows)} flows to {save_path}")
                sub_idx += 1
            else:
                print(f"Skipping folder {subdir}: not enough images")


def build_parser():
    parser = argparse.ArgumentParser(description="Precompute optical flow files for FlowPalm.")
    parser.add_argument("--input_dir", type=str, default=None, help="Root directory containing identity subfolders.")
    parser.add_argument("--output_dir", type=str, default="outputs/flow", help="Directory to save flow files (.npz or .npy).")
    parser.add_argument("--recognize_ckpt", type=str, default=None, help="Path to recognition checkpoint.")
    parser.add_argument("--max_pairs", type=int, default=40, help="Maximum image pairs per identity folder.")
    parser.add_argument(
        "--max_identities",
        type=int,
        default=None,
        help="Maximum number of identity folders that successfully write a flow file (>5 images each). Omit or null for no limit.",
    )
    parser.add_argument("--cosine_threshold", type=float, default=0.4, help="Minimum cosine similarity to keep a flow.")
    parser.add_argument("--flow_threshold", type=float, default=0.01, help="Maximum abnormal ratio to keep a flow.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override such as cuda or cpu.")
    parser.add_argument(
        "--flow_compress",
        type=str2bool,
        default=True,
        help="If true, write zlib-compressed .npz with key 'flows' (much smaller on disk).",
    )
    parser.add_argument(
        "--flow_dtype",
        type=str,
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype; float16 is ~2x smaller with minor precision loss. Loaded as float32 for inference.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parse_args_with_config(parser)
    missing = [name for name, val in (("--input_dir", args.input_dir), ("--recognize_ckpt", args.recognize_ckpt)) if val is None]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
    demo_flow(args)
