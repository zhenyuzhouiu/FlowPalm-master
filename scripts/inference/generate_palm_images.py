import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli import (
    build_diffusion_from_checkpoint,
    ensure_dir,
    list_flow_files,
    load_stacked_flows,
    parse_args_with_config,
)
from utils.noise_warping import NoiseWarper, noise_to_xyωc, warp_image
from data.datasets import PalmDataset


def main():
    parser = argparse.ArgumentParser(description="Generate palm images from crease maps and flow fields.")
    parser.add_argument("--output_dir", "--save_dir", dest="output_dir", type=str, default="outputs/palm_samples")
    parser.add_argument("--crease_dir", type=str, help="Directory of crease images.")
    parser.add_argument("--flow_dir", type=str, help="Directory of flow .npz or .npy files.")
    parser.add_argument("--ckpt", type=str, help="Path to diffusion checkpoint.")
    parser.add_argument("--per_ID_num", type=int, default=40, help="Number of images per identity.")
    parser.add_argument("--uncondition_t", type=float, default=0.25, help="Unconditional guidance scale.")
    parser.add_argument("--clean_t", type=float, default=0.5, help="Clean trajectory guidance scale.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker count.")
    args = parse_args_with_config(parser)

    output_dir = ensure_dir(args.output_dir)

    accelerator = Accelerator()
    device = accelerator.device
    num_processes = accelerator.num_processes
    rank = accelerator.process_index

    dataset = PalmDataset(
        crease_root=args.crease_dir,
        palm_root=None,
        flow_path=None,
        per_ID_num=args.per_ID_num,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_ID_num,
        shuffle=False,
        num_workers=args.num_workers,
    )
    dataloader = accelerator.prepare(dataloader)

    diffusion = build_diffusion_from_checkpoint(
        device=device,
        ckpt_path=args.ckpt,
        sampling_timesteps=250,
        ddim_sampling_eta=0.0,
        strict=True,
        compile_model=False,
    )
    diffusion = accelerator.prepare(diffusion)
    sampling_model = accelerator.unwrap_model(diffusion)

    warper = NoiseWarper(c=3, h=256, w=256, device=device)
    flow_files = list_flow_files(args.flow_dir)

    for i, batch in enumerate(dataloader):
        crease = batch["crease"].to(device)
        batch_size = crease.size(0)

        flow_path = random.choice(flow_files)
        flow_np = load_stacked_flows(flow_path)[:batch_size]
        flow_tensor = torch.from_numpy(flow_np).float().to(device)

        warp_crease = warp_image(crease, flow_tensor)

        gaussian_noise = torch.randn(1, 3, 256, 256, device=device)
        consistent_noise = gaussian_noise.repeat(batch_size, 1, 1, 1)

        warp_noise = []
        for flow_item, noise_item in zip(flow_tensor, consistent_noise):
            dx, dy = flow_item
            warper._state = noise_to_xyωc(noise_item)
            warp_noise.append(warper(dx, dy).noise)
        warp_noise = torch.stack(warp_noise, dim=0)

        start_noise = torch.randn_like(warp_noise)
        fake_palm, _ = sampling_model.sample(
            batch_size=batch_size,
            crease=warp_crease,
            start_noise=start_noise,
            warp_noise=warp_noise,
            guidance_scale=1,
            uncondition_t=args.uncondition_t,
            clean_t=args.clean_t,
        )

        identity_index = i * num_processes + rank
        sample_dir = output_dir / f"{identity_index:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        for j, image in enumerate(fake_palm):
            save_image(image, sample_dir / f"{j:03d}.jpg")

        print(f"[GPU {rank}] Saved batch {i} as ID index {identity_index:04d}")


if __name__ == "__main__":
    main()
