import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.diffusion_model.diffusion import GaussianDiffusion, Trainer, Unet
from utils.cli import parse_args_with_config, str2bool


def main():
    parser = argparse.ArgumentParser(description="Train the FlowPalm crease-conditioned diffusion model.")
    parser.add_argument("--train_data", type=str, default=None, help="Directory of training images.")
    parser.add_argument(
        "--sample_crease_dir",
        type=str,
        default=None,
        help="Optional crease directory for periodic qualitative sampling during training.",
    )
    parser.add_argument("--results_dir", type=str, default="outputs/train_crease")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--train_lr", type=float, default=8e-5)
    parser.add_argument("--train_num_steps", type=int, default=200000)
    parser.add_argument("--gradient_accumulate_every", type=int, default=2)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--calculate_fid", type=str2bool, default=False)
    parser.add_argument("--save_and_sample_every", type=int, default=5000)
    parser.add_argument("--num_samples", type=int, default=25)
    args = parse_args_with_config(parser)
    if args.train_data is None:
        parser.error("the following arguments are required: --train_data")

    model = Unet(dim=64, dim_mults=(1, 1, 2, 3, 4), flash_attn=False)
    diffusion = GaussianDiffusion(
        model,
        image_size=256,
        timesteps=1000,
        sampling_timesteps=250,
    )

    trainer = Trainer(
        diffusion,
        args.train_data,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        amp=args.amp,
        calculate_fid=args.calculate_fid,
        save_and_sample_every=args.save_and_sample_every,
        num_samples=args.num_samples,
        results_folder=args.results_dir,
        sample_crease_dir=args.sample_crease_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
