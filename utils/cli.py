import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.diffusion_model.diffusion import GaussianDiffusion, Unet


def str2bool(value):
    if isinstance(value, bool):
        return value

    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args_with_config(parser):
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file. CLI arguments override config values.",
    )

    config_args, _ = parser.parse_known_args()
    if config_args.config is not None:
        config_path = Path(config_args.config).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        if not isinstance(config, dict):
            raise ValueError(f"Config file must contain a JSON object: {config_path}")

        parser.set_defaults(**config)

    return parser.parse_args()


def resolve_path(path_value):
    return Path(path_value).expanduser().resolve()


def ensure_dir(path_value):
    path = resolve_path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_existing_dir(path_value, description):
    path = resolve_path(path_value)
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def require_existing_file(path_value, description):
    path = resolve_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def list_flow_files(flow_dir):
    flow_root = require_existing_dir(flow_dir, "Flow directory")
    flow_files = sorted(
        list(flow_root.glob("*.npz")) + list(flow_root.glob("*.npy")),
        key=lambda p: p.name,
    )
    if not flow_files:
        raise FileNotFoundError(f"No .npz or .npy flow files found in: {flow_root}")
    return [str(path) for path in flow_files]


def load_stacked_flows(path_value):
    """
    Load stacked flow tensor [N, 2, H, W] from .npz (key 'flows') or legacy .npy.
    Always returns float32 for downstream torch / warping.
    """
    path = resolve_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Flow file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as z:
            if "flows" not in z.files:
                raise ValueError(f"Expected array key 'flows' in {path}, found {z.files}")
            arr = z["flows"]
    elif suffix == ".npy":
        arr = np.load(path)
    else:
        raise ValueError(f"Unsupported flow file type: {path}")
    return np.asarray(arr, dtype=np.float32)


def build_diffusion_from_checkpoint(
    *,
    device,
    ckpt_path,
    sampling_timesteps,
    ddim_sampling_eta=0.0,
    strict=True,
    compile_model=False,
):
    checkpoint_path = require_existing_file(ckpt_path, "Checkpoint")

    unet = Unet(dim=64, dim_mults=(1, 1, 2, 3, 4), flash_attn=False).to(device)
    diffusion = GaussianDiffusion(
        unet,
        image_size=256,
        timesteps=1000,
        sampling_timesteps=sampling_timesteps,
        ddim_sampling_eta=ddim_sampling_eta,
    ).to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model_keys = diffusion.state_dict().keys()
    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    diffusion.load_state_dict(state_dict, strict=strict)

    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("Current PyTorch version does not support torch.compile.")
        diffusion.model = torch.compile(diffusion.model, mode="reduce-overhead")

    return diffusion
