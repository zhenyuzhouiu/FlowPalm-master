import numpy as np
import rp
import torch
import torch.nn.functional as F
import torchvision.transforms
from PIL import Image
from torchvision.models.optical_flow import raft_large


class RaftOpticalFlow:
    """RAFT large (torchvision) wrapper for pairwise optical flow and image warping."""

    def __init__(self, device):
        self.device = device
        self.model = raft_large(pretrained=True, progress=False).to(device)
        self.model.eval()

    def _preprocess_image(self, image):
        assert rp.is_image(image)
        image = rp.as_float_image(rp.as_rgb_image(image))

        new_height, new_width = 256, 256
        transform_ops = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.5),
            torchvision.transforms.Resize(size=(new_height, new_width)),
        ])

        output = transform_ops(image)[None].to(self.device).float()
        assert output.shape == (1, 3, new_height, new_width)
        return output

    def get_flow_map(self, from_image, to_image):
        assert rp.is_image(from_image)
        assert rp.is_image(to_image)
        with torch.no_grad():
            img1 = self._preprocess_image(from_image)
            img2 = self._preprocess_image(to_image)
            list_of_flows = self.model(img1, img2)
            output_flow = list_of_flows[-1][0]
        return output_flow

    def create_coordinate_grid(self, height, width, device):
        x = torch.arange(0, width, dtype=torch.float32, device=device)
        y = torch.arange(0, height, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=0)

    def warp_image(self, image, flow, mode='bilinear'):
        assert rp.is_image(image)
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        transform = torchvision.transforms.ToTensor()
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        height, width = flow.shape[1], flow.shape[2]

        coordinate_grid = self.create_coordinate_grid(height, width, self.device)
        new_coordinates = coordinate_grid - flow
        new_coordinates[0] = 2.0 * new_coordinates[0] / (width - 1) - 1.0
        new_coordinates[1] = 2.0 * new_coordinates[1] / (height - 1) - 1.0

        sampling_grid = new_coordinates.permute(1, 2, 0).unsqueeze(0)
        warped_tensor = F.grid_sample(
            image_tensor,
            sampling_grid,
            mode=mode,
            padding_mode='border',
            align_corners=True,
        )

        warped_image = warped_tensor[0].permute(1, 2, 0).cpu().numpy()
        warped_image = np.clip(warped_image, 0, 1)
        return warped_tensor, warped_image

    def check_flow_continuity(self, flow, threshold=2.0):
        flow_np = flow.detach().cpu().numpy()
        u, v = flow_np[0], flow_np[1]
        dudx = np.gradient(u, axis=1)
        dudy = np.gradient(u, axis=0)
        dvdx = np.gradient(v, axis=1)
        dvdy = np.gradient(v, axis=0)
        grad_mag = np.sqrt(dudx**2 + dudy**2 + dvdx**2 + dvdy**2)
        abnormal_mask = grad_mag > threshold
        abnormal_ratio = abnormal_mask.sum() / abnormal_mask.size
        return abnormal_ratio
