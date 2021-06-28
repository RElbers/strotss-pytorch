from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torch.nn import functional as F

# INTERPOLATION_MODE = 'nearest'
# INTERPOLATION_MODE = 'bicubic'
INTERPOLATION_MODE = 'bilinear'


def rgb_to_yuv(rgb):
    shape = rgb.shape
    w = [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.0],
         [112.0, -93.786, -18.214]]
    b = [16.0, 128.0, 128.0]
    w = normalize(torch.tensor(w, device=rgb.device))
    b = normalize(torch.tensor(b, device=rgb.device))

    rgb = rgb.flatten(-2, -1).transpose(-2, -1)
    yuv = rgb @ w + b
    yuv = yuv.transpose(-2, -1).view(shape)

    return yuv


def downscale(x, factor):
    if factor == 1:
        return x

    scaled = F.interpolate(x, scale_factor=1 / factor, mode=INTERPOLATION_MODE, align_corners=False)
    return scaled


def resize_like(x, other):
    return resize(x, (other.size(-2), other.size(-1)))


def resize(x, size):
    if x.shape[:-2] == size:
        return x

    return F.interpolate(x, size, mode=INTERPOLATION_MODE, align_corners=False)


def load(path: Union[Path, str]):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"File does not exist: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img)


def save(path: Union[Path, str], img, params=None):
    img = np.ascontiguousarray(img, np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img, params=params)
    return img


def to_torch(img):
    img = torch.from_numpy(img).float()
    img = torch.einsum('hwc->chw', img)
    return img.to('cuda')


def from_torch(img):
    img = torch.einsum('chw->hwc', img)
    img = img.detach().cpu().numpy()
    return img.astype(np.uint8)


def normalize(x):
    return x / 255.0


def denormalize(x):
    x = torch.clamp(x, 0, 1)
    return x * 255.0


def resize_long_side_to(img, size):
    h = img.shape[-2]
    w = img.shape[-1]

    new_size = (int(size * (h / w)), size) if w > h else (size, int(size * (w / h)))
    y = resize(img.unsqueeze(0), new_size).squeeze(0)
    return y