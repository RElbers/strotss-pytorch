from torch import nn

from util import downscale, resize_like


class LaplacianPyramid(nn.Module):
    def __init__(self, image, depth):
        super().__init__()
        self.depth = depth
        self.image = image
        pyramid = LaplacianPyramid._init_pyramid(image, depth)
        pyramid = nn.ParameterList([nn.Parameter(l) for l in pyramid])
        self.pyramid = pyramid

    @staticmethod
    def _init_pyramid(image, depth):
        pyramid = []
        current = image
        for i in range(depth):
            if i == depth - 1:
                laplacian = current
            else:
                subsampled = downscale(current, 2)
                upscaled = resize_like(subsampled, current)
                laplacian = current - upscaled
                current = subsampled

            pyramid.append(laplacian)

        return pyramid

    def reconstruct(self):
        img = self.pyramid[-1]
        for laplacian in reversed(self.pyramid[:-1]):
            img = resize_like(img, laplacian)
            img += laplacian

        return img


def laplacian(x):
    x_small = downscale(x, 2)
    x_small = resize_like(x_small, x)
    return x - x_small
