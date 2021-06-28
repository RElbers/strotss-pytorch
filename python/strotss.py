import math

from torch import nn
from torch.optim import RMSprop
from tqdm import tqdm

import util
from feature_extraction import VGGFeatureExtractor
from laplacian import LaplacianPyramid, laplacian
from loss import *
from sampling import sample_points
from util import rgb_to_yuv, downscale, resize_like


class STROTSS(nn.Module):
    def __init__(self,
                 prior='content',
                 min_size=64,
                 n_update_steps=200,
                 n_points=1024,
                 alpha=16.0,
                 lr=0.002):
        super().__init__()

        self.prior = prior
        self.min_size = min_size
        self.n_update_steps = n_update_steps
        self.n_points = n_points
        self.alpha = alpha
        self.lr = lr

        self.feature_extractor = VGGFeatureExtractor()
        self.on_iteration_end = None

    def __call__(self, content, style):
        # Assume non-batched image (shape=[3,H,W])
        assert content.dim() == style.dim() == 3
        alpha, lr = self.alpha, self.lr

        content, style, output = self.pre_process(content, style)

        long_size = max(content.shape[-1], content.shape[-2])
        n_scales = math.ceil(math.log(long_size, 2)) - int(math.log(self.min_size, 2))
        pyramid_depth = n_scales + 1
        for i in tqdm(reversed(range(n_scales))):
            # Start with smallest size
            factor = 2 ** i

            content_small = downscale(content, factor)
            style_small = downscale(style, factor)

            if i == 0:
                lr /= 2
                output = resize_like(output, content_small)
            else:
                output = resize_like(output, content_small) + laplacian(content_small)

            output = self.style_transfer(content_small,
                                         style_small,
                                         output,
                                         pyramid_depth=pyramid_depth,
                                         alpha=alpha,
                                         lr=lr)
            alpha /= 2
        return self.post_process(output)

    def style_transfer(self, content, style, output, pyramid_depth, alpha, lr):
        style_yuv = rgb_to_yuv(style)

        laplacian_pyramid = LaplacianPyramid(output, depth=pyramid_depth)
        laplacian_pyramid.to(content.device)
        optimizer = RMSprop(laplacian_pyramid.parameters(), lr=lr)

        pbar = tqdm(range(self.n_update_steps))
        for i in pbar:
            output = laplacian_pyramid.reconstruct()

            content_vec, output_vec, output_yuv_vec = sample_points(self.feature_extractor(content),
                                                                    self.feature_extractor(output),
                                                                    [rgb_to_yuv(output)],
                                                                    n_samples=self.n_points)
            style_vec, style_yuv_vec = sample_points(self.feature_extractor(style),
                                                     [style_yuv],
                                                     n_samples=self.n_points * 4)

            loss = strotss_loss(content_vec,
                                style_vec,
                                output_vec,
                                style_yuv_vec,
                                output_yuv_vec,
                                alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                'loss': loss.item(),
                'shape': rf'{tuple(output.shape[-2:])}',
            })

            self._iteration_end(self.post_process(output), i)
        self._iteration_end(self.post_process(output), self.n_update_steps)
        return laplacian_pyramid.reconstruct()

    def pre_process(self, content, style):
        content = util.normalize(content.unsqueeze(0))
        style = util.normalize(style.unsqueeze(0))

        if self.prior == 'content':
            output = content
        else:
            output = torch.mean(style, dim=(-2, -1), keepdim=True)
        return content, style, output

    def post_process(self, output):
        output = output.squeeze(0)
        output = util.denormalize(output)
        return output

    def _iteration_end(self, output, iter):
        if self.on_iteration_end:
            self.on_iteration_end(output, iter)
