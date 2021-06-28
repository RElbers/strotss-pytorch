from torch import nn
from torchvision import models
from torchvision.transforms import transforms

import util


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self._vgg = models.vgg16(pretrained=True).features
        self._vgg.eval()
        for parameter in self._vgg.parameters():
            parameter.requires_grad = False

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.keep_idc = [1, 3, 6, 8, 11, 13, 15, 22, 29]

    def __call__(self, xs):
        assert xs.dim() == 4

        xs = util.denormalize(xs)
        xs = xs / 255.0

        xs = self.normalize(xs)

        feats = [xs]
        for i, layer in enumerate(self._vgg):
            xs = layer(xs)

            if i in self.keep_idc:
                feats.append(xs)

        return feats
