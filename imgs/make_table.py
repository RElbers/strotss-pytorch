from math import ceil, floor
from pathlib import Path

import cv2
import numpy as np

from util import load, save



def pad_square(img):
    h, w, c = img.shape
    size = max(h, w)
    pad_h = size - h
    pad_w = size - w

    img = np.pad(img, ((floor(pad_h / 2), ceil(pad_h / 2)),
                       (floor(pad_w / 2), ceil(pad_w / 2)),
                       (0, 0)),
                 'constant',
                 constant_values=255)
    return img


def f(img):
    img = pad_square(img)

    _, w__, _ = img.shape
    w_ = w__ * r
    pad = w_ - w__
    pad = pad / 2

    img = np.pad(img, ((0, 0), (floor(pad), ceil(pad)), (0, 0)), 'constant', constant_values=255)

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = np.pad(img, ((8, 8), (8, 8), (0, 0)), 'constant', constant_values=255)
    return img


path = Path(rf"../replicate_style")
# path = Path(rf"./replicate_content")

h = 440
w = 658
r = w / h

styles = [rf'./styles/style_{i:02}.png' for i in range(5)]
styles = [load(f) for f in styles]
styles = [f(img) for img in styles]

contents = [rf'./contents/content_{i:02}.png' for i in range(5)]
contents = [load(f) for f in contents]
contents = [f(img) for img in contents]

rows = []
rows.append(np.hstack([np.ones_like(styles[0]) *255, *styles]))
for j in range(5):
    row = [contents[j]]

    for i in range(5):
        file = path.joinpath(rf'output_{j:02}_{i:02}.png')
        img = load(file)[:h, :w]

        img = np.pad(img, ((8, 8), (8, 8), (0, 0)), 'constant', constant_values=255)

        row.append(img)

    row = np.hstack(row)
    rows.append(row)

img = np.vstack(rows)

save('tbl.png', img)
