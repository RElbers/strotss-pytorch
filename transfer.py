from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image

from strotss import STROTSS
from util import load, save, to_torch, from_torch, resize_long_side_to


def save_animation(out_file, temp_dir, size):
    files = Path(temp_dir).glob('*.png')
    files = list(sorted(files))

    img, *imgs = [Image.open(f) for f in sorted(files)]
    img, *imgs = [img.resize(size) for img in [img, *imgs]]
    img.save(fp=out_file, format=out_file.suffix.lstrip('.'), append_images=imgs, save_all=True, duration=100, loop=1)

    for file in files:
        file.unlink()


def save_current_output(img, iteration):
    if iteration % 10 == 0:
        path = Path('./temp')
        path.mkdir(exist_ok=True)
        save(path.joinpath(f'output_({img.size(-2):04},{img.size(-1):04})_{iteration:04}.png'), from_torch(img))


if __name__ == '__main__':
    parser = ArgumentParser()
    # Files
    parser.add_argument('--content', type=Path, default='./content.jpg')
    parser.add_argument('--style', type=Path, default='./style.jpg')
    parser.add_argument('--output', type=Path, default='./output.jpg')
    parser.add_argument('--animation', type=Path, default='./anim.gif')
    parser.add_argument('--temp', type=Path, default='./temp')

    # Hyperparams
    parser.add_argument('--min-size', type=int, default=64)
    parser.add_argument('--n-update-steps', type=int, default=200)
    parser.add_argument('--n-points', type=int, default=1024)
    parser.add_argument('--alpha', type=float, default=16.0)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--prior', type=str, default='mean-style', choices=['content', 'mean-style'])
    args = parser.parse_args()

    # Stylization model
    strotss = STROTSS(prior=args.prior,
                      min_size=args.min_size,
                      n_update_steps=args.n_update_steps,
                      n_points=args.n_points,
                      alpha=args.alpha,
                      lr=args.lr)
    strotss.on_iteration_end = save_current_output
    strotss.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # Load files
    content = to_torch(load(args.content))
    style = to_torch(load(args.style))

    # Resize style image to content image
    style = resize_long_side_to(style, max(content.shape[-2], content.shape[-1]))

    # Stylize content
    output = strotss(content, style)

    # Save output
    save(args.output, from_torch(output))
    save_animation(args.animation,
                   args.temp,
                   size=(output.shape[-1], output.shape[-2]))
