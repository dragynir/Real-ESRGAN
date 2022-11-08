import cv2
from tifffile import tifffile
import os
from tqdm import tqdm
import argparse
import numpy as np


def center_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
    """"
        Perform center crop with given crop size
    """
    h, w = image.shape[:2]

    min_h = (h - crop_size) // 2
    min_w = (w - crop_size) // 2

    max_h = (h + crop_size) // 2
    max_w = (w + crop_size) // 2

    return image[min_h:max_h, min_w:max_w]


def tiff2rgb(image: np.ndarray) -> np.ndarray:
    """"
    Perform min-max percentile normalization
    Min-max normalization and 1 %, 99 % percentile clipping
    """
    min_v = image.min()
    max_v = image.max()
    image = (image - min_v) / (max_v - min_v + 1e-16)

    min_p = np.percentile(image, 1, axis=(0, 1))
    max_p = np.percentile(image, 99, axis=(0, 1))
    image = np.clip(image, min_p, max_p)

    return np.clip((image * 255.0).round(), 0, 255)


def main(args):

    os.makedirs(args.out, exist_ok=True)

    for name in tqdm(os.listdir(args.input)):
        if not 'tiff' in name:
            continue
        base_name = name.split('.')[0]
        dest_name = base_name + '.png'
        image = tifffile.imread(os.path.join(args.input, name))
        image = tiff2rgb(image)
        cv2.imwrite(os.path.join(args.out, dest_name), image)


if __name__ == '__main__':
    # TODO переписать на multithreding
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        help='Input folder, can be a list.')
    parser.add_argument(
        '--out',
        help='Output folder to write rgb.')

    args = parser.parse_args()
    main(args)
