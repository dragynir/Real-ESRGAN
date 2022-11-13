import cv2
import os
from tqdm import tqdm
import argparse


def main(args):

    os.makedirs(args.out, exist_ok=True)
    resize_scale = 2
    for name in tqdm(os.listdir(args.input)):
        image = cv2.imread(os.path.join(args.input, name))
        new_size = image.shape[0] // resize_scale
        image = cv2.resize(image, (new_size, new_size))
        cv2.imwrite(os.path.join(args.out, name), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        help='Input folder, can be a list.')
    parser.add_argument(
        '--out',
        help='Output folder to write rgb.')

    args = parser.parse_args()
    main(args)
