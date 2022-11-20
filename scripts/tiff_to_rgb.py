from tifffile import tifffile
import argparse
import cv2
import numpy as np
import os
import sys
from basicsr.utils import scandir
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm


def main(args):
    opt = {'n_thread': args.n_thread, 'input_folder': args.input, 'save_folder': args.out}
    convert_images_to_rgb(opt)


def convert_images_to_rgb(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        # sys.exit(1)

    # scan all images
    img_list = list(scandir(input_folder, full_path=True))

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


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


def worker(path, opt):

    save_folder = opt['save_folder']
    img_name, extension = osp.splitext(osp.basename(path))
    print(img_name, extension)
    if extension != 'tiff':
        return

    dest_name = img_name + '.png'
    image = tifffile.imread(path)
    image = tiff2rgb(image)
    cv2.imwrite(os.path.join(save_folder, dest_name), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        help='Input folder, can be a list.')
    parser.add_argument(
        '--out',
        help='Output folder to write rgb.')
    parser.add_argument('--n_thread', type=int, default=10, help='Thread number.')

    args = parser.parse_args()
    main(args)
