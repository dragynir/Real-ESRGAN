import argparse
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import json


def main(args):
    opt = {'n_thread': args.n_thread, 'mapping': args.mapping, 'save_folder': args.out}
    convert_images_to_rgb(opt)


def convert_images_to_rgb(opt):

    mapping_path = opt['mapping']
    save_folder = opt['save_folder']

    with open(mapping_path, 'r') as outfile:
        mapping = json.load(outfile)

    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        os.makedirs(os.path.join(save_folder, 'lr_images'))
        os.makedirs(os.path.join(save_folder, 'hr_images'))
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    pbar = tqdm(total=len(mapping), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for pair in mapping.items():
        pool.apply_async(worker, args=(pair, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def create_circle_mask(image, radius=500):
    center = image.shape[0] // 2
    mask = np.zeros(image.shape[:2])
    center_coordinates = (center, center)
    color = (1)
    thickness = -1
    mask = 1 - cv2.circle(mask, center_coordinates, radius, color, thickness)
    return mask


def create_image_crops(lr_image, hr_image, mask, crop_size, step, name, out_path, thresh_size=0):
    h_b, w_b = hr_image.shape[:2]

    h, w = lr_image.shape[:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    mask_size = crop_size * crop_size

    RESIZE_FACTOR = 4
    index = 0
    for x in h_space:
        for y in w_space:
            index += 1

            mask_crop = mask[x:x + crop_size, y:y + crop_size, ...]
            if np.sum(mask_crop) / mask_size > 0.6:
                continue

            x_b = int((x / h) * h_b)
            y_b = int((y / w) * w_b)
            crop_size_b_x = int((crop_size / h) * h_b)
            crop_size_b_y = int((crop_size / w) * w_b)

            hr_cropped = hr_image[x_b:x_b + crop_size_b_x, y_b:y_b + crop_size_b_y, ...]
            lr_cropped = lr_image[x:x + crop_size, y:y + crop_size, ...]
            cv2.imwrite(os.path.join(out_path, 'lr_images', f'{name}_{index}.png'), lr_cropped)

            hr_cropped = cv2.resize(hr_cropped, (crop_size * RESIZE_FACTOR, crop_size * RESIZE_FACTOR))
            cv2.imwrite(os.path.join(out_path, 'hr_images', f'{name}_{index}.png'), hr_cropped)


def worker(pair, opt):
    lr_image_path, hr_image_path = pair
    name = os.path.basename(lr_image_path)
    hr_image_path = hr_image_path.replace('5x_cropped', '5x_cropped_fullsize')
    lr_image = cv2.imread(os.path.join('notebooks', lr_image_path))
    hr_image = cv2.imread(os.path.join('notebooks', hr_image_path))

    mask_radius = int((lr_image.shape[0] // 2) * 0.96)
    mask = create_circle_mask(lr_image, mask_radius)
    create_image_crops(lr_image, hr_image, mask, crop_size=128, step=96, name=name, out_path=opt['save_folder'])


if __name__ == '__main__':
    # python scripts/extract_paired_images.py --mapping ./notebooks/glass_mapping_2_to_5.json --out datasets/real/rgb_cropped/glass
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mapping',
        help='Images mapping.')
    parser.add_argument(
        '--out',
        help='Output folder to write rgb paired images.')
    parser.add_argument('--n_thread', type=int, default=10, help='Thread number.')

    args = parser.parse_args()
    main(args)
