import cv2
from tifffile import tifffile
import os
import numpy as np
from tqdm import tqdm


def tiff2rgb(image):

    min_v = image.min()
    max_v = image.max()

    image = (image - min_v) / (max_v - min_v)

    return image * 255


def center_crop(image, crop_size):

    h, w = image.shape[:2]

    min_h = (h - crop_size) // 2
    min_w = (w - crop_size) // 2

    max_h = (h + crop_size) // 2
    max_w = (w + crop_size) // 2

    return image[min_h:max_h, min_w:max_w]


def crop_by_mask(image):
    return image[364:-314, 364:-296, :]


if __name__ == '__main__':

    tiff_folder = '/home/v_nikitin/data/APS/2022-10-rec-Fokin/loading3_stream_5MP_0659_rec'
    dest_folder = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/real/sandstone_fast_original'

    os.makedirs(dest_folder, exist_ok=True)

    for name in tqdm(os.listdir(tiff_folder)):
        if not 'tiff' in name:
            continue
        base_name = name.split('.')[0]
        dest_name = base_name + '.png'
        image = tifffile.imread(os.path.join(tiff_folder, name))
        image = np.stack([image] * 3, axis=-1)

        image = tiff2rgb(image)
        # print(image.min(), image.max())
        # image = center_crop(image, crop_size=1024)
        image = crop_by_mask(image)

        cv2.imwrite(os.path.join(dest_folder, dest_name), image)
        break
