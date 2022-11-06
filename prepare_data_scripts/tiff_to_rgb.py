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


if __name__ == '__main__':

    tiff_folder = '/home/v_nikitin/data/APS/2022-10-rec-Fokin/loading3_stream_5MP_0659_rec'
    dest_folder = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/real/sandstone_fast'

    os.makedirs(dest_folder, exist_ok=True)

    for name in tqdm(os.listdir(tiff_folder)):
        if not 'tiff' in name:
            continue
        base_name = name.split('.')[0]
        dest_name = base_name + '.png'
        image = tifffile.imread(os.path.join(tiff_folder, name))
        image = np.stack([image] * 3, axis=-1)

        image = tiff2rgb(image)
        print(image.min(), image.max())

        cv2.imwrite(os.path.join(dest_folder, dest_name), image)
