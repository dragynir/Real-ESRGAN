import cv2
from tifffile import tifffile
import os
import numpy as np
from tqdm import tqdm
import shutil

if __name__ == '__main__':

    # 1x glass
    # tiff_folder = '/home/v_nikitin/data/APS/2022-10-rec-Fokin/glass_beads_1x_31MP_007_rec'
    # dest_folder = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/real/glass/1x'

    # 5x glass
    tiff_folder = '/home/v_nikitin/data/APS/2022-10-rec-Fokin/glass_beads_7p5x_31MP_010_rec'
    dest_folder = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/real/glass/5x'

    # =================================== sandstone
    # 1x sandstone
    # tiff_folder = '/home/v_nikitin/data/APS/2022-10/Nikitin_rec/sandstone_1x_091_rec'
    # dest_folder = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/real/sandstone/1x'

    # 5x sandstone
    # tiff_folder = '/home/v_nikitin/data/APS/2022-10/Nikitin_rec/sandstone_7p5x_094_rec'
    # dest_folder = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/real/sandstone/5x'

    os.makedirs(dest_folder, exist_ok=True)

    for name in tqdm(os.listdir(tiff_folder)):
        if not 'tiff' in name:
            continue
        shutil.copy(os.path.join(tiff_folder, name), os.path.join(dest_folder, name))
