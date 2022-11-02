import cv2
import os
import numpy as np


if __name__ == '__main__':

    sr_path = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/predictions/tomo_test_down_4_sr'
    lr_path = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/tomo_test_down_4/'
    gt_path = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/tomo_test/'

    results_path = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/predictions/tomo_test_np'

    for img_name in os.listdir(sr_path):
        sr = cv2.imread(os.path.join(sr_path, img_name))
        lr = cv2.imread(os.path.join(lr_path, img_name))
        gt = cv2.imread(os.path.join(gt_path, img_name))
        print(img_name)
        print(sr.shape)
        print(lr.shape)
        print(gt.shape)

        arr = np.stack([lr, gt, sr], axis=1)

        print(arr.shape)

        break
