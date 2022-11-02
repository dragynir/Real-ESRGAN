import cv2
import os
import numpy as np


if __name__ == '__main__':

    sr_path = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/predictions/tomo_test_down_4_sr'
    lr_path = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/tomo_test_down_4/'
    gt_path = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/tomo_test/'

    results_path = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/predictions/tomo_test_np'

    for img_name in os.listdir(gt_path):

        if 'png' not in img_name:
            continue

        sr_image_name = img_name.split('.')[0] + '_out.png'
        sr = cv2.imread(os.path.join(sr_path, sr_image_name))
        lr = cv2.imread(os.path.join(lr_path, img_name))
        gt = cv2.imread(os.path.join(gt_path, img_name))
        
        lr = cv2.resize(lr, (gt.shape[0], gt.shape[1]))
        
        print(img_name)
        print(sr.shape)
        print(lr.shape)
        print(gt.shape)

        arr = np.concatenate([lr, gt, sr], axis=1)

        print(arr.shape)

        break
