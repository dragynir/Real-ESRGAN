import os
import shutil
import pandas as pd
from tqdm import tqdm
import cv2


def create_dataset(df: pd.DataFrame, dest_path: str, split='test', resize: int = None):

    os.makedirs(dest_path, exist_ok=True)

    df = df[df['split'] == split]
    print('Count:', len(df))

    for _, row in tqdm(df.iterrows()):
        dest_image = os.path.join(dest_path, os.path.basename(row['path']))
        if resize is not None:
            image = cv2.imread(row['path'])
            h, w = image.shape[0], image.shape[1]
            image = cv2.resize(image, (h//resize, w//resize))
            cv2.imwrite(dest_image, image)
        else:
            shutil.copy(row['path'], dest_image)


if __name__ == '__main__':
    df = pd.read_csv('/home/d_korostelev/Projects/super_resolution/data/v1_dataset_DeepRockSR.csv')
    dest_path = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/tomo_train'
    create_dataset(df, dest_path, split='train')
