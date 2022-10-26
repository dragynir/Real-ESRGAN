import os
import shutil
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('/home/d_korostelev/Projects/super_resolution/data/v1_dataset_DeepRockSR.csv')
dest_path = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/tomo_test'

os.makedirs(dest_path, exist_ok=True)

df = df[df['split'] == 'test']
print('Count:', len(df))

for _, row in tqdm(df.iterrows()):
    shutil.copy(row['path'], os.path.join(dest_path, os.path.basename(row['path'])))
