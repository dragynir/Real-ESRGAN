export PATH=/opt/slurm/bin:$PATH

srun -p gpuserv --pty bash

srun -p gpuserv  --time 00:50:00 --pty bash

scancel job_id

cd 
python scripts/generate_meta_info.py  --input /home/d_korostelev/Projects/super_resolution/data/DeepRockSR/973_2D/images/DeepRockSR-2D/carbonate2D/carbonate2D_test_HR --root /home/d_korostelev/Projects/super_resolution/data/DeepRockSR/973_2D/images/DeepRockSR-2D/carbonate2D/carbonate2D_test_HR  --meta_info datasets/tomo/meta_info/meta_info_tomo.txt

CUDA_VISIBLE_DEVICES=0 python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --debug



/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/tomo

import os
import shutil
import pandas as pd

df = pd.read_csv('/home/d_korostelev/Projects/super_resolution/data/v1_dataset_DeepRockSR.csv')
[shutil.copy(row['path'], os.path.join(dest_path, os.path.basename(row['path']))) for _, row in tqdm(df.iterrows())]