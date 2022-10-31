export PATH=/opt/slurm/bin:$PATH

# чтобы после рестарта запускалось
echo "export PATH=/opt/slurm/bin:$PATH" >> ~/.bashrc

srun -p gpuserv --pty bash

srun -p gpuserv  --time 00:10:00 --pty bash
srun -p gpuserv  --time 12:00:00 --pty bash


sbatch ./launch_slurm

scancel job_id

python scripts/generate_meta_info.py  --input /home/d_korostelev/Projects/super_resolution/data/DeepRockSR/973_2D/images/DeepRockSR-2D/carbonate2D/carbonate2D_test_HR --root /home/d_korostelev/Projects/super_resolution/data/DeepRockSR/973_2D/images/DeepRockSR-2D/carbonate2D/carbonate2D_test_HR  --meta_info datasets/tomo/meta_info/meta_info_tomo.txt


# Первая стадия обучения - realesrnet - обучаем без дискриминатора (на MAE)
CUDA_VISIBLE_DEVICES=0,1 python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml 
--debug

CUDA_VISIBLE_DEVICES=0 nohup python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --auto_resume &

13300 mb видеопамяти

# Вторая стадия обучения - realesrgan - обучаем как GAN

Используем генератор обученные на первой стадии, добавляем его в конфиг
experiments/train_RealESRNetx4plus_1000k_B12G4/models/net_g_70000.pth


CUDA_VISIBLE_DEVICES=0,1 python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --debug

CUDA_VISIBLE_DEVICES=0,1 nohup python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml &

CUDA_VISIBLE_DEVICES=0,1 nohup python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --auto_resume &


# Генерация метаинформации
python scripts/generate_meta_info.py  --input datasets/tomo_test --root datasets/tomo_test  --meta_info datasets/tomo_test/meta_info/meta_info_tomo.txt
 
python scripts/generate_meta_info.py  --input datasets/tomo_train --root datasets/tomo_train  --meta_info datasets/tomo_train/meta_info/meta_info_tomo.txt

# Прогон изображения
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test/11146.png  -o predictions/11146.png 
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test_down_4/11146.png  -o predictions/11146_sr.png 

# Прогон обученной resnet
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test_down_4/11146.png  -o predictions/11146_sr_trained.png --model_path experiments/train_RealESRNetx4plus_1000k_B12G4/models/net_g_70000.pth

# Прогон обученной со второй стадией
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test_down_4/11146.png  -o predictions/11146_sr_trained_gan.png --model_path experiments/train_gan_long/models/net_g_80000.pth
