export PATH=/opt/slurm/bin:$PATH

# TODO
# Надо модифицировать деградационную модель
# Опрерации приводят к None в обучении или проблемы с GT min_max
# datasets/real/sub/sandstone/5x/recon_00004_s1357.tiff List index out of range

# Полезные команды
echo "export PATH=/opt/slurm/bin:$PATH" >> ~/.bashrc

srun -p gpuserv --pty bash

srun -p gpuserv  --time 00:10:00 --pty bash
srun -p gpuserv  --time 12:00:00 --pty bash

ssh -t -t d_korostelev@84.237.52.229 -L 8888:localhost:8881 ssh gpuserv -L 8881:localhost:8881

ssh -t -t d_korostelev@10.2.70.222 -L 8888:localhost:8881 ssh gpuserv -L 8881:localhost:8881

sbatch ./launch_slurm

scancel job_id

python scripts/generate_meta_info.py  --input /home/d_korostelev/Projects/super_resolution/data/DeepRockSR/973_2D/images/DeepRockSR-2D/carbonate2D/carbonate2D_test_HR --root /home/d_korostelev/Projects/super_resolution/data/DeepRockSR/973_2D/images/DeepRockSR-2D/carbonate2D/carbonate2D_test_HR  --meta_info datasets/tomo/meta_info/meta_info_tomo.txt

CUDA_VISIBLE_DEVICES=0 python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --debug



# Подготовка датасета

## Подготовка benchmark датасета
python scripts/generate_meta_info.py  --input datasets/tomo_test --root datasets/tomo_test  --meta_info datasets/tomo_test/meta_info/meta_info_tomo.txt
python scripts/generate_meta_info.py  --input datasets/tomo_train --root datasets/tomo_train  --meta_info datasets/tomo_train/meta_info/meta_info_tomo.txt

## Подготовка real датасета
/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/
Оригиналы tiff 1x: datasets/real/glass/1x datasets/real/sandstone/1x
Оригиналы tiff 5x: datasets/real/glass/5x datasets/real/sandstone/5x

# Tiff to RGB with min-max percentile normalization
5x:
python scripts/tiff_to_rgb.py --input datasets/real/glass/5x --out datasets/real/rgb/glass/5x
python scripts/tiff_to_rgb.py --input datasets/real/sandstone/5x --out datasets/real/rgb/sandstone/5x

1x:
python scripts/tiff_to_rgb.py --input datasets/real/glass/1x --out datasets/real/rgb/glass/1x
python scripts/tiff_to_rgb.py --input datasets/real/sandstone/1x --out datasets/real/rgb/sandstone/1x

# sandstone fast
python scripts/tiff_to_rgb.py --input /home/v_nikitin/data/APS/2022-10-rec-Fokin/loading3_stream_5MP_0659_rec --out datasets/test_real/rgb/sandstone_fast



# Create sub images
python scripts/extract_subimages.py --input datasets/real/rgb/glass/5x --output datasets/real/sub/glass/5x --crop_size 400 --step 350
python scripts/extract_subimages.py --input datasets/real/rgb/sandstone/5x --output datasets/real/sub/sandstone/5x --crop_size 400 --step 350

# for validation
python scripts/extract_subimages.py --input datasets/real/rgb/sandstone/1x --output datasets/real/sub/sandstone/1x_1024 --crop_size 1024 --step 1024
python scripts/extract_subimages.py --input datasets/real/rgb/sandstone/5x --output datasets/real/sub/sandstone/5x_4096 --crop_size 4096 --step 4096

python scripts/extract_subimages.py --input datasets/real/rgb/sandstone/5x --output datasets/real/sub/sandstone/5x_4096 --crop_size 4096 --step 4096


# Create synt resize images from sub images

python prepare_data_scripts/synt_image_resize.py --input datasets/real/sub/sandstone/5x_4096 --out  datasets/real/sub/sandstone/5x_4096_down4x

python prepare_data_scripts/synt_image_resize.py --input datasets/real/sub/sandstone/5x --out  datasets/real/sub/sandstone/5x_down4x

python prepare_data_scripts/synt_image_resize.py --input datasets/real/sub/sandstone/5x_4096 --out  datasets/real/sub/sandstone/5x_down4x_4096

python prepare_data_scripts/synt_image_resize.py --input datasets/test_real/rgb/sandstone_fast --out  datasets/test_real/rgb/sandstone_fast_down2x


# Generate meta-info
python scripts/generate_meta_info.py  --input datasets/real/sub/glass/5x datasets/real/sub/sandstone/5x --root datasets/real/sub datasets/real/sub --meta_info datasets/real/meta_info_train.txt

# Первая стадия обучения - realesrnet - обучаем без дискриминатора (на MAE)

debug проверка:
CUDA_VISIBLE_DEVICES=0 python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --debug

CUDA_VISIBLE_DEVICES=0,1 python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml 

CUDA_VISIBLE_DEVICES=0 nohup python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --auto_resume &

13300 mb видеопамяти

# Вторая стадия обучения - realesrgan - обучаем как GAN

Используем генератор обученные на первой стадии, добавляем его в конфиг
experiments/train_RealESRNetx4plus_1000k_B12G4/models/net_g_70000.pth

CUDA_VISIBLE_DEVICES=0,1 python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --debug

CUDA_VISIBLE_DEVICES=0,1 nohup python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml &

CUDA_VISIBLE_DEVICES=0,1 nohup python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --auto_resume &


# Прогон изображения
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test/11146.png  -o predictions/11146.png 
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test_down_4/11146.png  -o predictions/11146_sr.png 

# Прогон обученной resnet

#benchmark
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test_down_4/11146.png  -o predictions/11146_sr_trained.png --model_path experiments/train_RealESRNetx4plus_1000k_B12G4/models/net_g_70000.pth

#real 
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/real/rgb/glass/1x/recon_00000.png -o predictions/glass_recon_00000.png --model_path experiments/train_resnet_real_exp0/models/net_g_95000.pth


# Прогон обученной со второй стадией
# benchmark
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test_down_4/11146.png  -o predictions/11146_sr_trained_gan.png --model_path experiments/train_gan_long/models/net_g_80000.pth
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/real/sandstone_fast_original  -o predictions/sandstone_fast_sr_original --model_path experiments/train_gan_long/models/net_g_80000.pth

python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test_down_4/ -o predictions/tomo_test_down_4_sr_long --model_path experiments/train_gan_long/models/net_g_latest.pth

# real

python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/real/rgb/glass/1x -o predictions/real/glass/1x --model_path experiments/train_gan_real_exp0/models/net_g_85000.pth

python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/real/sub/sandstone/1x_1024 -o predictions/sandstone/1x_1024 --model_path experiments/train_gan_real_exp0/models/net_g_85000.pth

python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/real/sub/sandstone/5x_4096_down4x -o predictions/sandstone/5x_4096_down4x --model_path experiments/train_gan_real_exp0/models/net_g_85000.pth

python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/test_real/sandstone_fast -o predictions/sandstone/sandstone_fast --model_path experiments/train_gan_real_exp0/models/net_g_85000.pth


python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/test_real/sandstone_fast  -o predictions/sandstone/sandstone_fast_benchmark --model_path experiments/train_gan_long/models/net_g_80000.pth
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/test_real/sandstone_fast_original  -o predictions/sandstone/sandstone_fast_benchmark_original --model_path experiments/train_gan_long/models/net_g_80000.pth


python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/test_real/rgb/sandstone_fast  -o predictions/sandstone/sandstone_fast_full --model_path experiments/train_gan_long/models/net_g_80000.pth



python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/test_real/sandstone_fast -o predictions/sandstone_fast_4x --model_path experiments/train_gan_real_exp0/models/net_g_85000.pth

python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/real/sub/sandstone/5x_down4x -o predictions/sandstone_5x_down4x --model_path experiments/train_gan_real_exp0/models/net_g_85000.pth


python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/real/sub/sandstone/5x_down4x_4096 -o predictions/sandstone_5x_down4x_4096 --model_path experiments/train_gan_real_exp0/models/net_g_85000.pth


# Валидация с помощью modeling_sr
# Нужно делать валидацию чтобы выбрать лучший чекпоинт т. к. последний может быть не самым лучшим =)

python validate.py --images_path /home/d_korostelev/Projects/super_resolution/Real-ESRGAN/predictions/tomo_test_np_long --img_size 500 --rgb True
