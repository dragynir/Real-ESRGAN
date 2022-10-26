export PATH=/opt/slurm/bin:$PATH

srun -p gpuserv --pty bash

srun -p gpuserv  --time 00:50:00 --pty bash

scancel job_id

cd 
python scripts/generate_meta_info.py  --input /home/d_korostelev/Projects/super_resolution/data/DeepRockSR/973_2D/images/DeepRockSR-2D/carbonate2D/carbonate2D_test_HR --root /home/d_korostelev/Projects/super_resolution/data/DeepRockSR/973_2D/images/DeepRockSR-2D/carbonate2D/carbonate2D_test_HR  --meta_info datasets/tomo/meta_info/meta_info_tomo.txt

CUDA_VISIBLE_DEVICES=0 python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --debug


# Генерация метаинформации
python scripts/generate_meta_info.py  --input datasets/tomo_test --root datasets/tomo_test  --meta_info datasets/tomo_test/meta_info/meta_info_tomo.txt
# Прогон изображения
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test/11146.png  -o predictions/11146.png 
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test_down_4/11146.png  -o predictions/11146_sr.png 
