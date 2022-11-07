export PATH=/opt/slurm/bin:$PATH

# TODO
# Надо модифицировать деградационную модель
# Опрерации приводят к None в обучении

# чтобы после рестарта запускалось
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



#SBATCH -J tomo  # Job name
#SBATCH -p gpuserv # Queue name (another queue - compclass)
#SBATCH -e launch.err # Name of stderr file (%j expands to %jobId)
#SBATCH -o launch.out  # Name of stdout output file (%j expands to %jobId)
#SBATCH -N 1   # Total number of nodes requested
#SBATCH -c 4   # CPUs per task
#SBATCH -t 01:00:00 # Maximal run time (hh:mm:ss) - 1 minute
. $CONDA_ROOT/etc/profile.d/conda.sh

module load nvidia/cuda
echo "Current path=`pwd`"
echo "node=`hostname`"
echo "nproc=`nproc`"
nvcc --version
echo $SLURM_JOBID
echo $SLURM_SUBMIT_DIR
echo $SLURM_JOB_NODELIST
echo $SLURM_CPUS_PER_TASK
echo $SLURM_NTASKS
nvidia-smi
conda activate env_realsr
cd /home/d_korostelev/Projects/super_resolution
jupyter notebook --port 8881 --no-browser


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


# Генерация метаинформации benchmark
python scripts/generate_meta_info.py  --input datasets/tomo_test --root datasets/tomo_test  --meta_info datasets/tomo_test/meta_info/meta_info_tomo.txt
python scripts/generate_meta_info.py  --input datasets/tomo_train --root datasets/tomo_train  --meta_info datasets/tomo_train/meta_info/meta_info_tomo.txt

# Генерация метаинформации real

nohup python scripts/extract_subimages.py --input datasets/real/glass/5x --output datasets/real/sub/glass/5x --crop_size 400 --step 50 --tiff &
python scripts/extract_subimages.py --input datasets/real/sandstone/5x --output datasets/real/sub/sandstone/5x --crop_size 400 --step 50 --tiff

python scripts/generate_meta_info.py  --input datasets/real/sub/glass/5x --root datasets/real/sub --meta_info datasets/real/meta_info_train.txt
python scripts/generate_meta_info.py  --input datasets/real/sub/sandstone/5x --root datasets/real/sub --meta_info datasets/real/meta_info_train.txt


[comment]: <> (python scripts/generate_meta_info.py  --input datasets/real/glass/5x datasets/real/sandstone/5x --root datasets/real datasets/real  --meta_info datasets/real/meta_info_train.txt)
python scripts/generate_meta_info.py  --input datasets/real/glass/1x datasets/real/sandstone/1x --root datasets/real datasets/real  --meta_info datasets/real/meta_info_test.txt


# Прогон изображения
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test/11146.png  -o predictions/11146.png 
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test_down_4/11146.png  -o predictions/11146_sr.png 

# Прогон обученной resnet
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test_down_4/11146.png  -o predictions/11146_sr_trained.png --model_path experiments/train_RealESRNetx4plus_1000k_B12G4/models/net_g_70000.pth

# Прогон обученной со второй стадией
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test_down_4/11146.png  -o predictions/11146_sr_trained_gan.png --model_path experiments/train_gan_long/models/net_g_80000.pth
python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/real/sandstone_fast_original  -o predictions/sandstone_fast_sr_original --model_path experiments/train_gan_long/models/net_g_80000.pth


python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/tomo_test_down_4/ -o predictions/tomo_test_down_4_sr_long --model_path experiments/train_gan_long/models/net_g_latest.pth

# Валидация с помощью modeling_sr
# Нужно делать валидацию чтобы выбрать лучший чекпоинт т. к. последний может быть не самым лучшим =)

python validate.py --images_path /home/d_korostelev/Projects/super_resolution/Real-ESRGAN/predictions/tomo_test_np_long --img_size 500 --rgb True
