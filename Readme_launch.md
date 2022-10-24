export PATH=/opt/slurm/bin:$PATH

srun -p gpuserv --pty bash

scancel job_id


python scripts/generate_meta_info.py  --input /home/d_korostelev/Projects/super_resolution/data/DeepRockSR/973_2D/images/DeepRockSR-2D/carbonate2D/carbonate2D_test_HR --root /home/d_korostelev/Projects/super_resolution/data/DeepRockSR/973_2D/images/DeepRockSR-2D/carbonate2D/carbonate2D_test_HR  --meta_info datasets/tomo/meta_info/meta_info_tomo.txt
