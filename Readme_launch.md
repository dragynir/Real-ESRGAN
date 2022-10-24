export PATH=/opt/slurm/bin:$PATH

srun -p gpuserv --pty bash

scancel job_id
