






python scripts/generate_meta_info_pairdata.py --input datasets/real/rgb_cropped/glass/hr_images datasets/real/rgb_cropped/glass/lr_images --meta_info datasets/real/rgb_cropped/glass/meta_info/glass_paired.txt


python scripts/generate_meta_info_pairdata.py --input datasets/real/rgb_cropped_good/glass/hr_images datasets/real/rgb_cropped_good/glass/lr_images --meta_info datasets/real/rgb_cropped_good/glass/meta_info/glass_paired.txt


CUDA_VISIBLE_DEVICES=1  python realesrgan/train.py -opt options/finetune.yml --auto_resume --debug


CUDA_VISIBLE_DEVICES="0,1" python realesrgan/train.py -opt options/finetune.yml --auto_resume


CUDA_VISIBLE_DEVICES="0,1"  nohup python realesrgan/train.py -opt options/finetune.yml --auto_resume &


CUDA_VISIBLE_DEVICES="1" python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/real/rgb_cropped/glass/lr_images -o predictions/glass/lr_images --model_path experiments/finetune_paired_glass/models/net_g_55000.pth


CUDA_VISIBLE_DEVICES="1" python inference_realesrgan.py -n RealESRGAN_x4plus -i datasets/real/rgb_cropped/glass/lr_images -o predictions/glass/lr_images_latest --model_path experiments/finetune_paired_glass/models/net_g_latest.pth
