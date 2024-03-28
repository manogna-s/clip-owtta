# CUDA_VISIBLE_DEVICES=2 nohup python tta_rosita.py --out_dir logs/analysis_queue --dataset cifar10OOD --strong_OOD MNIST --model maple --k_p 5 --k_n 5 --tta_method Rosita --N_m 64&
# CUDA_VISIBLE_DEVICES=2 nohup python tta_rosita.py --out_dir logs/analysis_queue --dataset cifar10OOD --strong_OOD MNIST --model maple --k_p 5 --k_n 5 --tta_method Rosita --N_m 128&

# CUDA_VISIBLE_DEVICES=2 nohup python tta_rosita.py --out_dir logs/analysis_queue --dataset cifar100OOD --strong_OOD MNIST --model coop --k_p 5 --k_n 5 --tta_method Rosita --N_m 64&
# CUDA_VISIBLE_DEVICES=2 nohup python tta_rosita.py --out_dir logs/analysis_queue --dataset cifar100OOD --strong_OOD MNIST --model coop --k_p 5 --k_n 5 --tta_method Rosita --N_m 128

# CUDA_VISIBLE_DEVICES=1 nohup python tta_rosita.py --out_dir logs/analysis_queue --tesize 30000 --dataset ImagenetR30kOOD --strong_OOD MNIST --model coop --k_p 5 --k_n 5 --tta_method Rosita --N_m 128&
# CUDA_VISIBLE_DEVICES=1 nohup python tta_rosita.py --out_dir logs/analysis_queue --tesize 30000 --dataset ImagenetR30kOOD --strong_OOD MNIST --model coop --k_p 5 --k_n 5 --tta_method Rosita --N_m 2048

CUDA_VISIBLE_DEVICES=1 nohup python tta_rosita.py --out_dir logs/analysis_queue --tesize 50000 --dataset ImagenetCOOD --strong_OOD MNIST --model coop --k_p 3 --k_n 10 --tta_method Rosita --N_m 64
CUDA_VISIBLE_DEVICES=1 nohup python tta_rosita.py --out_dir logs/analysis_queue --tesize 50000 --dataset ImagenetCOOD --strong_OOD MNIST --model coop --k_p 3 --k_n 10 --tta_method Rosita --N_m 128
CUDA_VISIBLE_DEVICES=1 nohup python tta_rosita.py --out_dir logs/analysis_queue --tesize 50000 --dataset ImagenetCOOD --strong_OOD MNIST --model coop --k_p 3 --k_n 10 --tta_method Rosita --N_m 256

CUDA_VISIBLE_DEVICES=1 nohup python tta_rosita.py --out_dir logs/analysis_queue --tesize 30000 --dataset ImagenetR30kOOD --strong_OOD MNIST --model coop --k_p 5 --k_n 5 --tta_method Rosita --N_m 64&
