# CUDA_VISIBLE_DEVICES=0 nohup python tta_rosita.py --out_dir logs/analysis_ratio --dataset cifar10OOD --strong_OOD MNIST --model maple --strong_ratio 0.2 --tta_method ZSEval&
# CUDA_VISIBLE_DEVICES=0 nohup python tta_rosita.py --out_dir logs/analysis_ratio --dataset cifar10OOD --strong_OOD MNIST --model maple --strong_ratio 0.2 --tta_method PromptAlign&

# CUDA_VISIBLE_DEVICES=1 nohup python tta_rosita.py --out_dir logs/analysis_ratio --dataset cifar10OOD --strong_OOD MNIST --model maple --strong_ratio 0.4 --tta_method ZSEval&
# CUDA_VISIBLE_DEVICES=1 nohup python tta_rosita.py --out_dir logs/analysis_ratio --dataset cifar10OOD --strong_OOD MNIST --model maple --strong_ratio 0.4 --tta_method PromptAlign&

# CUDA_VISIBLE_DEVICES=2 nohup python tta_rosita.py --out_dir logs/analysis_ratio --dataset cifar10OOD --strong_OOD MNIST --model maple --strong_ratio 0.6 --tta_method ZSEval&
# CUDA_VISIBLE_DEVICES=2 nohup python tta_rosita.py --out_dir logs/analysis_ratio --dataset cifar10OOD --strong_OOD MNIST --model maple --strong_ratio 0.6 --tta_method PromptAlign&

# CUDA_VISIBLE_DEVICES=0 nohup python tta_rosita.py --out_dir logs/analysis_ratio --dataset cifar10OOD --strong_OOD MNIST --model maple --strong_ratio 0.8 --tta_method ZSEval&
CUDA_VISIBLE_DEVICES=0 nohup python tta_rosita.py --out_dir logs/analysis_ratio --dataset cifar10OOD --strong_OOD MNIST --model maple --strong_ratio 1.0 --tta_method PromptAlign&



