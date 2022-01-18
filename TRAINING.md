# Training

We provide ImageNet-1K training, ImageNet-22K pre-training, and ImageNet-1K fine-tuning commands here.
Please check [INSTALL.md](INSTALL.md) for installation instructions first.

## Multi-node Training
We use multi-node training on a SLURM cluster with [submitit](https://github.com/facebookincubator/submitit) for producing the results and models in the paper. Please install:
```
pip install submitit
```
We will give example commands for both multi-node and single-machine training below.

## ImageNet-1K Training 
ConvNeXt-T training on ImageNet-1K with 4 8-GPU nodes:
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_tiny --drop_path 0.1 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

- You may need to change cluster-specific arguments in `run_with_submitit.py`.
- You can add `--use_amp true` to train in PyTorch's Automatic Mixed Precision (AMP).
- Use `--resume /path_or_url/to/checkpoint.pth` to resume training from a previous checkpoint; use `--auto_resume true` to auto-resume from latest checkpoint in the specified output folder.
- `--batch_size`: batch size per GPU; `--update_freq`: gradient accumulation steps.
- The effective batch size = `--nodes` * `--ngpus` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `4*8*128*1 = 4096`. You can adjust these four arguments together to keep the effective batch size at 4096 and avoid OOM issues, based on the model size, number of nodes and GPU memory.

You can use the following command to run this experiment on a single machine: 
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_tiny --drop_path 0.1 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k 
--output_dir /path/to/save_results
```

- Here, the effective batch size = `--nproc_per_node` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `8*128*4 = 4096`. Running on one machine, we increased `update_freq` so that the total batch size is unchanged.

To train other ConvNeXt variants, `--model` and `--drop_path` need to be changed. Examples are given below, each with both multi-node and single-machine commands:

<details>
<summary>
ConvNeXt-S
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_small --drop_path 0.4 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_small --drop_path 0.4 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
</details>
<details>
<summary>
ConvNeXt-B
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_base --drop_path 0.5 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
``` 

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_base --drop_path 0.5 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
``` 

</details>
<details>
<summary>
ConvNeXt-L
</summary>

Multi-node
```
python run_with_submitit.py --nodes 8 --ngpus 8 \
--model convnext_large --drop_path 0.5 \
--batch_size 64 --lr 4e-3 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
``` 

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_large --drop_path 0.5 \
--batch_size 64 --lr 4e-3 --update_freq 8 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
``` 

</details>

<details>
<summary>
ConvNeXt-S (isotropic)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_isotropic_small --drop_path 0.1 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--layer_scale_init_value 0 \
--warmup_epochs 50 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
``` 

Single-machine 
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_isotropic_small --drop_path 0.1 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--layer_scale_init_value 0 \
--warmup_epochs 50 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
``` 

</details>

<details>
<summary>
ConvNeXt-B (isotropic)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_isotropic_base --drop_path 0.2 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--layer_scale_init_value 0 \
--warmup_epochs 50 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
``` 

Single-machine 
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_isotropic_base --drop_path 0.2 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--layer_scale_init_value 0 \
--warmup_epochs 50 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
``` 

</details>

<details>
<summary>
ConvNeXt-L (isotropic)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 8 --ngpus 8 \
--model convnext_isotropic_large --drop_path 0.5 \
--batch_size 64 --lr 4e-3 --update_freq 1 \
--warmup_epochs 50 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
``` 

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_isotropic_large --drop_path 0.5 \
--batch_size 64 --lr 4e-3 --update_freq 8 \
--warmup_epochs 50 --model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
``` 

</details>

## ImageNet-22K Pre-training
ImageNet-22K is significantly larger than ImageNet-1K in terms of data size, so we use 16 8-GPU nodes for pre-training on ImageNet-22K.

ConvNeXt-B pre-training on ImageNet-22K:

Multi-node
```
python run_with_submitit.py --nodes 16 --ngpus 8 \
--model convnext_base --drop_path 0.1 \
--batch_size 32 --lr 4e-3 --update_freq 1 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true \
--data_path /path/to/imagenet-22k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_base --drop_path 0.1 \
--batch_size 32 --lr 4e-3 --update_freq 16 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true \
--data_path /path/to/imagenet-22k \
--output_dir /path/to/save_results
```

<details>
<summary>
ConvNeXt-L
</summary>

Multi-node
```
python run_with_submitit.py --nodes 16 --ngpus 8 \
--model convnext_large --drop_path 0.1 \
--batch_size 32 --lr 4e-3 --update_freq 1 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true \
--data_path /path/to/imagenet-22k \
--job_dir /path/to/save_results
``` 
    
Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_large --drop_path 0.1 \
--batch_size 32 --lr 4e-3 --update_freq 16 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true \
--data_path /path/to/imagenet-22k \
--output_dir /path/to/save_results
``` 

</details>

<details>
<summary>
ConvNeXt-XL
</summary>

Multi-node
```
python run_with_submitit.py --nodes 16 --ngpus 8 \
--model convnext_xlarge --drop_path 0.2 \
--batch_size 32 --lr 4e-3 --update_freq 1 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true \
--data_path /path/to/imagenet-22k \
--job_dir /path/to/save_results
``` 
    
Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_xlarge --drop_path 0.2 \
--batch_size 32 --lr 4e-3 --update_freq 16 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true \
--data_path /path/to/imagenet-22k \
--output_dir /path/to/save_results
``` 

</details>


## ImageNet-1K Fine-tuning
### Finetune from ImageNet-1K pre-training 
The training commands given above for ImageNet-1K use the default resolution (224). We also fine-tune these trained models with a larger resolution (384). Please specify the path or url to the checkpoint in `--finetune`.

ConvNeXt-B fine-tuning on ImageNet-1K (384x384):

Multi-node
```
python run_with_submitit.py --nodes 2 --ngpus 8 \
--model convnext_base --drop_path 0.8 --input_size 384 \
--batch_size 32 --lr 5e-5 --update_freq 1 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.7 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_base --drop_path 0.8 --input_size 384 \
--batch_size 32 --lr 5e-5 --update_freq 2 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.7 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

<details>
<summary>
ConvNeXt-L (384x384)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 2 --ngpus 8 \
--model convnext_large --drop_path 0.95 --input_size 384 \
--batch_size 32 --lr 5e-5 --update_freq 1 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.7 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
``` 

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_large --drop_path 0.95 --input_size 384 \
--batch_size 32 --lr 5e-5 --update_freq 2 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.7 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
``` 

- The fine-tuning for ImageNet-1K pre-trained ConvNeXt-L starts from the best ema weights during pre-training. You can add `--model_key model_ema` to load from a saved checkpoint that has `model_ema` as a key (e.g., obtained by training with `--model_ema true`), to load ema weights. Note that our provided pre-trained checkpoints only have `model` as the only key.

</details>

### Fine-tune from ImageNet-22K pre-training
We finetune from ImageNet-22K pre-trained models, in both 224 and 384 resolutions.

ConvNeXt-B fine-tuning on ImageNet-1K (224x224)

Multi-node
```
python run_with_submitit.py --nodes 2 --ngpus 8 \
--model convnext_base --drop_path 0.2 --input_size 224 \
--batch_size 32 --lr 5e-5 --update_freq 1 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_base --drop_path 0.2 --input_size 224 \
--batch_size 32 --lr 5e-5 --update_freq 2 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

<details>
<summary>
ConvNeXt-L (224x224)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 2 --ngpus 8 \
--model convnext_large --drop_path 0.3 --input_size 224 \
--batch_size 32 --lr 5e-5 --update_freq 1 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
``` 

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_large --drop_path 0.3 --input_size 224 \
--batch_size 32 --lr 5e-5 --update_freq 2 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
``` 

</details>

<details>
<summary>
ConvNeXt-XL (224x224)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_xlarge --drop_path 0.4 --input_size 224 \
--batch_size 16 --lr 5e-5 --update_freq 1 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results \
--model_ema true --model_ema_eval true
``` 

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_xlarge --drop_path 0.4 --input_size 224 \
--batch_size 16 --lr 5e-5 --update_freq 4 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results \
--model_ema true --model_ema_eval true
``` 

</details>

<details>
<summary>
ConvNeXt-B (384x384)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_base --drop_path 0.2 --input_size 384 \
--batch_size 16 --lr 5e-5 --update_freq 1 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
``` 

Single-machine   
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_base --drop_path 0.2 --input_size 384 \
--batch_size 16 --lr 5e-5 --update_freq 4 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
``` 

</details>

<details>
<summary>
ConvNeXt-L (384x384)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_large --drop_path 0.3 --input_size 384 \
--batch_size 16 --lr 5e-5 --update_freq 1 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
``` 

Single-machine    
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_large --drop_path 0.3 --input_size 384 \
--batch_size 16 --lr 5e-5 --update_freq 4 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
``` 

</details>

<details>
<summary>
ConvNeXt-XL (384x384)
</summary>

Multi-node
```
python run_with_submitit.py --nodes 8 --ngpus 8 \
--model convnext_xlarge --drop_path 0.4 --input_size 384 \
--batch_size 8 --lr 5e-5 --update_freq 1 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results \
--model_ema true --model_ema_eval true
``` 

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_xlarge --drop_path 0.4 --input_size 384 \
--batch_size 8 --lr 5e-5 --update_freq 8 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results \
--model_ema true --model_ema_eval true
``` 

</details>

