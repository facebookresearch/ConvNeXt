# COCO Object detection with ConvNeXt

## Getting started 

We add ConvNeXt model and config files to [Swin Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/tree/6a979e2164e3fb0de0ca2546545013a4d71b2f7d).
Our code has been tested with commit `6a979e2`. Please refer to [README.md](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/6a979e2164e3fb0de0ca2546545013a4d71b2f7d/README.md) for installation and dataset preparation instructions.

## Results and Fine-tuned Models

| name | Pretrained Model | Method | Lr Schd | box mAP | mask mAP | #params | FLOPs | Fine-tuned Model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:| :---:|
| ConvNeXt-T | [ImageNet-1K](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth) | Mask R-CNN | 3x | 46.2 | 41.7 | 48M | 262G | [model](https://dl.fbaipublicfiles.com/convnext/coco/mask_rcnn_convnext_tiny_1k_3x.pth) |
| ConvNeXt-T | [ImageNet-1K](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth) | Cascade Mask R-CNN | 3x | 50.4 | 43.7 | 86M | 741G | [model](https://dl.fbaipublicfiles.com/convnext/coco/cascade_mask_rcnn_convnext_tiny_1k_3x.pth) |
| ConvNeXt-S | [ImageNet-1K](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224.pth) | Cascade Mask R-CNN | 3x | 51.9 | 45.0 | 108M | 827G | [model](https://dl.fbaipublicfiles.com/convnext/coco/cascade_mask_rcnn_convnext_small_1k_3x.pth) |
| ConvNeXt-B | [ImageNet-1K](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224.pth) | Cascade Mask R-CNN | 3x | 52.7 | 45.6 | 146M | 964G | [model](https://dl.fbaipublicfiles.com/convnext/coco/cascade_mask_rcnn_convnext_base_1k_3x.pth) |
| ConvNeXt-B | [ImageNet-22K](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth) | Cascade Mask R-CNN | 3x | 54.0 | 46.9 | 146M | 964G | [model](https://dl.fbaipublicfiles.com/convnext/coco/cascade_mask_rcnn_convnext_base_22k_3x.pth) |
| ConvNeXt-L | [ImageNet-22K](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth) | Cascade Mask R-CNN | 3x | 54.8 | 47.6 | 255M | 1354G | [model](https://dl.fbaipublicfiles.com/convnext/coco/cascade_mask_rcnn_convnext_large_22k_3x.pth) |
| ConvNeXt-XL | [ImageNet-22K](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth) | Cascade Mask R-CNN | 3x | 55.2 | 47.7 | 407M | 1898G | [model](https://dl.fbaipublicfiles.com/convnext/coco/cascade_mask_rcnn_convnext_xlarge_22k_3x.pth) |


### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [other optional arguments] 
```
For example, to train a Cascade Mask R-CNN model with a `ConvNeXt-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/convnext/cascade_mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py 8 --cfg-options model.pretrained=https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth
```

More config files can be found at [`configs/convnext`](configs/convnext).

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

## Acknowledgment 

This code is built using [mmdetection](https://github.com/open-mmlab/mmdetection), [timm](https://github.com/rwightman/pytorch-image-models) libraries, and [BeiT](https://github.com/microsoft/unilm/tree/f8f3df80c65eb5e5fc6d6d3c9bd3137621795d1e/beit), [Swin Transformer](https://github.com/microsoft/Swin-Transformer) repositories.