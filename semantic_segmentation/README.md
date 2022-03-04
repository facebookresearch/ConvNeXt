# ADE20k Semantic segmentation with ConvNeXt

## Getting started 

### [Update: ConvNeXt now in official OpenMMLab/MMSegmentation]
If you want to try ConvNeXt with the latest MMSegmentation toolbox (recommended), please follow the instructions in
https://github.com/open-mmlab/mmsegmentation/tree/master/configs/convnext

Alternatively, if you want to run ConvNeXt in the original codebase, follow instructions below:

We add ConvNeXt model and config files to the semantic_segmentation folder of [BeiT](https://github.com/microsoft/unilm/tree/f8f3df80c65eb5e5fc6d6d3c9bd3137621795d1e/beit/semantic_segmentation).
Our code has been tested with commit `8b57ed1`. Please refer to [README.md](https://github.com/microsoft/unilm/tree/f8f3df80c65eb5e5fc6d6d3c9bd3137621795d1e/beit/semantic_segmentation/README.md) for installation and dataset preparation instructions.

## Results and Fine-tuned Models

| name | Pretrained Model | Method | Crop Size | Lr Schd | mIoU | mIoU (ms+flip) | #params | FLOPs | Fine-tuned Model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:| :---:|:---:|
| ConvNeXt-T | [ImageNet-1K](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth) | UPerNet | 512x512 | 160K | 46.0 | 46.7 | 60M | 939G | [model](https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_tiny_1k_512x512.pth) |
| ConvNeXt-S | [ImageNet-1K](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224.pth) | UPerNet | 512x512 | 160K | 48.7 | 49.6 | 82M | 1027G | [model](https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_small_1k_512x512.pth) |
| ConvNeXt-B | [ImageNet-1K](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224.pth) | UPerNet | 512x512 | 160K | 49.1 | 49.9 | 122M | 1170G | [model](https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_base_1k_512x512.pth) |
| ConvNeXt-B | [ImageNet-22K](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth) | UPerNet | 640x640 | 160K | 52.6 | 53.1 | 122M | 1828G | [model](https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_base_22k_640x640.pth) |
| ConvNeXt-L | [ImageNet-22K](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth) | UPerNet | 640x640 | 160K | 53.2 | 53.7 | 235M | 2458G | [model](https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_large_22k_640x640.pth) |
| ConvNeXt-XL | [ImageNet-22K](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth) | UPerNet | 640x640 | 160K | 53.6 | 54.0 | 391M | 3335G | [model](https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_xlarge_22k_640x640.pth) |

### Training
Note: Please add `from backbone import convnext` to tools/train.py.

Command format:
```
tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS> --work-dir <SAVE_PATH> --seed 0 --deterministic --options model.pretrained=<PRETRAIN_MODEL>
```

For example, using a `ConvNeXt-T` backbone with UperNet:
```bash
bash tools/dist_train.sh \
    configs/convnext/upernet_convnext_tiny_512_160k_ade20k_ms.py 8 \
    --work-dir /path/to/save --seed 0 --deterministic \
    --options model.pretrained=https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth
```

More config files can be found at [`configs/convnext`](configs/convnext).


## Evaluation
Note: Please add `from backbone import convnext` to tools/test.py.

Command format for multi-scale testing:
```
tools/dist_test.sh <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU --aug-test
```

For example, evaluate a `ConvNeXt-T` backbone with UperNet:
```bash
bash tools/dist_test.sh configs/convnext/upernet_convnext_tiny_512_160k_ade20k_ms.py \ 
    https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_tiny_1k_512x512.pth 4 --eval mIoU --aug-test
```

Command format for single-scale testing:
```
tools/dist_test.sh <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```

For example, evaluate a `ConvNeXt-T` backbone with UperNet:
```bash
bash tools/dist_test.sh configs/convnext/upernet_convnext_tiny_512_160k_ade20k_ss.py \ 
    https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_tiny_1k_512x512.pth 4 --eval mIoU
```

## Acknowledgment 

This code is built using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [timm](https://github.com/rwightman/pytorch-image-models) libraries, and [BeiT](https://github.com/microsoft/unilm/tree/f8f3df80c65eb5e5fc6d6d3c9bd3137621795d1e/beit), [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [XCiT](https://github.com/facebookresearch/xcit), [SETR](https://github.com/fudan-zvg/SETR) repositories.
