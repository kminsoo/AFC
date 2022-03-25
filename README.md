## Class-Incremental Learning by Knowledge Distillation with Adaptive Feature Consolidation

#### 1. Code Dependencies
Install required packages: 
```bash
conda env create --file environment.yaml
```

Swith to new environment: 
```bash
conda activate afc
```

#### 2. Experiments
To reproduce Table1 with 50 steps on CIFAR100 with three different class orders
LSC with CNN:

```bash
python3 -minclearn --options options/AFC/AFC_cnn_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 1 --fixed-memory \
    --device <GPU_ID> --label AFC_cnn_cifar100_50steps \
    --data-path <PATH/TO/DATA>
```
LSC with NME:

```bash
python3 -minclearn --options options/AFC/AFC_nme_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 1 --fixed-memory \
    --device <GPU_ID> --label AFC_nme_cifar100_50steps \
    --data-path <PATH/TO/DATA>
```
Likewise, for ImageNet100 (Table 2):

```bash
python3 -minclearn --options options/AFC/AFC_cnn_imagenet100.yaml options/data/imagenet100_1order.yaml \
    --initial-increment 50 --increment 1 --fixed-memory \
    --device <GPU_ID> --label AFC_cnn_imagenet100_50steps \
    --data-path <PATH/TO/DATA>
```

And ImageNet1000 (Table 2):

```bash
python3 -minclearn --options options/AFC/AFC_cnn_imagenet1000.yaml options/data/imagenet1000_1order.yaml \
    --initial-increment 500 --increment 50 --fixed-memory --memory-size 20000 \
    --device <GPU_ID> --label AFC_cnn_imagenet1000_10steps \
    --data-path <PATH/TO/DATA>
```

## Citation

If you use this code for your research, please cite our paper.
  ```shell
  @inproceedings{Kang2022afc,
	author = {Kang, Minsoo and Park, Jaeyoo and Han, Bohyung},
	booktitle = {CVPR},
	title = "{Class-Incremental Learning by Knowledge Distillation with Adaptive Feature Consolidation}",
	year = {2022}
	}
```

## Acknowledgements

This repository is developed based on [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch)
