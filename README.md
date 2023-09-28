# When hard negative sampling meets supervised contrastive learning

Official PyTorch implementation of Paper: When hard negative sampling meets supervised contrastive learning.

## Setup


### install required packages:
```
pip install -r requirements.txt
```

### Download Checkpoints of BEiT-3

1. Models pretrained on ImageNet-21k images, 160 GB text documents, and web-scale image-text pairs (collected from [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/), [English LAION-2B](https://laion.ai/blog/laion-5b/), [COYO-700M](https://github.com/kakaobrain/coyo-dataset), and CC15M). 
   - [`BEiT3-base`](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_patch16_224.pth): #layer=12; hidden=768; FFN factor=4x; #head=12; patch=16x16; #parameters: 276M
   - [`BEiT3-large`](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_patch16_224.pth): #layer=24; hidden=1024; FFN factor=4x; #head=16; patch=16x16; #parameters: 746M

### prepare data

#### ImageNet-1k example
Download and extract ImageNet-1k from http://image-net.org/.

The directory structure is the standard layout of torchvision's [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder). The training and validation data are expected to be in the `train/` folder and `val/` folder, respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

We then generate the index json files using the following command. 
```
from datasets import ImageNetDataset

ImageNetDataset.make_dataset_index(
    train_data_path = "/path/to/your_data/train",
    val_data_path = "/path/to/your_data/val",
    index_path = "/path/to/your_data"
)
```


## Few-shot learning

The proposed objective can be evaluated using 8 A100-40GB:
```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_base_patch16_224 \
        --task CIFAR-FS \
        --batch_size 128 \
        --layer_decay 0.65 \
        --lr 7e-4 \
        --update_freq 1 \
        --epochs 100 \
        --warmup_epochs 5 \
        --drop_path 0.15 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --dist_eval \
        --mixup 0.8 \
        --cutmix 1.0 \
        --loss 'SCHaNe' \
        --alpha 0.9 \
        --temp 0.5 \
        --estimator 'hard' \
        --eval_fewshot

```
- `--batch_size`: batch size per GPU. Effective batch size = `number of GPUs` * `--batch_size` * `--update_freq`. So in the above example, the effective batch size is `8*128*1 = 1024`.
- `--finetune`: weight path of your pretrained models.
- `--alpha`: weight of the contrastive loss.
- `--task`: dataset to be used for fine-tuning, e.g. `CIFAR-FS` and `miniImageNet`.
- `--temp`: temperature of the contrastive loss
- `--estimator`: estimator of the contrastive loss, `hard` means hard negative sampling.
- `--loss`: loss function, `SCHaNe` or `CE` or `SupCon` or `SimCLR`.

## Full dataset Fine-tuning on ImageNet-1k (Image Classification)

The detailed instructions can be found at [`get_started_for_image_classification.md`](get_started/get_started_for_image_classification.md). We only use vision-related parameters for image classification fine-tuning.

```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_base_patch16_224 \
        --task imagenet \
        --batch_size 128 \
        --layer_decay 0.65 \
        --lr 7e-4 \
        --update_freq 1 \
        --epochs 50 \
        --warmup_epochs 5 \
        --drop_path 0.15 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.05 \
        --seed 42 \
        --save_ckpt_freq 5 \
        --dist_eval \
        --mixup 0.8 \
        --cutmix 1.0 \
        --loss 'SCHaNe' \
        --alpha 0.9 \
        --temp 0.5 \
        --estimator 'hard' \

```



## Citation

If you find this repository useful, please consider citing works:
```
```


## Acknowledgement

This repository is built using the [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3) repository and the [timm](https://github.com/rwightman/pytorch-image-models) library.


