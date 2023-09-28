import os.path

import pandas as pd
import torch.backends.cudnn as cudnn
import math
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from beit3_tools import utils
from fewshot_dataset.transform_cfg import transforms_options

from fewshot_dataset.mini_imagenet import ImageNet, MetaImageNet
from fewshot_dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from fewshot_dataset.cifar import CIFAR100, MetaCIFAR100
from fewshot_dataset.transform_cfg import transforms_options, transforms_list
from PIL import Image
def merge_batch_tensors(batch):
    if isinstance(batch, (list, tuple)):
        batch = batch
    else:
        raise TypeError('Unsupported type for batch: {}'.format(type(batch)))
    return batch

def create_dataloader(dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False):
    if is_train or dist_eval:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        if not is_train and dist_eval and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')

        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )

    else:
        sampler = torch.utils.data.SequentialSampler(dataset,)

    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=merge_batch_tensors,
    )

import cv2
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        # print(x.layers)
        return [self.transform(x), self.transform(x)]

class TwoCropTransformcaltech256:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        # print(x.layers)
        if x.layers == 1:
            x  = np.array(x)
            x = np.atleast_3d(x)
            x = x.repeat(3, axis=2)
            # convert numpy array to PIL Image
            x = torchvision.transforms.functional.to_pil_image(x)

        x = x.convert('RGB')
        return [self.transform(x), self.transform(x)]


class CovertToRGB:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        if x.layers == 1:
            x  = np.array(x)
            x = np.atleast_3d(x)
            x = x.repeat(3, axis=2)
            # convert numpy array to PIL Image
            x = torchvision.transforms.functional.to_pil_image(x)
        x = x.convert('RGB')
        return self.transform(x)

import json
def load_file(root, dataset):
    if dataset == 'inaturelist2017':
        year_flag = 7
    elif dataset == 'inaturelist2018':
        year_flag = 8

    if dataset == 'inaturelist2018':
        with open(os.path.join(root, 'categories.json'), 'r') as f:
            map_label = json.load(f)
        map_2018 = dict()
        for _map in map_label:
            map_2018[int(_map['id'])] = _map['name'].strip().lower()

    with open(os.path.join(root, f'val201{year_flag}.json'), 'r') as f:
        val_class_info = json.load(f)
    with open(os.path.join(root, f'train201{year_flag}.json'), 'r') as f:
        train_class_info = json.load(f)

    if dataset == 'inaturelist2017':
        categories_2017 = [x['name'].strip().lower() for x in val_class_info['categories']]
        class_to_idx = {c: idx for idx, c in enumerate(categories_2017)}
        id2label = dict()
        for categorie in val_class_info['categories']:
            id2label[int(categorie['id'])] = categorie['name'].strip().lower()
    elif dataset == 'inaturelist2018':
        categories_2018 = [x['name'].strip().lower() for x in map_label]
        class_to_idx = {c: idx for idx, c in enumerate(categories_2018)}
        id2label = dict()
        for categorie in val_class_info['categories']:
            name = map_2018[int(categorie['name'])]
            id2label[int(categorie['id'])] = name.strip().lower()

    return train_class_info, val_class_info, class_to_idx, id2label

def find_images_and_targets_2017_2018(root,dataset,istrain=False,aux_info=False):
    train_class_info,val_class_info,class_to_idx,id2label = load_file(root,dataset)
    miss_hour = (dataset == 'inaturelist2017')

    class_info = train_class_info if istrain else val_class_info

    images_and_targets = []

    for image,annotation in zip(class_info['images'],class_info['annotations']):
        file_path = os.path.join(root,image['file_name'])
        id_name = id2label[int(annotation['category_id'])]
        target = class_to_idx[id_name]
        image_id = image['id']

        if aux_info:
            raise NotImplementedError

        else:
            images_and_targets.append((file_path,target))
    return images_and_targets,class_to_idx,

class DatasetMeta(torch.utils.data.Dataset):
    def __init__(
            self,
            root,
            load_bytes=False,
            transform=None,
            train=False,
            aux_info=False,
            dataset='inaturelist2021',
            class_ratio=1.0,
            per_sample=1.0):
        self.aux_info = aux_info
        self.dataset = dataset
        if dataset in ['inaturelist2021', 'inaturelist2021_mini']:
            images, class_to_idx, images_info = find_images_and_targets(root, train, aux_info)
        elif dataset in ['inaturelist2017', 'inaturelist2018']:
            images, class_to_idx, = find_images_and_targets_2017_2018(root, dataset, train, aux_info)
        elif dataset == 'cub-200':
            images, class_to_idx, images_info = find_images_and_targets_cub200(root, dataset, train, aux_info)
        elif dataset == 'stanfordcars':
            images, class_to_idx, images_info = find_images_and_targets_stanfordcars(root, dataset, train)
        elif dataset == 'oxfordflower':
            images, class_to_idx, images_info = find_images_and_targets_oxfordflower(root, dataset, train, aux_info)
        elif dataset == 'stanforddogs':
            images, class_to_idx, images_info = find_images_and_targets_stanforddogs(root, dataset, train)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx

        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        if self.aux_info:
            path, target, aux_info = self.samples[index]
        else:
            path, target = self.samples[index]
        path = path.split('/')[4:]
        path = os.path.join(self.root,*path)
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.aux_info:
            raise NotImplementedError

        else:
            return img, target

    def __len__(self):
        return len(self.samples)



def split_dataset(dataset, opt):
    train_len = int(len(dataset) * opt.train_ratio)
    # print(f"traing length is {train_len}")
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])
    return train_dataset, val_dataset


def create_image_downstream_datasets(opt):
    # construct data loader
    # set mean and std
    if opt.task == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.task == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.task == 'imagenet':
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif opt.task == 'bird':
        mean = [0.4859, 0.4996, 0.4318]
        std = [0.1750, 0.1739, 0.1859]
    elif opt.task == 'caltech256':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # mean = [ 0.456, 0.406]
        # std = [0.224, 0.225]
    elif opt.task == 'flowers102':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif opt.task == 'oxfordpet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif opt.task == 'places365':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif opt.task == 'inat2017':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif opt.task == 'inat2018':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif opt.task == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.task))

    normalize = transforms.Normalize(mean=mean, std=std)

    # set train_transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.input_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        normalize,
    ])


    if opt.task == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_path,
                                         transform=TwoCropTransform(train_transform),
                                            train=True,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_path,
                                        transform=val_transform,
                                        train=False,
                                        download=True)


    elif opt.task == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_path,
                                          transform=TwoCropTransform(train_transform),
                                          train=True,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_path,
                                        transform=val_transform,
                                        train=False,
                                        download=True)
    elif opt.task == 'path':
        dataset = datasets.ImageFolder(root=opt.data_path,
                                            transform=TwoCropTransform(train_transform))
        train_dataset, val_dataset = split_dataset(dataset, opt)

    elif opt.task == 'imagenet':
        train_dataset = datasets.ImageFolder(root=opt.imagenet_folder+'train',
                             transform=TwoCropTransform(train_transform),
                             )
        val_dataset = datasets.ImageFolder(root=opt.imagenet_folder+'val',
                                transform=val_transform,
                                )

    elif opt.task == 'bird':
        dataset = datasets.ImageFolder(root=os.path.join(opt.data_path,'cal_bird/CUB_200_2011/images/'),
                                             transform=TwoCropTransform(train_transform),
                                             )
        train_dataset, _ = split_dataset(dataset, opt)
        dataset = datasets.ImageFolder(root=os.path.join(opt.data_path,'cal_bird/CUB_200_2011/images/'),
                                       transform=val_transform,
                                       )
        _, val_dataset = split_dataset(dataset, opt)

    elif opt.task == 'caltech256':
        dataset = datasets.Caltech256(root=opt.data_path,
                                          transform=TwoCropTransformcaltech256(train_transform),
                                          download=True)
        train_dataset, _ = split_dataset(dataset, opt)
        dataset = datasets.Caltech256(root=opt.data_path,
                                            transform=CovertToRGB(val_transform),
                                            download=True)
        _, val_dataset = split_dataset(dataset, opt)

    elif opt.task == 'flowers102':
        train_dataset = datasets.Flowers102(root=opt.data_path,
                                          transform=TwoCropTransform(train_transform),
                                            split='train',
                                          download=True)
        val_dataset = datasets.Flowers102(root=opt.data_path,
                                        transform=val_transform,
                                        split='test',
                                        download=True)

    elif opt.task == 'oxfordpet':
        train_dataset = datasets.OxfordIIITPet(root=opt.data_path,
                                          transform=TwoCropTransform(train_transform),
                                            split='trainval',
                                          download=True)
        val_dataset = datasets.OxfordIIITPet(root=opt.data_path,
                                        transform=val_transform,
                                        split='test',
                                        download=True)
    elif opt.task == 'places365':

        train_dataset = datasets.ImageFolder(root=opt.data_path +'/places365/' +'train',
                             transform=TwoCropTransform(train_transform),
                             )
        val_dataset = datasets.ImageFolder(root=opt.data_path +'/places365/' +'val',
                                transform=val_transform,
                                )

    elif opt.task == 'inat2021':
        train_dataset = datasets.INaturalist(root=opt.data_path,
                                             transform=TwoCropTransform(train_transform),
                                             version='2021_train',
                                             download=True)
        val_dataset = datasets.INaturalist(root=opt.data_path,
                                           transform=val_transform,
                                           version='2021_val',
                                           download=True)
    elif opt.task == 'inat2017':

        train_dataset = DatasetMeta(root=opt.data_path+'/2017/',transform=TwoCropTransform(train_transform),train=True,
                                    dataset='inaturelist2017')
        val_dataset = DatasetMeta(root=opt.data_path+'/2017/',transform=val_transform,train=False,
                                    dataset='inaturelist2017')

    elif opt.task == 'inat2018':
        dataset = datasets.INaturalist(root=opt.data_path,
                                       transform=TwoCropTransformcaltech256(train_transform),
                                       version='2018',)
        train_dataset, _ = split_dataset(dataset, opt)
        dataset = datasets.INaturalist(root=opt.data_path,
                                       transform=CovertToRGB(val_transform),
                                       version='2018',)
        _, val_dataset = split_dataset(dataset, opt)

    else:
        raise ValueError(opt.task)

    if hasattr(opt, "eval_batch_size") and opt.eval_batch_size is not None:
        val_batch_size = opt.eval_batch_size
    else:
        val_batch_size = int(opt.batch_size * 1.5)

    train_dataloader = create_dataloader(
        train_dataset, is_train=True, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_mem=opt.pin_mem, dist_eval=opt.dist_eval,
    )
    val_dataloader = create_dataloader(
        val_dataset, is_train=False, batch_size=val_batch_size,
        num_workers=opt.num_workers, pin_mem=opt.pin_mem, dist_eval=opt.dist_eval,
    )

    print(f"This trianing uses {len(train_dataloader)*opt.batch_size} data. It is {opt.train_ratio} of whole dataset.")
    print(f"This validation uses {len(val_dataloader)*val_batch_size} data. It is {1 - opt.train_ratio} of whole dataset.")

    return train_dataloader, val_dataloader


def create_fewshot_dataset(opt, eval=False):
    train_partition = 'trainval' if opt.use_trainval else 'train'
    if opt.task == 'CIFAR-FS' or opt.task == 'FC100':
        transform = 'D'
    else:
        transform = 'A'

    train_trans, test_trans = transforms_options[transform]
    if opt.task == 'miniImageNet':
        if not eval:
            train_dataset = ImageNet(args=opt, partition=train_partition, transform=TwoCropTransform(train_trans))

            val_dataset = ImageNet(args=opt, partition='val', transform=test_trans)
        else:
            meta_testdataset = MetaImageNet(args=opt, partition='test',
                                                      train_transform=train_trans,
                                                      test_transform=test_trans)
            meta_valdataset = MetaImageNet(args=opt, partition='val',
                                                     train_transform=train_trans,
                                                     test_transform=test_trans)
    elif opt.task == 'tieredImageNet':

        if not eval:
            train_dataset = TieredImageNet(args=opt, partition=train_partition, transform=TwoCropTransform(train_trans))
            val_dataset = TieredImageNet(args=opt, partition='train_phase_val', transform=test_trans)
        else:
            meta_testdataset = MetaTieredImageNet(args=opt, partition='test',
                                                            train_transform=train_trans,
                                                            test_transform=test_trans)
            meta_valdataset = MetaTieredImageNet(args=opt, partition='val',
                                                           train_transform=train_trans,
                                                           test_transform=test_trans)
    elif opt.task == 'CIFAR-FS' or opt.task == "FC100":
        if not eval:
            train_dataset = CIFAR100(args=opt, partition=train_partition, transform=TwoCropTransform(train_trans))
            val_dataset = CIFAR100(args=opt, partition='train', transform=test_trans)
        else:
            meta_testdataset = MetaCIFAR100(args=opt, partition='test',
                                                      train_transform=train_trans,
                                                      test_transform=test_trans)
            meta_valdataset = MetaCIFAR100(args=opt, partition='val',
                                                     train_transform=train_trans,
                                                     test_transform=test_trans)

    if hasattr(opt, "eval_batch_size") and opt.eval_batch_size is not None:
        val_batch_size = opt.eval_batch_size
    else:
        val_batch_size = int(opt.batch_size * 1.5)

    if not eval:
        train_dataloader = create_dataloader(
            train_dataset, is_train=True, batch_size=opt.batch_size,
            num_workers=opt.num_workers, pin_mem=opt.pin_mem, dist_eval=opt.dist_eval,
        )
        val_dataloader = create_dataloader(
            val_dataset, is_train=False, batch_size=val_batch_size,
            num_workers=opt.num_workers, pin_mem=opt.pin_mem, dist_eval=opt.dist_eval,
        )

        return train_dataloader, val_dataloader
    else:
        meta_valloader = create_dataloader(
            meta_valdataset, is_train=False, batch_size=1,
            num_workers=opt.num_workers, pin_mem=opt.pin_mem, dist_eval=opt.dist_eval,
        )
        meta_testloader = create_dataloader(
            meta_testdataset, is_train=False, batch_size=1,
            num_workers=opt.num_workers, pin_mem=opt.pin_mem, dist_eval=opt.dist_eval,
        )
        return meta_valloader, meta_testloader