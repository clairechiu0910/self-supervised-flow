import os
import cv2
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

from dataset.self_supervised_tasks import patch_ex
from dataset.self_supervised_tasks_constants import *

GREY_SCALE_CLASS = [
    'zipper',
    'screw',
    'grid',
    'Class1',
    'Class2',
    'Class3',
    'Class4',
    'Class5',
    'Class6',
    'Class7',
    'Class8',
    'Class9',
    'Class10',
]


class ImageDataset(Dataset):

    def __init__(self, c, is_train=True, synthesis_args=None):
        self.normal_type = c.normal_type
        self.dataset_path = c.dataset_path

        self.dataset = c.dataset
        self.class_name = c.class_name

        self.is_train = is_train
        self.img_size = c.img_size
        self.few_shot = c.few_shot

        if is_train:
            self.repeat = c.repeat
        else:
            self.repeat = 1

        # load dataset
        self.x, self.y, self.mask, self.label_types = self.load_dataset_folder()
        self.dataset_len = len(self.x)
        # set transforms
        if is_train:
            self.transform_x = transforms.Compose(
                [transforms.Resize(c.img_size, Image.ANTIALIAS),
                 transforms.RandomAdjustSharpness(sharpness_factor=2)])
        # test:
        else:
            self.transform_x = transforms.Compose([transforms.Resize(c.img_size, Image.ANTIALIAS)])
        # mask
        self.transform_mask = transforms.Compose([transforms.Resize(c.img_size, Image.NEAREST), transforms.ToTensor()])

        self.normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(c.norm_mean, c.norm_std)])

        # synthesis
        self.prev_idx = np.random.randint(len(self.x))
        self.synthesis_args = synthesis_args

    def __getitem__(self, idx):
        actual_idx = idx % self.dataset_len

        x, y, mask = self.x[actual_idx], self.y[actual_idx], self.mask[actual_idx]

        # mask
        if y == 0:
            mask = torch.zeros([1, self.img_size[0], self.img_size[1]])
        else:
            if mask is None:
                mask = torch.ones([1, self.img_size[0], self.img_size[1]])
            else:
                mask = Image.open(mask)
                mask = self.transform_mask(mask)

        x = Image.open(x)
        x = self.handle_greyscale_image(x)
        x = self.transform_x(x)

        # get synthetic images
        if self.is_train:
            p = self.x[self.prev_idx]
            p = Image.open(p)
            p = self.handle_greyscale_image(p)
            p = self.transform_x(p)
            p = np.asarray(p)
            x = np.asarray(x)

            syn_x, syn_mask = patch_ex(x, p, **self.synthesis_args)
            self.prev_idx = actual_idx
        else:
            syn_x = x
            syn_mask = mask

        x = self.normalize(x)
        syn_x = self.normalize(syn_x)

        return x, y, mask, syn_x, syn_mask

    def __len__(self):
        return self.dataset_len * self.repeat

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask, label_types = [], [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        path_types = sorted(os.listdir(img_dir))
        label_types.append(self.normal_type)
        for ptype in path_types:
            if ptype != self.normal_type:
                label_types.append(ptype)

        for idx in range(len(label_types)):
            label_type = label_types[idx]

            # load images
            img_type_dir = os.path.join(img_dir, label_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir)])
            x.extend(img_fpath_list)

            # load gt labels
            if label_type == self.normal_type:
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([idx] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, label_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]

                gt_fpath_list = []
                if self.dataset == 'MVTec':
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png') for img_fname in img_fname_list]
                elif self.dataset == 'BTAD':
                    if self.class_name == '03':
                        gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.bmp') for img_fname in img_fname_list]
                    else:
                        gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png') for img_fname in img_fname_list]
                elif self.dataset == 'DAGM':
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_label.PNG') for img_fname in img_fname_list]

                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        if phase == 'train' and self.few_shot != 0:
            select_idx = random.sample(range(len(x)), self.few_shot)
            x = [x[i] for i in select_idx]

        return list(x), list(y), list(mask), list(label_types)

    def handle_greyscale_image(self, x):
        if self.class_name in GREY_SCALE_CLASS:
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)

            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        return x

    def configure_self_sup(self, on=True, synthesis_args={}):
        self.self_sup = on
        self.synthesis_args.update(synthesis_args)


def load_datasets(c):
    trainset = ImageDataset(c,
                            is_train=True,
                            synthesis_args={
                                'gamma_params': (2, 0.05, 0.03),
                                'resize': True,
                                'shift': True,
                                'same': False,
                                'mode': 1,
                                'label_mode': 'logistic-intensity'
                            })

    trainset.configure_self_sup(
        synthesis_args={
            'gamma_params': (2, 0.05, 0.03),
            'resize': True,
            'shift': True,
            'same': False,
            'mode': cv2.NORMAL_CLONE,
            'label_mode': 'logistic-intensity'
        })
    trainset.configure_self_sup(synthesis_args={'skip_background': BACKGROUND.get(c.class_name)})
    if c.class_name in TEXTURES:
        trainset.configure_self_sup(synthesis_args={'resize_bounds': (.5, 2)})
    trainset.configure_self_sup(on=True,
                                synthesis_args={
                                    'width_bounds_pct': WIDTH_BOUNDS_PCT.get(c.class_name),
                                    'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(c.class_name),
                                    'num_patches': NUM_PATCHES.get(c.class_name),
                                    'min_object_pct': MIN_OBJECT_PCT.get(c.class_name),
                                    'min_overlap_pct': MIN_OVERLAP_PCT.get(c.class_name)
                                })

    testset = ImageDataset(c, is_train=False)

    return trainset, testset


def make_dataloaders(c, trainset, testset):
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=c.batch_size, shuffle=True, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=c.batch_size, shuffle=False, drop_last=False)
    return trainloader, testloader


def prepare_dataloaders(c):
    train_set, test_set = load_datasets(c)
    train_loader, test_loader = make_dataloaders(c, train_set, test_set)
    print('train/test loader length', len(train_loader.dataset), len(test_loader.dataset))
    print('train/test loader batches', len(train_loader), len(test_loader))

    return train_loader, test_loader
