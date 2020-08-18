import os.path as osp
from PIL import Image
import math
import random
from collections import defaultdict

import torchvision.transforms as tv_transforms
import torch.utils.data

from .market1501 import Market1501
from .dukemtm import DukeMTMCreID
from .msmt17 import MSMT17
from .cuhk03 import CUHK03Detected, CUHK03Labeled
from .sampler import RandomIdentitySampler

dataset_factory = {'market1501': Market1501,
                   'dukemtm': DukeMTMCreID,
                   'msmt17': MSMT17,
                   'cuhk03-detected': CUHK03Detected,
                   'cuhk03-labeled': CUHK03Labeled}


def construct_dataset(args, config):
    normalize = tv_transforms.Normalize(mean=config.dataset_parameters['mean'],
                                        std=config.dataset_parameters['std'])

    if 'random_scale' in config.dataset_parameters:
        train_transform_list = [
            Random2DTranslation(*config.dataset_parameters['crop_size']),
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.ToTensor(),
            normalize,
        ]
    else:
        train_transform_list = [
            tv_transforms.Resize(
                config.dataset_parameters['resize_size'], interpolation=config.dataset_parameters['resize_interpolation']),
            tv_transforms.RandomHorizontalFlip(),
        ]
        if 'pad_size' in config.dataset_parameters:
            train_transform_list += [
                tv_transforms.Pad(config.dataset_parameters['pad_size']),
                tv_transforms.RandomCrop(
                    config.dataset_parameters['crop_size'])
            ]
        train_transform_list += [tv_transforms.ToTensor(), normalize]

    if 'random_erasing_prob' in config.dataset_parameters:
        train_transform_list.append(RandomErasing(
            probability=config.dataset_parameters['random_erasing_prob'], mean=config.dataset_parameters['random_erasing_mean']))

    train_transform = tv_transforms.Compose(train_transform_list)
    val_size = config.dataset_parameters['crop_size'] if 'crop_size' in config.dataset_parameters else config.dataset_parameters['resize_size']
    """
    val_transform = tv_transforms.Compose([
        tv_transforms.Resize(
            val_size, interpolation=3),
        tv_transforms.ToTensor(),
        normalize,
    ])
    """
    val_transform = tv_transforms.Compose([
        tv_transforms.Resize(
            val_size, interpolation=config.dataset_parameters['resize_interpolation']),
        tv_transforms.ToTensor(),
        normalize,
    ])

    dataset = dataset_factory[args.dataset_name](args.data_directory)
    if args.flag == 'train':
        if 'train-all' in args.version:
            train_dataset = ImageDataset(dataset.train, train_transform)
            val_dataset = None
        else:
            # split train/val
            train, val = split_train_val(dataset.train)
            train_dataset = ImageDataset(train, train_transform)
            val_dataset = ImageDataset(val, val_transform)

        # training sampler setup
        if config.dataset_parameters['sampler'] == 'triplet':
            sampler = RandomIdentitySampler(train_dataset, **config.dataset_parameters['sampler_paras'])
            shuffle = False
        elif config.dataset_parameters['sampler'] == 'random':
            sampler = None
            shuffle = True
        else:
            raise NotImplementedError

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True, sampler=sampler, drop_last=True)

        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.validation_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        else:
            val_loader = None

    query_dataset = ImageDataset(dataset.query, val_transform)
    gallery_dataset = ImageDataset(dataset.gallery, val_transform)

    test_loaders = []
    test_batch_size = args.validation_batch_size if args.flag == 'train' else args.batch_size
    for test_dataset in [query_dataset, gallery_dataset]:
        test_loaders.append(torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True))

    if 'id' in args.version:
        id_dataset = ImageDataset(dataset.train, val_transform)
        id_loader = torch.utils.data.DataLoader(
            id_dataset, batch_size=args.validation_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.flag == 'train':
        if 'id' in args.version:
            return train_loader, val_loader, test_loaders[0], test_loaders[1], id_loader
        return train_loader, val_loader, test_loaders[0], test_loaders[1]
    else:
        return test_loaders


def split_train_val(train_all):
    print('==> split training and validation dataset')
    pid_count = defaultdict(lambda: 0)
    train = []
    val = []
    for img_file, label in train_all:
        pid = label['pid']
        pid_count[pid] += 1
        if pid_count[pid] == 1:
            val.append([img_file, label])
        else:
            train.append([img_file, label])
    training_pid_count = 0
    for key in pid_count:
        if pid_count[key] != 1:
            training_pid_count += 1
    print('    --> training all: {}'.format(len(train_all)))
    print('    --> total pids: {}'.format(len(pid_count.keys())))
    print('    --> training split: {}'.format(len(train)))
    print('    --> training pids: {}'.format(training_pid_count))
    print('    --> val split: {}'.format(len(val)))
    return train, val


############ a lot copy from reid-strong-baseline ###################


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                img_path))
            pass
    return img


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, label = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.
    Args:
    - height (int): target image height.
    - width (int): target image width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(
            round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop(
            (x1, y1, x1 + self.width, y1 + self.height))
        return croped_img
