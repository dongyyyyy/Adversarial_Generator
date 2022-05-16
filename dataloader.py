import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import os 
from PIL import Image
import random

def get_train_valid_test_loader(data_dir='./cifar',
                           batch_size=512,
                           image_size=224,
                           augment=True,
                           random_seed=2,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=True):

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    batch_size = int(batch_size)
    # normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    # )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(int(image_size)),
            # normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            # transforms.RandomCrop(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Resize(int(image_size)),
            # normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(int(image_size)),
            # normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=valid_transform,
    )
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )


    return train_loader, valid_loader, test_loader


class get_pascalvoc_loader(object):
    def __init__(self,data_dir='/data/ssd2/VOC2012/JPEGImages/',
                           image_size=224,
                           augment=True,
                           random_seed=2,
                           shuffle=True,
                           make_samples=1000):

        self.data_dir = data_dir
        self.dataset_list = os.listdir(data_dir)
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(self.dataset_list)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(int(image_size)),
            # normalize,
        ])
        
        self.make_samples = make_samples
    def __getitem__(self,index):
        image = self.transform(Image.open(self.data_dir+self.dataset_list[index]))
        return image

    def __len__(self):
        return self.make_samples

def split_dataset(dataset_path,train_split=0.8):
    train_files = []
    val_files = []
    dataset_list = os.listdir(dataset_path)

    for dataset_folder in dataset_list:
        signals_path = dataset_path+dataset_folder+'/'
        signals_list = os.listdir(signals_path)
        random.shuffle(signals_list)
        length = len(signals_list)//2
        train_length = int(length*train_split)
        # val_length = int(length - train_length)
        index = 0
        for signals_filename in signals_list:
            if signals_filename.split('.')[-1] == 'png':
                signals_file = signals_path + signals_filename
                if train_length > index:
                    train_files.append(signals_file)
                else:
                    val_files.append(signals_file)
                index += 1

    return train_files,val_files

class created_image_dataloader(object):
    def __init__(self,
                 dataset_list,
                 image_size=224
                 ):

        
        self.image_files_path = dataset_list
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(int(image_size)),
            # normalize,
        ])
        
    def __getitem__(self, index):
        image = self.transform(Image.open(self.image_files_path[index]))
        #/home/eslab/dataset/created_image_withoutNorm_final/epsilon_0.01/threshold_0.9/0_success/
        labels = int(self.image_files_path.split('/')[-2])
        print(labels)
        exit(1)
        return image, labels

    def __len__(self):
        return len(self.image_files_path)