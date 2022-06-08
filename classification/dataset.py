import torch
import re
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from utils.pyutils import multiscale_online_crop
from torchvision import transforms

def get_file_label(filename, num_class=3):
    l = []
    begin = -6
    for i in range(num_class):
        l.insert(0, int(filename[begin-3*i]))
    return np.array(l)

class OriginPatchesDataset(Dataset):
    def __init__(self, data_path_name=None, transform=None, num_class=2):
        self.path = data_path_name
        self.files = os.listdir(data_path_name)
        self.transform = transform
        self.filedic = {}
        self.num_class = num_class

    def __len__(self):
        return len(self.files)
        # return 50

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = Image.open(image_path)
        label = get_file_label(filename=self.files[idx], num_class=self.num_class)
        area = None
        if self.cutmix_fn and label.sum() == 1:
            # This cutmix and area regression part is exclusively for the luad dataset with three class
            activate = np.random.randint(3)
            mixcategory = np.array((0, 0, 0))
            mixcategory[activate] = 1
            mixcategory = tuple(mixcategory)
            # randomly select a image in that category
            if mixcategory != tuple(label):
                pick = np.random.randint(len(self.filedic[mixcategory]))
                miximage = Image.open(os.path.join(self.path, self.filedic[mixcategory][pick]))
                im = transforms.ToTensor()(im)
                miximage = transforms.ToTensor()(miximage)
                im, ratiox, ratioy = self.cutmix_fn(im, miximage, label)
                area = label.astype(np.float32) * ratiox
                area[activate] = ratioy
                label = np.logical_or(label, np.array(mixcategory)).astype(np.int32)
            else:
                im = transforms.ToTensor()(im)
                area = label.astype(np.float32)
            
        else:
            im = transforms.ToTensor()(im)
            area = np.full(self.num_class, -1.).astype(np.float32)

        if self.transform:
            im = self.transform(im)
        
        return im, label, area

class OnlineDataset(Dataset):
    def __init__(self, data_path_name, transform, patch_size, stride, scales):
        self.path = data_path_name
        self.files = os.listdir(data_path_name)
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.scales = scales

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = np.asarray(Image.open(image_path))
        scaled_im_list, scaled_position_list = multiscale_online_crop(im, self.patch_size, self.stride, self.scales)
        if self.transform:
            for im_list in scaled_im_list:
                for patch_id in range(len(im_list)):
                    im_list[patch_id] = self.transform(im_list[patch_id])

        return self.files[idx], scaled_im_list, scaled_position_list, self.scales

class OfflineDataset(Dataset):
    def __init__(self, dataset_path, transform):
        self.path = dataset_path
        self.files = os.listdir(self.path)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = Image.open(image_path)
        positions = self.files[idx]
        positions = list(map(lambda x: int(x), re.findall(r'\d+', positions)))
        if self.transform:
            im = self.transform(im)
        return im, np.array(positions)

class TrainingSetCAM(Dataset):
    def __init__(self, data_path_name, transform, patch_size, stride, scales, num_class):
        self.path = data_path_name
        self.files = os.listdir(data_path_name)
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.scales = scales
        self.num_class = num_class

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.files[idx])
        im = np.asarray(Image.open(image_path))
        scaled_im_list, scaled_position_list = multiscale_online_crop(im, self.patch_size, self.stride, self.scales)
        if self.transform:
            for im_list in scaled_im_list:
                for patch_id in range(len(im_list)):
                    im_list[patch_id] = self.transform(im_list[patch_id])
        if self.num_class == 0:
            label = np.array([0])
        else:
            label = get_file_label(image_path, num_class=self.num_class)

        return self.files[idx], scaled_im_list, scaled_position_list, self.scales, label
