import random
import os.path
import PIL.Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as vision_tf

from util.imutils import RandomResizeLong,\
    random_crop_with_saliency, HWC_to_CHW, Normalize


def load_img_id_list(img_id_file):
    img_gt_name_list = open(img_id_file).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][12:-4] for img_gt_name in img_gt_name_list]
    return img_name_list




def load_img_label_list_from_npy(img_name_list, dataset):
    cls_labels_dict = np.load(f'metadata/{dataset}/cls_labels.npy', allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_reliable_path(img_name, reliable_root='SALImages'):
    return os.path.join(reliable_root, img_name + '.png')


class ImageDataset(Dataset):
    """
    Base image dataset. This returns 'img_id' and 'image'
    """
    def __init__(self, dataset, img_id_list_file, img_root, transform=None):
        self.dataset = dataset
        self.img_id_list = load_img_id_list(img_id_list_file)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.img_id_list)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = PIL.Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img_id, img


class ClassificationDataset(ImageDataset):
    """
    Classification Dataset (base)
    """
    def __init__(self, dataset, img_id_list_file, img_root, transform=None):
        super().__init__(dataset, img_id_list_file, img_root, transform)
        self.label_list = load_img_label_list_from_npy(self.img_id_list, dataset)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)
        label = torch.from_numpy(self.label_list[idx])
        return name, img, label


class ClassificationDatasetWithReliable_root(ImageDataset):

    def __init__(self, dataset, img_id_list_file, img_root, reliable_root=None, crop_size=224, resize_size=(256, 512)):
        super().__init__(dataset, img_id_list_file, img_root, transform=None)
        self.reliable_root = reliable_root
        self.crop_size = crop_size
        self.resize_size = resize_size

        self.resize = RandomResizeLong(resize_size[0], resize_size[1])
        self.color = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        self.normalize = Normalize()

        self.label_list = load_img_label_list_from_npy(self.img_id_list, dataset)

    def __getitem__(self, idx):
        img_id = self.img_id_list[idx]
        img = PIL.Image.open(os.path.join(self.img_root, img_id + '.jpg')).convert("RGB")
        reliable = PIL.Image.open(get_reliable_path(img_id, self.reliable_root))#pil 1024 682
        img, reliable = self.transform_with_mask(img, reliable)#

        label = torch.from_numpy(self.label_list[idx])
        return img_id, img, reliable, label

    def transform_with_mask(self, img, mask):
        # randomly resize
        # target_size = random.randint(self.resize_size[0], self.resize_size[1])
        # img = self.resize(img, target_size)#429 286     889*592
        # mask = self.resize(mask, target_size)#429 286     889*592

        # randomly flip
        if random.random() > 0.5:
            img = vision_tf.hflip(img)
            mask = vision_tf.hflip(mask)

        # add color jitter
        img = self.color(img)

        img = np.asarray(img)
        mask = np.asarray(mask)

        # normalize
        img = self.normalize(img)#286 429 3


        # permute the order of dimensions
        img = HWC_to_CHW(img)
        # mask = HWC_to_CHW(mask)

        # make tensor
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask).unsqueeze(0)
        # a = torch.mean(img, dim=0, keepdim=True)
        # img = torch.from_numpy(img)
        # mask = torch.from_numpy(mask)
        # mask = torch.mean(mask, dim=0, keepdim=True)


        return img, mask