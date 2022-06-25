import os
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, random_split
from einops import repeat
from torchvision import transforms as T
import random
from utils import Denormalize

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def augmentation(crop_size=224, crop_scale=(0.5,1.0), jitter_params=(0.4, 0.4, 0.4, 0.1), jitter_prob=0.8,
                gray_prob=0.2, gaussian_sigma=[0.1, 2.0], gaussian_prob=0.5):
    aug = T.Compose([T.ToPILImage(),
                     T.RandomResizedCrop(crop_size, scale=crop_scale),
                     T.RandomApply([
                         T.ColorJitter(*jitter_params)  # not strengthened
                     ], p=jitter_prob),
                     T.RandomGrayscale(p=gray_prob),
                     T.RandomApply([GaussianBlur(gaussian_sigma)], p=gaussian_prob),
                     T.RandomHorizontalFlip(),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    return aug

class ImageNet(Dataset):
    def __init__(self, root, mode, file_pathes, transforms, class_num_dict, class_name_dict, test_labels=None):
        super(ImageNet, self).__init__()
        self.root = root
        self.mode = mode
        self.file_pathes = file_pathes
        self.transforms = transforms
        self.class_num_dict = class_num_dict
        self.class_name_dict = class_name_dict
        self.test_labels = test_labels
        
        self.num_class_dict = dict()
        
    def __len__(self):
        return len(self.file_pathes)
    
    def __getitem__(self, index):
        image = T.ToTensor()(Image.open(os.path.join(self.root, self.mode, self.file_pathes[index])))
        if image.size(0) != 3:
            image = repeat(image[0], 'h w -> c h w', c=3)
            
        image = self.transforms(image)
        
        c = self.file_pathes[index].split('/')[0] if self.mode == 'train' else self.test_labels[index].split()[1]
        
        label = self.class_num_dict[c]
        
        return image, label
    
    def get_label_name(self, label_num):
        if len(self.num_class_dict.keys()) == 0:
            for k, v in self.class_num_dict.items():
                self.num_class_dict[v] = k
        return self.class_name_dict[self.num_class_dict[label_num]]
    
class ImageNetSR(ImageNet):
    def __init__(self, root, mode, file_pathes, transforms, class_num_dict, class_name_dict, test_labels=None):
        super(ImageNetSR, self).__init__(root, mode, file_pathes, transforms, class_num_dict, class_name_dict, test_labels)
        self.reduction = T.Compose([Denormalize(),
                                    T.Resize((56,56)),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    def __getitem__(self, index):
        image = T.ToTensor()(Image.open(os.path.join(self.root, self.mode, self.file_pathes[index])))
        if image.size(0) != 3:
            image = repeat(image[0], 'h w -> c h w', c=3)
            
        image = self.transforms(image)
        reducted_image = self.reduction(image)
        
        c = self.file_pathes[index].split('/')[0] if self.mode == 'train' else self.test_labels[index].split()[1]
        
        return image, reducted_image
    
def make_dataset(root, mode, file_pathes, transforms, class_num_dict, class_name_dict, 
                 test_labels=None, validation=False, val_size=None, purpose=None):
    if mode == 'test': mode = 'val'
    assert purpose in [None, 'SR'], "'purpose' should be None, or SR"
    if purpose is None:
        dataset = ImageNet(root, mode, file_pathes, transforms, class_num_dict, class_name_dict, test_labels)
    elif purpose == 'SR':
        dataset = ImageNetSR(root, mode, file_pathes, transforms, class_num_dict, class_name_dict, test_labels)
    
    if mode == 'train' and validation:
        assert val_size is not None, "validation set size should be set!"
        val_size = int(len(dataset)*val_size) if isinstance(val_size, float) else val_size
        train_data, val_data = random_split(dataset, [len(dataset)-val_size, val_size])
        
        return train_data, val_data
    
    elif mode == 'train' and not validation:
        return dataset
        
    elif mode == 'val' and not validation:
        assert test_labels is not None, "test data should have test labels!"
        return dataset
    
    elif mode == 'val' and validation:
        assert val_size is not None, "validation set size should be set!"
        val_size = int(len(dataset)*val_size) if isinstance(val_size, float) else val_size
        train_data, val_data = random_split(dataset, [len(dataset)-val_size, val_size])
        
        return train_data, val_data