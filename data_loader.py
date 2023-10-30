from torchvision import datasets, transforms
import torch
import os
from PIL import Image
import random

def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2),
        transforms.ToTensor()])
    data = CustomImageFolder(root=os.path.join(root_path, dir), transform=[transform,transform])
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True ,**kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images

def make_dataset(root):
    dir_names = []
    images = []
    for dir_name in os.listdir(root):
        if os.path.isdir(os.path.join(root,dir_name)):
            dir_names.append(dir_name)
    dir_names = sorted(dir_names)
    dir_names = {v:k for k,v in enumerate(dir_names)}
    for dir_name in os.listdir(root):
        if os.path.isdir(os.path.join(root,dir_name)):
            for item in os.listdir(os.path.join(root,dir_name)):
                images.append((os.path.join(root,dir_name,item),dir_names[dir_name]))
    random.shuffle(images)
    classes = dir_names
    return images,classes

def default_loader(path):
    return Image.open(path).convert('RGB')

class ObjectImage(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        imgs,classes = make_dataset(root)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.classes = classes

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)      
        return img, target

    def __len__(self):
        return len(self.imgs)

class ObjectImage_mul(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        imgs,classes = make_dataset(root)
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.classes = classes

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            if type(self.transform).__name__=='list':
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
    def __getitem__(self, index1):
        path1 = self.imgs[index1][0] 
        label1 = self.imgs[index1][1]
        img1 = self.loader(path1)
        if isinstance(self.transform,list):
            images = []
            for t in self.transform:
                images.append(t(img1))
            return images,label1,index1
        if self.transform is not None:
            img1 = self.transform(img1)
        return img1, label1,index1