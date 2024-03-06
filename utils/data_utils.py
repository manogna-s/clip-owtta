from utils.datasets import CIFAR10, CIFAR100, CIFAR100_openset, CIFAR10_openset, \
     noise_dataset, MNIST_openset, SVHN_openset, TinyImageNet_OOD_nonoverlap, ImageNetR, VISDA
import json
import torch
import os
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageNet


cifar_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

imagenet_templates = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

class ImageNetCorruption(ImageNet):
    def __init__(self, root, corruption_name="gaussian_noise", transform=None, is_carry_index=False):
        super().__init__(root, 'val', transform=transform)
        self.root = root
        self.corruption_name = corruption_name
        self.transform = transform
        self.is_carry_index = is_carry_index
        self.load_data()
    
    def load_data(self):
        self.data = torch.load(os.path.join(self.root, 'corruption', self.corruption_name + '.pth')).numpy()
        self.target = [i[1] for i in self.imgs]
        return
    
    def __getitem__(self, index):
        img = self.data[index, :, :, :]
        target = self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.is_carry_index:
            img = [img, index]
        return img, target
    
    def __len__(self):
        return self.data.shape[0]

class ImageNet_(ImageNet):
    def __init__(self, *args, is_carry_index=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_carry_index = is_carry_index
    
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        if self.is_carry_index:
            if type(img) == list:
                img.append(index)
            else:
                img = [img, index]
        return img, target

def prepare_ood_test_data(args, te_transforms):

    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    data_dict = {}
    if args.dataset == 'cifar10OOD':
        # self.classnames = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
        # self.lab2cname = self.classnames
        ID_dataset = 'cifar10'
        data_dict['ID_class_descriptions'] = json.load(open(f'/home/manogna/TTA/PromptAlign/data/ood/prompt_templates/{ID_dataset}_prompts_full.json'))
        data_dict['ID_classes'] = list(data_dict['ID_class_descriptions'].keys())
        data_dict['N_classes'] = len(data_dict['ID_classes'])
        data_dict['templates'] = cifar_templates

        tesize = 10000
        if args.corruption in common_corruptions:
            
            print('Test on %s level %d' %(args.corruption, args.level))
            teset_raw_100 = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
            teset_raw_100 = teset_raw_100[(args.level-1)*tesize: args.level*tesize]
            teset_raw_10 = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
            teset_raw_10 = teset_raw_10[(args.level-1)*tesize: args.level*tesize]
            teset_10 = CIFAR10(root=args.dataroot,
                            train=False, download=True, transform=te_transforms)
            teset_10.data = teset_raw_10

            if args.strong_OOD == 'MNIST':
                te_rize = transforms.Compose([transforms.Resize(size=(32, 32)), transforms.Grayscale(3), te_transforms ])
                noise = MNIST_openset(root=args.dataroot,
                            train=False, download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([teset_10,noise])
            
            elif args.strong_OOD == 'noise':
                noise = noise_dataset(te_transforms, args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([teset_10,noise])

            elif args.strong_OOD =='cifar100':
                teset_raw_100 = np.load(args.dataroot + '/CIFAR-100-C/snow.npy')
                teset_raw_100 = teset_raw_100[(args.level-1)*tesize: args.level*tesize]
                teset_100 = CIFAR100_openset(root=args.dataroot,
                                train=False, download=True, transform=te_transforms, tesize=tesize, ratio=args.strong_ratio)
                teset_100.data = teset_raw_100[:int(10000*args.strong_ratio)]
                teset = torch.utils.data.ConcatDataset([teset_10,teset_100])

            elif args.strong_OOD =='SVHN': 
                te_rize = transforms.Compose([te_transforms ])
                noise = SVHN_openset(root=args.dataroot,
                            split='test', download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([teset_10,noise])
                
            elif args.strong_OOD =='Tiny':

                transform_test = transforms.Compose([transforms.Resize(32), te_transforms ])
                testset_tiny = TinyImageNet_OOD_nonoverlap(args.dataroot +'/tiny-imagenet-200', transform=transform_test, train=True)
                teset = torch.utils.data.ConcatDataset([teset_10,testset_tiny])
                print(len(teset_10),len(testset_tiny),len(teset))
                
            else:
                raise

    elif args.dataset == 'cifar100OOD':
        
        tesize = 10000
        ID_dataset = 'cifar100'
        data_dict['ID_class_descriptions'] = json.load(open(f'/home/manogna/TTA/PromptAlign/data/ood/prompt_templates/{ID_dataset}_prompts_full.json'))
        data_dict['ID_classes'] = list(data_dict['ID_class_descriptions'].keys())
        data_dict['N_classes'] = len(data_dict['ID_classes'])
        data_dict['templates'] = cifar_templates


        if args.corruption in common_corruptions:
            print('Test on %s level %d' %(args.corruption, args.level))
            teset_raw_100 = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
            teset_raw_100 = teset_raw_100[(args.level-1)*tesize: args.level*tesize]
            teset_raw_10 = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
            teset_raw_10 = teset_raw_10[(args.level-1)*tesize: args.level*tesize]
            teset_100 = CIFAR100(root=args.dataroot,
                            train=False, download=True, transform=te_transforms)
            teset_100.data = teset_raw_100

            if args.strong_OOD == 'MNIST':
                te_rize = transforms.Compose([transforms.Resize(size=(32, 32)), transforms.Grayscale(3), te_transforms ])
                noise = MNIST_openset(root=args.dataroot,
                            train=False, download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([teset_100,noise])
            
            elif args.strong_OOD == 'noise':
                noise = noise_dataset(te_transforms, args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([teset_100,noise])

            elif args.strong_OOD =='cifar10':
                teset_raw_10 = np.load(args.dataroot + '/CIFAR-10-C/snow.npy')
                teset_raw_10 = teset_raw_10[(args.level-1)*tesize: args.level*tesize]
                teset_10 = CIFAR10_openset(root=args.dataroot,
                                train=False, download=True, transform=te_transforms, tesize=tesize, ratio=args.strong_ratio)
                teset_10.data = teset_raw_10[:int(10000*args.strong_ratio)]
                teset = torch.utils.data.ConcatDataset([teset_100,teset_10])

            elif args.strong_OOD =='SVHN': 
                te_rize = transforms.Compose([te_transforms ])
                noise = SVHN_openset(root=args.dataroot,
                            split='test', download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([teset_100,noise])
                
            elif args.strong_OOD =='Tiny':

                transform_test = transforms.Compose([transforms.Resize(32), te_transforms ])
                testset_tiny = TinyImageNet_OOD_nonoverlap(args.dataroot +'/tiny-imagenet-200', transform=transform_test, train=True)
                teset = torch.utils.data.ConcatDataset([teset_100,testset_tiny])

            else:
                raise
    
    elif args.dataset == 'ImagenetROOD':
        # self.classnames = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
        # self.lab2cname = self.classnames
        ID_dataset = 'imagenetr'
        imagenet_descriptions = json.load(open(f'/home/manogna/TTA/PromptAlign/data/ood/prompt_templates/imagenetr_prompts_full.json'))
        # data_dict['ID_classes'] = list(data_dict['ID_class_descriptions'].keys())

        testset = ImageNetR(root= args.dataroot)
        print(testset.classnames)
        data_dict['ID_classes'] = testset.classnames
        data_dict['N_classes'] = len(data_dict['ID_classes'])
        data_dict['templates'] = imagenet_templates
        data_dict['ID_class_descriptions'] = {}
        for classname in data_dict['ID_classes']:
            data_dict['ID_class_descriptions'][classname] = imagenet_descriptions[classname]
        # json.dump(data_dict['ID_class_descriptions'], open(f'/home/manogna/TTA/PromptAlign/data/ood/prompt_templates/imagenetr_prompts_full.json', 'w'))

        tesize = 10000
        testset = ImageNetR(root= args.dataroot, transform=te_transforms, train=True, tesize=tesize)

        if True: 

            if args.strong_OOD == 'MNIST':
                te_rize = transforms.Compose([transforms.Resize(size=(32, 32)), transforms.Grayscale(3), te_transforms ])
                noise = MNIST_openset(root=args.dataroot,
                            train=True, download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([testset,noise])
                print(len(testset), len(noise), len(teset))
            
            elif args.strong_OOD == 'noise':
                noise = noise_dataset(te_transforms, args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([testset,noise])

            elif args.strong_OOD =='cifar10':
                teset_raw_10 = np.load(args.dataroot + '/CIFAR-10-C/snow.npy')
                teset_raw_10 = teset_raw_10[(args.level-1)*tesize: args.level*tesize]
                teset_10 = CIFAR10_openset(root=args.dataroot,
                                train=True, download=True, transform=te_transforms, tesize=tesize, ratio=args.strong_ratio)
                teset_10.data = teset_raw_10[:int(tesize*args.strong_ratio)]
                teset = torch.utils.data.ConcatDataset([testset,teset_10])

            elif args.strong_OOD =='cifar100':
                teset_raw_100 = np.load(args.dataroot + '/CIFAR-100-C/snow.npy')
                teset_raw_100 = teset_raw_100[(args.level-1)*tesize: args.level*tesize]
                teset_100 = CIFAR100_openset(root=args.dataroot,
                                train=True, download=True, transform=te_transforms, tesize=tesize, ratio=args.strong_ratio)
                teset_100.data = teset_raw_100[:int(tesize*args.strong_ratio)]
                teset = torch.utils.data.ConcatDataset([testset,teset_100])

            elif args.strong_OOD =='SVHN': 
                te_rize = transforms.Compose([te_transforms ])
                noise = SVHN_openset(root=args.dataroot,
                            split='train', download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)
                teset = torch.utils.data.ConcatDataset([testset,noise])
                print(len(testset), len(noise), len(teset))
                
            elif args.strong_OOD =='Tiny':

                transform_test = transforms.Compose([transforms.Resize(32), te_transforms ])
                testset_tiny = TinyImageNet_OOD_nonoverlap(args.dataroot +'/tiny-imagenet-200', transform=transform_test, train=True)
                teset = torch.utils.data.ConcatDataset([testset,testset_tiny])
                # print(len(teset_10),len(testset_tiny),len(teset))
                
            else:
                raise
    
    elif args.dataset == "VisdaOOD":
        tesize = 10000
        ID_dataset = 'visda'
        data_dict['ID_class_descriptions'] = json.load(open(f'/home/manogna/TTA/PromptAlign/data/ood/prompt_templates/{ID_dataset}_prompts_full.json'))
        data_dict['ID_classes'] = list(data_dict['ID_class_descriptions'].keys())
        data_dict['N_classes'] = len(data_dict['ID_classes'])
        data_dict['templates'] = cifar_templates

        testset = VISDA(root= f'{args.dataroot}/visda-2017', label_files=f'{args.dataroot}/visda-2017/validation_list.txt' , transform=te_transforms, tesize=tesize)

        if True: 

            if args.strong_OOD == 'MNIST':
                te_rize = transforms.Compose([transforms.Resize(size=(32, 32)), transforms.Grayscale(3), te_transforms ])
                noise = MNIST_openset(root=args.dataroot,
                            train=True, download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)

                teset = torch.utils.data.ConcatDataset([testset,noise])
            
            elif args.strong_OOD =='SVHN': 
                te_rize = transforms.Compose([te_transforms ])
                noise = SVHN_openset(root=args.dataroot,
                            split='train', download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)
                teset = torch.utils.data.ConcatDataset([testset,noise])
            
            print(len(testset), len(noise), len(teset))

    ID_OOD_loader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=True)

    return data_dict, teset, ID_OOD_loader


# TPT Transforms

# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()   # Resizing with scaling and ratio
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views
