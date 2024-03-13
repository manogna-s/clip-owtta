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
import utils.augmix_ops as augmentations
from utils.robustbench_loader import CustomImageFolder


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


def get_weak_ood_data(args, te_transforms, tesize=10000):
    data_dict = {}
    if args.dataset == 'cifar10OOD':
        ID_dataset = 'cifar10'
        data_dict['ID_class_descriptions'] = json.load(open(f'{args.dataroot}/prompt_templates/{ID_dataset}_prompts_full.json'))
        data_dict['ID_classes'] = list(data_dict['ID_class_descriptions'].keys())
        data_dict['N_classes'] = len(data_dict['ID_classes'])
        data_dict['templates'] = cifar_templates

        print('Test on %s level %d' %(args.corruption, args.level))
        teset_raw_10 = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' %(args.corruption))
        teset_raw_10 = teset_raw_10[(args.level-1)*tesize: args.level*tesize]
        weak_ood_dataset = CIFAR10(root=args.dataroot,
                        train=False, download=True, transform=te_transforms)
        weak_ood_dataset.data = teset_raw_10

    elif args.dataset == 'cifar100OOD':
        ID_dataset = 'cifar100'
        data_dict['ID_class_descriptions'] = json.load(open(f'{args.dataroot}/prompt_templates/{ID_dataset}_prompts_full.json'))
        data_dict['ID_classes'] = list(data_dict['ID_class_descriptions'].keys())
        data_dict['N_classes'] = len(data_dict['ID_classes'])
        data_dict['templates'] = cifar_templates

        print('Test on %s level %d' %(args.corruption, args.level))
        teset_raw_100 = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' %(args.corruption))
        teset_raw_100 = teset_raw_100[(args.level-1)*tesize: args.level*tesize]
        weak_ood_dataset = CIFAR100(root=args.dataroot,
                        train=False, download=True, transform=te_transforms)
        weak_ood_dataset.data = teset_raw_100

    elif args.dataset == 'ImagenetROOD':

        ID_dataset = 'imagenetr'
        imagenet_descriptions = json.load(open(f'{args.dataroot}/prompt_templates/imagenetr_prompts_full.json'))

        testset = ImageNetR(root= args.dataroot)
        print(testset.classnames)
        data_dict['ID_classes'] = testset.classnames
        data_dict['N_classes'] = len(data_dict['ID_classes'])
        data_dict['templates'] = imagenet_templates
        data_dict['ID_class_descriptions'] = {}
        for classname in data_dict['ID_classes']:
            data_dict['ID_class_descriptions'][classname] = imagenet_descriptions[classname]

        weak_ood_dataset = ImageNetR(root= args.dataroot, transform=te_transforms, train=True, tesize=tesize)

    elif args.dataset == "VisdaOOD":
        ID_dataset = 'visda'
        data_dict['ID_class_descriptions'] = json.load(open(f'{args.dataroot}/prompt_templates/{ID_dataset}_prompts_full.json'))
        data_dict['ID_classes'] = list(data_dict['ID_class_descriptions'].keys())
        data_dict['N_classes'] = len(data_dict['ID_classes'])
        data_dict['templates'] = cifar_templates

        weak_ood_dataset = VISDA(root= f'{args.dataroot}/visda-2017', label_files=f'{args.dataroot}/visda-2017/validation_list.txt' , transform=te_transforms, tesize=tesize)

    elif args.dataset == 'ImagenetCOOD':
        ID_dataset = 'imagenetc'


        data_dict['ID_class_descriptions'] = json.load(open(f'{args.dataroot}/prompt_templates/imagenet_prompts_full.json'))
        data_dict['ID_classes'] = list(data_dict['ID_class_descriptions'].keys())
        data_dict['N_classes'] = len(data_dict['ID_classes'])
        data_dict['templates'] = imagenet_templates

        corruption_dir_path = os.path.join(args.dataroot, 'ImageNet-C/all', args.corruption,  str(args.level))
        weak_ood_dataset = CustomImageFolder(corruption_dir_path, te_transforms, tesize=tesize)

    return data_dict, weak_ood_dataset

def get_strong_ood_data(args, te_transforms, tesize=10000):

    if args.strong_OOD == 'MNIST':
        te_rize = transforms.Compose([transforms.Resize(size=(32, 32)), transforms.Grayscale(3), te_transforms ])
        strong_ood_dataset = MNIST_openset(root=args.dataroot,
                    train=True, download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)
    
    elif args.strong_OOD == 'noise':
        strong_ood_dataset = noise_dataset(te_transforms, args.strong_ratio)


    elif args.strong_OOD =='cifar10':
        teset_raw_10 = np.load(args.dataroot + '/CIFAR-10-C/snow.npy')
        teset_raw_10 = teset_raw_10[(args.level-1)*tesize: args.level*tesize]
        strong_ood_dataset = CIFAR10_openset(root=args.dataroot,
                        train=True, download=True, transform=te_transforms, tesize=tesize, ratio=args.strong_ratio)
        strong_ood_dataset.data = teset_raw_10[:int(tesize*args.strong_ratio)]

    elif args.strong_OOD =='cifar100':
        teset_raw_100 = np.load(args.dataroot + '/CIFAR-100-C/snow.npy')
        teset_raw_100 = teset_raw_100[(args.level-1)*tesize: args.level*tesize]
        strong_ood_dataset = CIFAR100_openset(root=args.dataroot,
                        train=True, download=True, transform=te_transforms, tesize=tesize, ratio=args.strong_ratio)
        strong_ood_dataset.data = teset_raw_100[:int(tesize*args.strong_ratio)]

    elif args.strong_OOD =='SVHN': 
        te_rize = transforms.Compose([te_transforms ])
        strong_ood_dataset = SVHN_openset(root=args.dataroot,
                    split='train', download=True, transform=te_rize, tesize=tesize, ratio=args.strong_ratio)
        
    elif args.strong_OOD =='Tiny':

        transform_test = transforms.Compose([te_transforms ])
        strong_ood_dataset = TinyImageNet_OOD_nonoverlap(args.dataroot +'/tiny-imagenet-200', transform=transform_test, train=True)

    return strong_ood_dataset


def prepare_ood_test_data(args, te_transforms):

    data_dict, weak_ood_dataset = get_weak_ood_data(args, te_transforms, tesize=args.tesize)
    strong_ood_dataset = get_strong_ood_data(args, te_transforms, tesize=args.tesize)
    id_ood_dataset = torch.utils.data.ConcatDataset([weak_ood_dataset, strong_ood_dataset])
    print(f'weak ood data: {len(weak_ood_dataset)};  strong ood data: {len(strong_ood_dataset)}')
    
    ID_OOD_loader = torch.utils.data.DataLoader(id_ood_dataset, batch_size=args.batch_size, shuffle=True)

    return data_dict, id_ood_dataset, ID_OOD_loader


    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    data_dict = {}
    if args.dataset == 'cifar10OOD':
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

        ID_dataset = 'imagenetr'
        imagenet_descriptions = json.load(open(f'/home/manogna/TTA/PromptAlign/data/ood/prompt_templates/imagenetr_prompts_full.json'))

        testset = ImageNetR(root= args.dataroot)
        print(testset.classnames)
        data_dict['ID_classes'] = testset.classnames
        data_dict['N_classes'] = len(data_dict['ID_classes'])
        data_dict['templates'] = imagenet_templates
        data_dict['ID_class_descriptions'] = {}
        for classname in data_dict['ID_classes']:
            data_dict['ID_class_descriptions'][classname] = imagenet_descriptions[classname]

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

    elif args.dataset == 'ImagenetCOOD':
        ID_dataset = 'imagenetc'
        tesize = 10000

        corruption_dir_path = os.path.join(args.dataroot, 'ImageNet-C/all', args.corruption,  str(args.level))
        testset = CustomImageFolder(corruption_dir_path, te_transforms, tesize=tesize)

        data_dict['ID_class_descriptions'] = json.load(open(f'/home/manogna/TTA/PromptAlign/data/ood/prompt_templates/imagenet_prompts_full.json'))
        data_dict['ID_classes'] = list(data_dict['ID_class_descriptions'].keys())
        data_dict['N_classes'] = len(data_dict['ID_classes'])
        data_dict['templates'] = imagenet_templates

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
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views
