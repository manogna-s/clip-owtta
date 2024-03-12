import argparse
import random
import torch
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import numpy as np
from clip import clip
from utils.data_utils import prepare_ood_test_data, AugMixAugmenter
from utils.clip_tta_utils import get_classifiers
from methods import zs_baselines, tpt, promptalign_aug, rosita, rosita_loss, rosita_knn
tta_methods = {'zsclip': zs_baselines.tta_id_ood, 
    'tpt': tpt.tta_id_ood, 'promptalign_aug': promptalign_aug.tta_id_ood, 
    'rosita': rosita.tta_id_ood, 'rosita_loss': rosita_loss.tta_id_ood, 'rosita_knn': rosita_knn.tta_id_ood}

def load_clip_to_cpu():
    url = clip._MODELS['ViT-B/16']
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def get_model_and_transforms(args):
    if args.model == 'clip':
        model = load_clip_to_cpu()
        model.cuda()
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])

        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,])
        
        if args.tta_method in ['tpt', 'promptalign']:
            base_transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224)])
            preprocess = transforms.Compose([
                        transforms.ToTensor(),
                        normalize])
            preprocess = AugMixAugmenter(base_transform, preprocess, n_views=args.n_views-1, 
                                                    augmix=False)
        
    return model, preprocess



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10OOD')
parser.add_argument('--strong_OOD', default='MNIST')
parser.add_argument('--strong_ratio', default=1, type=float)
parser.add_argument('--dataroot', default="/home/manogna/TTA/PromptAlign/data/ood", help='path to dataset')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--n_views', default=64, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--out_dir', default='./logs', help='folder to output log')
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--N_m', default=512, type=int, help='queue length')
parser.add_argument('--corruption', default='snow')
parser.add_argument('--model', default='clip', help='resnet50')
parser.add_argument('--model_type', default='ViT-B/16', help='resnet50')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--ood_detector', default='maxlogit', type=str)
parser.add_argument('--tta_method', default='zsclip', type=str)
parser.add_argument('--pl_thresh', default=0.7, type=float)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--classifier_type', default='txt', type=str)
parser.add_argument('--k_p', default=3, type=int)
parser.add_argument('--k_n', default=10, type=int)
parser.add_argument('--loss_pl', default=1, type=int)
parser.add_argument('--loss_simclr', default=1, type=int)


    
# ----------- Args and Dataloader ------------

if __name__ == "__main__":
    args = parser.parse_args()


    print(args)
    print('\n')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    model, preprocess = get_model_and_transforms(args)

    data_dict, test_set, test_loader = prepare_ood_test_data(args, preprocess)

    ID_classifiers = get_classifiers(model, data_dict['ID_classes'], data_dict['templates'], data_dict['ID_class_descriptions'])


    result_metrics = tta_methods[args.tta_method](args, model, test_loader, ID_classifiers)

    print('\n\n\n')