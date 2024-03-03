import argparse
import random
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import copy

from clip import clip
from clip import tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from models.maple import CustomCLIP

from utils.data_utils import prepare_ood_test_data
from utils.clip_tta_utils import get_classifiers
from methods import zs_baselines, ft_pl_baseline, ft_pl_neg_proto_bank_v0


def load_maple_to_cpu():
    url = clip._MODELS['ViT-B/16']
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'PromptAlign',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": 2}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


def get_model(args, classnames):

    if args.model == 'maple':

        clip_model = load_maple_to_cpu()

        model = CustomCLIP(classnames, clip_model)

        checkpoint = torch.load('/home/manogna/TTA/PromptAlign/weights/maple/ori/seed1/MultiModalPromptLearner/model.pth.tar-2')
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]

        # Ignore fixed token vectors
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]

        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]

        model.load_state_dict(state_dict, strict=False)

        model.cuda()

        return model


def zeroshot_classifier_maple(model, classnames, templates, ensemble=False):
    # with torch.no_grad():
    #     zeroshot_weights = []
    #     for classname in classnames:
    #         texts = [template.format(classname) for template in templates] #format with class
    #         texts = clip.tokenize(texts).cuda() #tokenize
    #         # prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = model.prompt_learner()
    #         # class_embeddings = model.text_encoder(prompts, model.tokenized_prompts, deep_compound_prompts_text)
    #         class_embeddings = model.get_text_features()
    #         print(class_embeddings.shape)

    #         # class_embeddings = model.encode_text(texts) #embed with text encoder
    #         class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    #         if ensemble:
    #             class_embedding = class_embeddings.mean(dim=0)
    #             class_embedding /= class_embedding.norm()
    #         else:
    #             class_embedding = class_embeddings[0]
    #             class_embedding /= class_embedding.norm()
    #         zeroshot_weights.append(class_embedding)
    #     zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
    return model.get_text_features().detach()


def get_classifiers_maple(model, classes, templates, prompt_dict):
    classifiers = {}
    classifiers['txt'] = zeroshot_classifier_maple(model, classes, templates, ensemble=False)

    # ENSEMBLE CLASSIFIER
    # classifiers['ens'] = zeroshot_classifier_maple(model, classes, templates, ensemble=True)

    # # CUPL CLASSIFIER
    # prompt_dict = {k.lower().replace("_", " "): v for k, v in prompt_dict.items()}

    # classifier_cupl = []
    # k=0
    # for single_key in classes:
    #     single_class_prompts = prompt_dict[single_key.lower().replace("_", " ")]
    #     k += 1
    #     x_tokenized = torch.cat([clip.tokenize(p) for p in single_class_prompts])
    #     with torch.no_grad():
    #         text_features = model.encode_text(x_tokenized.cuda())
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    #     classifier_cupl.append(text_features.mean(0).unsqueeze(0))
    # classifier_cupl = torch.cat(classifier_cupl, dim=0)
    # classifiers['cupl'] = classifier_cupl / classifier_cupl.norm(dim=-1, keepdim=True)

    # # ENSEMBLE CUPL CLASSIFIER
    # classifier_ens_cupl = torch.cat([classifiers['ens'].unsqueeze(0), classifiers['cupl'].unsqueeze(0)], dim=0).mean(0)
    # classifiers['ens_cupl'] = classifier_ens_cupl / classifier_ens_cupl.norm(dim=-1, keepdim=True)

    return classifiers


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10OOD')
parser.add_argument('--strong_OOD', default='MNIST')
parser.add_argument('--strong_ratio', default=1, type=float)
parser.add_argument('--dataroot', default="/home/manogna/TTA/PromptAlign/data/ood", help='path to dataset')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--out_dir', default='./logs', help='folder to output log')
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--N_m', default=512, type=int, help='queue length')
parser.add_argument('--corruption', default='snow')
parser.add_argument('--model', default='maple', help='resnet50')
parser.add_argument('--model_type', default='ViT-B/16', help='resnet50')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--ood_detector', default='maxlogit', type=str)
parser.add_argument('--tta_method', default='zsclip', type=str)
parser.add_argument('--pl_thresh', default=0.6, type=float)
parser.add_argument('--classifier_type', default='txt', type=str)




tta_methods = {'zsclip': zs_baselines.tta_id_ood, 'ft_pl_baseline': ft_pl_baseline.tta_id_ood, 
    'ft_pl_neg_proto_bank_v0': ft_pl_neg_proto_bank_v0.tta_id_ood }
    
# ----------- Args and Dataloader ------------
args = parser.parse_args()

print(args)
print('\n')

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,])

data_dict, test_set, test_loader = prepare_ood_test_data(args, preprocess)

model = get_model(args, data_dict['ID_classes'])

ID_classifiers = get_classifiers_maple(model, data_dict['ID_classes'], data_dict['templates'], data_dict['ID_class_descriptions'])

result_metrics = tta_methods[args.tta_method](args, model, test_loader, ID_classifiers)

print('\n\n\n')