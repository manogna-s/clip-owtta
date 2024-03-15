
import numpy as np
import torch
import torchvision
import torch.nn as nn

from clip import clip
from tqdm.notebook import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
import json
from sklearn import metrics
from sklearn.metrics import accuracy_score as Acc
from sklearn.metrics import roc_auc_score as Auc
from sklearn.metrics import roc_curve as Roc
from scipy import interpolate
from scipy.special import logsumexp

import os
from torchvision.datasets import CIFAR10, CIFAR100

warnings.filterwarnings("ignore")


def zeroshot_classifier(model, classnames, templates, ensemble=False):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            if ensemble:
                texts = [template.format(classname) for template in templates] #format with class
                texts = clip.tokenize(texts).cuda() #tokenize
                class_embeddings = model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
            else:
                template = 'a photo of a {}.'
                texts = [template.format(classname)]
                texts = clip.tokenize(texts).cuda()
                class_embeddings = model.encode_text(texts) #embed with text encoder
                class_embedding = class_embeddings
                class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
        zeroshot_weights = zeroshot_weights.squeeze()
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def eval_zs_clip(model, loader, classifier, plot=False):
  features = []
  labels = []
  with torch.no_grad():
      top1, top5, n = 0., 0., 0.
      for i, (images, target) in enumerate(loader):
          images = images.cuda()
          target = target.cuda()

          # predict
          image_features = model.encode_image(images)
          image_features /= image_features.norm(dim=-1, keepdim=True)
          features.append(image_features)
          labels.append(target)
          logits = 100. * image_features @ classifier.T

          # measure accuracy
          acc1, acc5 = accuracy(logits, target, topk=(1, 5))
          top1 += acc1
          top5 += acc5
          n += images.size(0)

  top1 = (top1 / n) * 100
  top5 = (top5 / n) * 100

  features_cat = torch.vstack(features)
  labels_cat = torch.cat(labels)
  tsne = TSNE(n_components=2)
  tsne_f = tsne.fit_transform(features_cat.cpu().numpy())
  if plot:
    plt.scatter(tsne_f[:,0], tsne_f[:,1],c=labels_cat.cpu().numpy(), s=1)
    plt.colorbar()
  return features_cat, labels_cat, top1, top5


def extract_features(loader, model, plot=False):
  features = []
  labels = []
  with torch.no_grad():
      for i, (images, target) in enumerate(loader):
          images = images.cuda()
          target = target.cuda()

          # predict
          image_features = model.encode_image(images)
          image_features /= image_features.norm(dim=-1, keepdim=True)
          features.append(image_features)
          labels.append(target)

  features_cat = torch.vstack(features)
  labels_cat = torch.cat(labels)
  tsne = TSNE(n_components=2)
  tsne_f = tsne.fit_transform(features_cat.cpu().numpy())
  if plot:
    plt.scatter(tsne_f[:,0], tsne_f[:,1],c=labels_cat.cpu().numpy(), s=1)
    plt.colorbar()
  return {'feat': features_cat, 'labels': labels_cat}


def eval_clip(features, labels, classifier):
  logits = (features @ classifier.T)
  maxlogit, pred = logits.max(1)
  msp, _ = (logits * 100).softmax(1).max(1)
  n = logits.shape[0]
  acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
  top1 = (acc1 / n) * 100
  top5 = (acc5 / n) * 100
  # print(f"accuracy with text proxy: {top1:.2f}")
  summary = {'logits': logits, 'pred': pred, 'maxlogit': maxlogit, 'msp':msp, 'acc1':top1, 'acc5': top5}
  return summary


# TEXT CLASSIFIER
def get_classifiers(args, model, classes, templates, prompt_dict):
    classifiers = {}
    if args.model == 'clip':
        classifiers['txt'] = zeroshot_classifier(model, classes, templates, ensemble=False)

        # ENSEMBLE CLASSIFIER
        # classifiers['ens'] = zeroshot_classifier(model, classes, templates, ensemble=True)

        # CUPL CLASSIFIER
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

    elif args.model == 'coop' or args.model =='maple':
        with torch.no_grad():
            classifiers['txt'] = model.get_text_features().detach()

    return classifiers


def get_ln_params(model):
    names, params = [], []
    for nm, p in model.named_parameters():
        if ('visual' in nm or 'image_encoder' in nm) and 'ln' in nm:
            names.append(nm)
            params.append(p)
    # print(names)
    return params

def HM(a,b):
    return 2*a*b/(a+b)
    
def cal_auc_fpr(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))
    auroc = metrics.roc_auc_score(ind_indicator, conf)
    fpr,tpr,thresh = Roc(ind_indicator, conf, pos_label=1)
    fpr = float(interpolate.interp1d(tpr, fpr)(0.95))
    return auroc, fpr

def compute_os_variance(os, th):
    """
    Calculate the area of a rectangle.

    Parameters:
        os : OOD score queue.
        th : Given threshold to separate weak and strong OOD samples.

    Returns:
        float: Weighted variance at the given threshold th.
    """
    
    thresholded_os = np.zeros(os.shape)
    thresholded_os[os >= th] = 1

    # compute weights
    nb_pixels = os.size
    nb_pixels1 = np.count_nonzero(thresholded_os)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = os[thresholded_os == 1]
    val_pixels0 = os[thresholded_os == 0]

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0 * var0 + weight1 * var1