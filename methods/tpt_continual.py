import numpy as np
import torch 
import torch.nn as nn
import os
import json
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
from utils.clip_tta_utils import compute_os_variance, accuracy, cal_auc_fpr, HM, get_ln_params
from torch.nn import functional as F
from utils.registry import METHODS_REGISTRY


TPT_THRESHOLD = 0.1
ALIGN_THRESHOLD = 0.1
DISTR_LOSS_W = 100.0
visual_vars = torch.load('weights/features/ImgNetpre_vis_means.pt')
visual_means = torch.load('weights/features/ImgNetpre_vis_vars.pt')
ALIGN_LAYER_FROM = 0
ALIGN_LAYER_TO = 3

def select_confident_samples(logits, topTPT, topAlign):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idxTPT = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topTPT)]
    idxAlign = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topAlign)]
    return logits[idxTPT], idxAlign


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def tpt_test_time_tuning(model, inputs, optimizer, scaler):

    selected_idx = None
    DISTR_LOSS_W = 100.0
    for j in range(1):
        with torch.cuda.amp.autocast():
            output = model(inputs) 

            output, selected_idx = select_confident_samples(output, TPT_THRESHOLD, ALIGN_THRESHOLD)

            loss = avg_entropy(output)
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()

    return model


@METHODS_REGISTRY.register()
def TPTContinual(args, model, ID_OOD_loader, ID_classifiers):

    tta_method = f'{args.tta_method}_{args.classifier_type}' 
    ood_thresh = 'otsu'
    ood_detect = args.ood_detector
    name = f'{ood_detect}'
    
    log_dir_path = os.path.join(args.out_dir, args.model, args.dataset, args.strong_OOD, tta_method)
    os.makedirs(log_dir_path, exist_ok=True)
    os.makedirs(f'{log_dir_path}/result_metrics', exist_ok=True)
    os.makedirs(f'{log_dir_path}/n_samples', exist_ok=True)
    os.makedirs(f'{log_dir_path}/ood_scores', exist_ok=True)
    os.makedirs(f'{log_dir_path}/out_logs', exist_ok=True)
    log_file = open(f'{log_dir_path}/out_logs/{name}.txt', 'w')
    


    n_samples= {}
    n_samples['ALL'] = 0 
    n_samples['ID'] = 0 
    n_samples['ID_det'] = 0
    n_samples['OOD_det'] = 0
    n_samples['ID_total'] = 0
    n_samples['OOD_total'] = 0


    metrics_exp = {'Method':tta_method , 'AUC':0, 'FPR95':0, 'ACC_ALL':0, 'ACC_ID':0, 'ACC_OOD':0, 'ACC_HM':0, 'kp':args.k_p, 'kn':args.k_n, 'alpha':args.alpha, 'loss_pl':args.loss_pl, 'loss_simclr':args.loss_simclr, 'OOD Detector':ood_detect}
    ood_data = {'ID': [], 'OOD': [], 'gt_idx': [], 'ood_scores': []}

    top1, top5, n = 0, 0, 0
    ood_scores = []
    scores_q = []
    queue_length = args.N_m

    model.set_prompt_inits() 
    for nm, param in model.named_parameters():
        if "prompt_learner" not in nm:
            param.requires_grad_(False)
            
    trainable_param = model.prompt_learner.parameters()
    # optimizer = torch.optim.AdamW(trainable_param, lr=4e-2)
    optimizer = torch.optim.SGD(trainable_param, lr=0.00001) #, momentum=0.9)
    log_file.write(f'{optimizer}')
    print(optimizer)
    optim_state = deepcopy(optimizer.state_dict())
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    model.eval()

    for i, (images, gt) in enumerate(ID_OOD_loader):
        images = images[:-1]
        if isinstance(images,list):
            for k in range(len(images)):
                images[k] = images[k].cuda()
            image = images[0]
        else:
            image = image.cuda()
        images = torch.cat(images, dim=0)
        image, gt = image.cuda(), gt.cuda()
        ood_data['ID'].append((gt<1000).item())
        ood_data['OOD'].append((gt>=1000).item())
        ood_data['gt_idx'].append(gt.item())
        
        #TPT for continuous update: No reset
        with torch.no_grad():
            # model.reset()
        
            # TTA
            image_features = model.encode_image(image)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)

            tta_classifier = model.get_text_features()
            logits = image_features @ tta_classifier.T
        maxlogit_tta, pred_tta = logits.max(1)
        msp, _ = (logits * 100).softmax(1).max(1)
        energy = torch.logsumexp(logits * 100, 1)/100
        
        best_thresh = ood_thresh
        ood_score= {'msp': msp, 'maxlogit': maxlogit_tta, 'energy': energy}
        if ood_thresh == 'otsu':
            threshold_range = np.arange(0,1,0.01)
            ood_data['ood_scores'].extend(ood_score[ood_detect].tolist())
            scores_q.extend(ood_score[ood_detect].tolist())
            scores_q = scores_q[-queue_length:]
            criterias = [compute_os_variance(np.array(scores_q), th) for th in threshold_range]
            best_thresh = threshold_range[np.argmin(criterias)]

        ID_curr, OOD_curr = gt<1000, gt>=1000
        ID_pred, OOD_pred = ood_score[ood_detect] >= best_thresh, ood_score[ood_detect] < best_thresh

        if ID_pred[0].item():    
            # optimizer.load_state_dict(optim_state) #for continuous update
            model = tpt_test_time_tuning(model, images, optimizer, scaler)


        # metrics
        n_samples['ID_det'] += torch.sum(ID_pred[ID_curr]).item()
        n_samples['ID_total'] += torch.sum(ID_curr).item()
        n_samples['OOD_det'] += torch.sum(OOD_pred[OOD_curr]).item()
        n_samples['OOD_total'] += torch.sum(OOD_curr).item()

        with torch.no_grad():
            with torch.cuda.amp.autocast():                
                imf_norm = model.encode_image(image)
                imf_norm = imf_norm/imf_norm.norm(dim=-1, keepdim=True)
                tta_classifier = model.get_text_features().detach()
                logits_txt =  (imf_norm @ tta_classifier.T)
                scores_txt = (logits_txt * 100).softmax(1)
                _, pred_tta = torch.max(scores_txt, dim=1)

        if ID_pred[0].item():
            n_samples['ID'] += torch.sum(gt==pred_tta).item()

        n_samples['ALL'] += torch.sum(gt[gt<1000]==pred_tta[gt<1000]).item()

        if (i+1) %1000 == 0:
            metrics_exp['ACC_ALL'] = n_samples['ALL']/n_samples['ID_total']
            metrics_exp['ACC_ID'] = n_samples['ID']/n_samples['ID_total']
            status_log = f'\nStep {i}: Top1: {top1/n:.4f}; Top5: {top5/n:.4f}\n{metrics_exp}'
            print(status_log)
            log_file.write(status_log)


        # measure accuracy
        acc1, acc5 = accuracy(logits_txt[gt<1000], gt[gt<1000], topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += gt[gt<1000].shape[0]

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100


    metrics_exp['ACC_ALL'] = n_samples['ALL']/n_samples['ID_total']
    metrics_exp['ACC_ID'] = n_samples['ID']/n_samples['ID_total']
    metrics_exp['ACC_OOD'] = n_samples['OOD_det']/n_samples['OOD_total']
        
    ood_data['ood_scores'] = np.array(ood_data['ood_scores'])
    metrics_exp['AUC'], metrics_exp['FPR95'] = cal_auc_fpr(ood_data['ood_scores'][ood_data['ID']], ood_data['ood_scores'][ood_data['OOD']])
    metrics_exp['ACC_HM'] = HM(metrics_exp['ACC_ID'], metrics_exp['ACC_OOD'])

    print(args.dataset, args.strong_OOD, tta_method, ood_detect)
    status_log = f"\n\nFinal metrics: Top-1 accuracy: {top1:.4f}; Top-5 accuracy: {top5:.4f}\n{metrics_exp}"
    print(status_log)
    print("---------------------------------------------------------------------------")
    log_file.write(status_log)

    # Save all metrics and scores
    json.dump(n_samples, open(f'{log_dir_path}/n_samples/{name}.json', 'w'))

    df_metrics = pd.DataFrame([metrics_exp])
    df_metrics.to_csv(f'{log_dir_path}/result_metrics/{name}.csv', index=False)  

    torch.save(ood_data, f'{log_dir_path}/ood_scores/{name}.pth')

    plt.hist(ood_data['ood_scores'][ood_data['ID']], bins=threshold_range, label='ID', alpha=0.5)
    plt.hist(ood_data['ood_scores'][ood_data['OOD']], bins=threshold_range, label='OOD', alpha=0.5)
    plt.savefig(f'{log_dir_path}/ood_scores/{name}.jpg')

    return metrics_exp


