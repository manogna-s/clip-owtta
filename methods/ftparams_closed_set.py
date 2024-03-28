import numpy as np
import torch 
import torch.nn as nn
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from utils.clip_tta_utils import compute_os_variance, accuracy, cal_auc_fpr, HM, get_ln_params
from utils.registry import METHODS_REGISTRY


def normal_dist(x, mean, sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density



@METHODS_REGISTRY.register()
def FtParamsClosedSet(args, model, ID_OOD_loader, ID_classifiers):

    tta_method = f'{args.tta_method}_{args.classifier_type}' 
    ood_thresh = 'otsu'
    ood_detect = args.ood_detector
    
    learnable_param = args.param_group
    print(learnable_param)

    name = f'{args.model}'

    if learnable_param =='ln':
        ln_params = get_ln_params(model)
        params = ln_params
        name += '_lnparams'
    if learnable_param == 'prompts':
        params = model.prompt_learner.parameters()
        name += '_prompts'
    if learnable_param == 'fullv':
        params = model.image_encoder.parameters()
        name += '_fullv'
    if learnable_param == 'fullt':
        params = model.text_encoder.parameters()
        name += '_fullt'
    if learnable_param == 'full':
        params = model.parameters()
        name += '_full'
    name+=f'_{str(args.tta_lr)}_{args.tesize}_{args.opt}'
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.tta_lr)
    if args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args.tta_lr)

    print(learnable_param, optimizer)


    log_dir_path = os.path.join(args.out_dir, args.model, args.dataset, args.strong_OOD, tta_method)
    os.makedirs(log_dir_path, exist_ok=True)
    os.makedirs(f'{log_dir_path}/result_metrics', exist_ok=True)
    os.makedirs(f'{log_dir_path}/n_samples', exist_ok=True)
    os.makedirs(f'{log_dir_path}/ood_scores', exist_ok=True)
    os.makedirs(f'{log_dir_path}/out_logs', exist_ok=True)
    log_file = open(f'{log_dir_path}/out_logs/{name}.txt', 'w')
    log_file.write(str(learnable_param))
    log_file.write(str(optimizer))



    n_samples= {}
    n_samples['ALL'] = 0 
    n_samples['ID'] = 0 
    n_samples['ID_det'] = 0
    n_samples['OOD_det'] = 0
    n_samples['ID_total'] = 0
    n_samples['OOD_total'] = 0

    metrics_exp = {'Method':tta_method, 'samples': args.tesize, 'lr': args.tta_lr, 'params':args.param_group, 'ACC_ID':0}
    ood_data = {'ID': [], 'OOD': [], 'gt_idx': [], 'ood_scores': []}

    viz_dict = {'features':[], 'pred':[], 'gt':[]}

    top1, top5, n = 0, 0, 0

    model.set_prompt_inits()

    for i, (image, gt) in enumerate(ID_OOD_loader):
        if isinstance(image,list):
            image = image[0].cuda()
        else:
            image = image.cuda()
        image, gt = image.cuda(), gt.cuda()
        if gt>=1000: continue
        
        ood_data['ID'].append((gt<1000).item())
        ood_data['OOD'].append((gt>=1000).item())
        ood_data['gt_idx'].append(gt.item())

        # #TPT
        # with torch.no_grad():
        #     model.reset()

        # TTA
        image_features_raw = model.encode_image(image)
        image_features = image_features_raw/image_features_raw.norm(dim=-1, keepdim=True)

        tta_classifier = model.get_text_features()
        logits = image_features @ tta_classifier.T
        maxlogit_tta, pred_tta = logits.max(1)
        msp, _ = (logits * 100).softmax(1).max(1)

        if msp > 0.7:
            loss = nn.CrossEntropyLoss()(logits, pred_tta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # metrics

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                imf_norm = model.encode_image(image)
                imf_norm = imf_norm/imf_norm.norm(dim=-1, keepdim=True)
                tta_classifier = model.get_text_features()
                logits_txt = image_features @ tta_classifier.T
                scores_txt = (logits_txt * 100).softmax(1)
                _, pred_tta = torch.max(scores_txt, dim=1)

                viz_dict['features'].append(imf_norm)
                viz_dict['gt'].append(gt)
                viz_dict['pred'].append(pred_tta)


        n_samples['ID'] += torch.sum(gt==pred_tta).item()
        n_samples['ID_total'] += torch.sum(gt<1000).item()

        if (i+1) %1000 == 0:
            # metrics_exp['ACC_ALL'] = n_samples['ALL']/n_samples['ID_total']
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

    metrics_exp['ACC_ID'] = n_samples['ID']/n_samples['ID_total']

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
    torch.save(viz_dict, f'{log_dir_path}/ood_scores/viz_{name}.pth')

    return metrics_exp