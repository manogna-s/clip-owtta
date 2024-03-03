import numpy as np
import torch 
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from utils.clip_tta_utils import compute_os_variance, accuracy, cal_auc_fpr, HM


def tta_id_ood(args, model, ID_OOD_loader, ID_classifiers):

    classifier = ID_classifiers[args.classifier_type]
    tta_method = f'zsclip_{args.classifier_type}'   
    log_dir_path = os.path.join(args.out_dir, args.model, args.dataset, args.strong_OOD, tta_method)
    os.makedirs(log_dir_path, exist_ok=True)
    os.makedirs(f'{log_dir_path}/result_metrics', exist_ok=True)
    os.makedirs(f'{log_dir_path}/n_samples', exist_ok=True)
    os.makedirs(f'{log_dir_path}/ood_scores', exist_ok=True)

    n_samples= {}
    n_samples['ALL'] = 0
    n_samples['ID'] = 0
    n_samples['ID_det'] = 0
    n_samples['OOD_det'] = 0
    n_samples['ID_total'] = 0
    n_samples['OOD_total'] = 0

    ood_thresh = 'otsu'
    ood_detect = args.ood_detector
    metrics_exp = {'Method':tta_method , 'OOD Detector':ood_detect, 'AUC':0, 'FPR95':0, 'ACC_ALL':0, 'ACC_ID':0, 'ACC_OOD':0, 'ACC_HM':0}
    gt_indicators = {'ID': [], 'OOD': [], 'gt_idx': []}

    top1, top5, n = 0, 0, 0
    ood_scores = []
    scores_q = []
    queue_length = 512
    for i, (image, gt) in enumerate(ID_OOD_loader):
        if isinstance(image,list):
            image = image[0].cuda()
        else:
            image = image.cuda()
        image, gt = image.cuda(), gt.cuda()
        gt_indicators['ID'].append((gt<1000).item())
        gt_indicators['OOD'].append((gt>=1000).item())
        gt_indicators['gt_idx'].append(gt.item())

        # Base eval

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                imf_norm = model.encode_image(image)
                imf_norm = imf_norm/imf_norm.norm(dim=-1, keepdim=True)
                logits_txt =  (imf_norm @ classifier.T)
                scores_txt = (logits_txt * 100).softmax(1)
                msp_txt, _ = scores_txt.max(1)
                maxlogit_txt, _ = logits_txt.max(1)
                conf_txt, pred_txt = torch.max(scores_txt, dim=1)

                best_thresh = ood_thresh
                ood_score= {'msp': msp_txt, 'maxlogit': maxlogit_txt}
                if ood_thresh == 'otsu':
                    threshold_range = np.arange(0,1,0.01)
                    ood_scores.extend(ood_score[ood_detect].tolist())
                    scores_q.extend(ood_score[ood_detect].tolist())
                    scores_q = scores_q[-queue_length:]
                    criterias = [compute_os_variance(np.array(scores_q), th) for th in threshold_range]
                    best_thresh = threshold_range[np.argmin(criterias)]

        ID_curr, OOD_curr = gt<1000, gt>=1000
        ID_pred, OOD_pred = ood_score[ood_detect] >= best_thresh, ood_score[ood_detect] < best_thresh

        # metrics
        n_samples['ID_det'] += torch.sum(ID_pred[ID_curr]).item()
        n_samples['ID_total'] += torch.sum(ID_curr).item()
        n_samples['OOD_det'] += torch.sum(OOD_pred[OOD_curr]).item()
        n_samples['OOD_total'] += torch.sum(OOD_curr).item()

        if ID_pred[0].item():
            n_samples['ID'] += torch.sum(gt==pred_txt).item()
        n_samples['ALL'] += torch.sum(gt[gt<1000]==pred_txt[gt<1000]).item()

        if (i+1) %1000 == 0:
            metrics_exp['ACC_ALL'] = n_samples['ALL']/n_samples['ID_total']
            metrics_exp['ACC_ID'] = n_samples['ID']/n_samples['ID_total']
            print(f'\nStep {i}: Top1: {top1/n:.4f}; Top5: {top5/n:.4f}')
            print(metrics_exp)


        # measure accuracy
        acc1, acc5 = accuracy(logits_txt[gt<1000], gt[gt<1000], topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += gt[gt<1000].shape[0]

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    metrics_exp['ACC_OOD'] = n_samples['OOD_det']/n_samples['OOD_total']

    metrics_exp['ACC_ALL'] = n_samples['ALL']/n_samples['ID_total']
    metrics_exp['ACC_ID'] = n_samples['ID']/n_samples['ID_total']
    metrics_exp['ACC_OOD'] = n_samples['OOD_det']/n_samples['OOD_total']

    ood_scores = np.array(ood_scores)
    metrics_exp['AUC'], metrics_exp['FPR95'] = cal_auc_fpr(ood_scores[gt_indicators['ID']], ood_scores[gt_indicators['OOD']])
    metrics_exp['ACC_HM'] = HM(metrics_exp['ACC_ID'], metrics_exp['ACC_OOD'])

    print(f'\n\n')
    print(args.dataset, args.strong_OOD, tta_method)
    print(f'Metrics: {metrics_exp}\n')

    print(f"Online evaluation: Top-1 accuracy: {top1:.4f}; Top-5 accuracy: {top5:.4f}")

    name = f'{ood_detect}'

    json.dump(n_samples, open(f'{log_dir_path}/n_samples/{name}.json', 'w'))

    df_metrics = pd.DataFrame([metrics_exp])
    df_metrics.to_csv(f'{log_dir_path}/result_metrics/{name}.csv', index=False)  

    np.save(f'{log_dir_path}/ood_scores/{name}.npy', ood_scores)

    plt.hist(ood_scores[gt_indicators['ID']], bins=threshold_range, label='ID', alpha=0.5)
    plt.hist(ood_scores[gt_indicators['OOD']], bins=threshold_range, label='OOD', alpha=0.5)
    plt.savefig(f'{log_dir_path}/ood_scores/{name}.jpg')

    np.save(f'{log_dir_path}/ood_scores/{name}.npy', ood_scores)

    return metrics_exp