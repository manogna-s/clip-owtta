import numpy as np
import torch 
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from utils.clip_tta_utils import compute_os_variance, accuracy, cal_auc_fpr, HM
from utils.registry import METHODS_CTTA_REGISTRY

@METHODS_CTTA_REGISTRY.register()
def ZSEval(args, model, ctta_loaders, ID_classifiers):

    classifier = ID_classifiers[args.classifier_type]
    tta_method = f'{args.tta_method}_{args.classifier_type}'   
    ood_detect = args.ood_detector

    log_dir_path = os.path.join(args.out_dir, args.model, args.dataset, args.strong_OOD, tta_method)
    os.makedirs(log_dir_path, exist_ok=True)
    os.makedirs(f'{log_dir_path}/result_metrics', exist_ok=True)
    os.makedirs(f'{log_dir_path}/n_samples', exist_ok=True)
    os.makedirs(f'{log_dir_path}/ood_scores', exist_ok=True)
    os.makedirs(f'{log_dir_path}/out_logs', exist_ok=True)

    metrics_ctta = []
    for corruption, ID_OOD_loader in ctta_loaders.items():
        name = f'{corruption}'
        log_file = open(f'{log_dir_path}/out_logs/{name}.txt', 'w')


        n_samples= {}
        n_samples['ALL'] = 0
        n_samples['ID'] = 0
        n_samples['ID_det'] = 0
        n_samples['OOD_det'] = 0
        n_samples['ID_total'] = 0
        n_samples['OOD_total'] = 0

        ood_thresh = 'otsu'
        ood_detect = args.ood_detector
        metrics_exp = {'Method':tta_method , 'corruption': corruption,  'AUC':0, 'FPR95':0, 'ACC_ALL':0, 'ACC_ID':0, 'ACC_OOD':0, 'ACC_HM':0, 'kp':args.k_p, 'kn':args.k_n, 'alpha':args.alpha, 'loss_pl':args.loss_pl, 'loss_simclr':args.loss_simclr, 'OOD Detector':ood_detect}
        ood_data = {'ID': [], 'OOD': [], 'gt_idx': [], 'ood_scores': []}

        top1, top5, n = 0, 0, 0
        scores_q = []
        queue_length = args.N_m
        for i, (image, gt) in enumerate(ID_OOD_loader):
            if isinstance(image,list):
                image = image[0].cuda()
            else:
                image = image.cuda()
            image, gt = image.cuda(), gt.cuda()
            ood_data['ID'].append((gt<1000).item())
            ood_data['OOD'].append((gt>=1000).item())
            ood_data['gt_idx'].append(gt.item())

            # Base eval

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    imf_norm = model.encode_image(image)
                    imf_norm = imf_norm/imf_norm.norm(dim=-1, keepdim=True)
                    logits_txt =  (imf_norm @ classifier.T)
                    scores_txt = (logits_txt * 100).softmax(1)
                    msp_txt, _ = scores_txt.max(1)
                    maxlogit_txt, _ = logits_txt.max(1)
                    energy = torch.logsumexp(logits_txt * 100, 1)/100

                    conf_txt, pred_txt = torch.max(scores_txt, dim=1)

                    best_thresh = ood_thresh
                    ood_score= {'msp': msp_txt, 'maxlogit': maxlogit_txt, 'energy': energy}

                    if ood_thresh == 'otsu':
                        threshold_range = np.arange(0,1,0.01)
                        ood_data['ood_scores'].extend(ood_score[ood_detect].tolist())
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

        metrics_exp['ACC_OOD'] = n_samples['OOD_det']/n_samples['OOD_total']

        metrics_exp['ACC_ALL'] = n_samples['ALL']/n_samples['ID_total']
        metrics_exp['ACC_ID'] = n_samples['ID']/n_samples['ID_total']
        metrics_exp['ACC_OOD'] = n_samples['OOD_det']/n_samples['OOD_total']

        # ood_scores = np.array(ood_scores)
        ood_data['ood_scores'] = np.array(ood_data['ood_scores'])
        metrics_exp['AUC'], metrics_exp['FPR95'] = cal_auc_fpr(ood_data['ood_scores'][ood_data['ID']], ood_data['ood_scores'][ood_data['OOD']])
        metrics_exp['ACC_HM'] = HM(metrics_exp['ACC_ID'], metrics_exp['ACC_OOD'])

        print(f'\n\n')
        print(args.dataset, args.strong_OOD, tta_method, ood_detect, corruption)
        status_log = f"\n\nFinal metrics: Top-1 accuracy: {top1:.4f}; Top-5 accuracy: {top5:.4f}\n{metrics_exp}"
        print(status_log)
        print("---------------------------------------------------------------------------")
        log_file.write(status_log)


        json.dump(n_samples, open(f'{log_dir_path}/n_samples/{name}.json', 'w'))

        df_metrics = pd.DataFrame([metrics_exp])
        df_metrics.to_csv(f'{log_dir_path}/result_metrics/{name}.csv', index=False)  

        torch.save(ood_data, f'{log_dir_path}/ood_scores/{name}.pth')

        plt.hist(ood_data['ood_scores'][ood_data['ID']], bins=threshold_range, label='ID', alpha=0.5)
        plt.hist(ood_data['ood_scores'][ood_data['OOD']], bins=threshold_range, label='OOD', alpha=0.5)
        plt.title(f'{tta_method}_{args.model}_{corruption}_{args.strong_OOD}')
        plt.savefig(f'{log_dir_path}/ood_scores/{name}.jpg')
        plt.clf()

        metrics_ctta.append(metrics_exp)
    metrics_ctta = pd.DataFrame.from_dict(metrics_ctta)
    metrics_ctta.to_csv(f'{log_dir_path}/result_metrics/ctta.csv', index=False)

    return metrics_ctta