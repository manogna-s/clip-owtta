import numpy as np
import torch 
import torch.nn as nn
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from utils.clip_tta_utils import compute_os_variance, accuracy, cal_auc_fpr, HM, get_ln_params


def normal_dist(x, mean, sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density


def compute_os_variance_stats(os, th):
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
        return np.inf, {'mu0': 0, 'mu1': 1, 'var0': 0, 'var1': 0}

    # find all pixels belonging to each class
    val_pixels1 = os[thresholded_os == 1]
    val_pixels0 = os[thresholded_os == 0]

    # compute mean of these classes
    mu0 = np.mean(val_pixels0) if len(val_pixels0) > 0 else 0
    mu1 = np.mean(val_pixels1) if len(val_pixels1) > 0 else 0

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    stats = {'mu0': mu0, 'mu1': mu1, 'var0': var0, 'var1': var1}
    return weight0 * var0 + weight1 * var1, stats



def tta_id_ood(args, model, ID_OOD_loader, ID_classifiers):

    classifier = ID_classifiers[args.classifier_type]
    tta_method = f'{args.tta_method}_{args.classifier_type}' 
    ood_thresh = 'otsu'
    ood_detect = args.ood_detector
    name = f'{ood_detect}_plthresh0{int(args.pl_thresh*10)}'

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

    metrics_exp = {'Method':tta_method , 'OOD Detector':name, 'AUC':0, 'FPR95':0, 'ACC_ALL':0, 'ACC_ID':0, 'ACC_OOD':0, 'ACC_HM':0}
    gt_indicators = {'ID': [], 'OOD': [], 'gt_idx': []}

    top1, top5, n = 0, 0, 0
    ood_scores = []
    scores_q = []
    queue_length = args.N_m

    ln_params = get_ln_params(model)
    optimizer = torch.optim.SGD(ln_params, lr=0.001)

    neg_proto = torch.zeros((1, 512)).cuda()

    proto_bank = {'ID': [], 'OOD': []}

    for i, (image, gt) in enumerate(ID_OOD_loader):
        if isinstance(image,list):
            image = image[0].cuda()
        else:
            image = image.cuda()
        image, gt = image.cuda(), gt.cuda()
        gt_indicators['ID'].append((gt<1000).item())
        gt_indicators['OOD'].append((gt>=1000).item())
        gt_indicators['gt_idx'].append(gt.item())

        # TTA
        image_features_raw = model.encode_image(image)
        image_features = image_features_raw/image_features_raw.norm(dim=-1, keepdim=True)

        logits = image_features @ classifier.T
        maxlogit_tta, pred_tta = logits.max(1)
        msp, _ = (logits * 100).softmax(1).max(1)
        energy = torch.logsumexp(logits * 100, 1)/100


        best_thresh = ood_thresh
        ood_score= {'msp': msp, 'maxlogit': maxlogit_tta, 'energy': energy}
        if ood_thresh == 'otsu':
            #auto threshold
            threshold_range = np.arange(0,1,0.01)
            ood_scores.extend(ood_score[ood_detect].tolist())
            scores_q.extend(ood_score[ood_detect].tolist())
            scores_q = scores_q[-queue_length:]
            criterias = []
            for th in threshold_range:
                inter_var, _ = compute_os_variance_stats(np.array(scores_q), th)
                # print(inter_var)
                criterias.append(inter_var)
            # criterias = [compute_os_variance(np.array(scores_q), th) for th in threshold_range]
            best_thresh = threshold_range[np.argmin(criterias)]
            _, stats = compute_os_variance_stats(np.array(scores_q), best_thresh)

        ID_curr, OOD_curr = gt<1000, gt>=1000
        ID_pred, OOD_pred = ood_score[ood_detect] >= best_thresh, ood_score[ood_detect] < best_thresh

        ID_sel = ID_pred  

        loss = 0
        if ID_pred[0].item(): # and ood_score[ood_detect].item()>stats['mu1']:
            proto_bank['ID'].append(image_features_raw.detach())
            proto_bank['ID'] = proto_bank['ID'][-queue_length:]
        if OOD_pred[0].item(): # and ood_score[ood_detect].item()<stats['mu0']:
            proto_bank['OOD'].append(image_features_raw.detach())
            proto_bank['OOD'] = proto_bank['OOD'][-queue_length:]

        if ID_sel.item():

            loss += nn.CrossEntropyLoss()(logits[ID_sel], pred_tta[ID_sel])

        if i>queue_length:
            # p_ood = normal_dist(ood_score[ood_detect].item(), stats['mu0'], np.sqrt(stats['var0']))
            # p_id = normal_dist(ood_score[ood_detect].item(), stats['mu1'], np.sqrt(stats['var1']))
            if True: #ood_score[ood_detect].item()>stats['mu1'] or ood_score[ood_detect].item()<stats['mu0']:
                if ID_pred[0].item():
                    pos_features = torch.vstack(proto_bank['ID'])
                    neg_features = torch.vstack(proto_bank['OOD'])

                elif OOD_pred[0].item(): 
                    pos_features = torch.vstack(proto_bank['OOD'])
                    neg_features = torch.vstack(proto_bank['ID'])

                pos_features = pos_features/pos_features.norm(dim=-1, keepdim=True)
                neg_features = neg_features/neg_features.norm(dim=-1, keepdim=True)

                pos_sim = image_features @ pos_features.T
                topk_pos_sim, _ = torch.topk(pos_sim, k=args.k_p, dim=-1, largest=True) 

                neg_sim = image_features @ neg_features.T
                topk_neg_sim, _ = torch.topk(neg_sim, k=args.k_n, dim=-1, largest=True) 

                pos_sim_k = topk_pos_sim.T
                neg_sim_k = topk_neg_sim.expand(args.k_p, -1)
                simclr_logits = torch.cat([pos_sim_k, neg_sim_k], dim=1)

                l_simclr = nn.CrossEntropyLoss()(simclr_logits, torch.zeros((args.k_p,), dtype=torch.long).cuda())
                loss += l_simclr * args.alpha

        if loss:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # metrics
        n_samples['ID_det'] += torch.sum(ID_pred[ID_curr]).item()
        n_samples['ID_total'] += torch.sum(ID_curr).item()
        n_samples['OOD_det'] += torch.sum(OOD_pred[OOD_curr]).item()
        n_samples['OOD_total'] += torch.sum(OOD_curr).item()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                imf_norm = model.encode_image(image)
                imf_norm = imf_norm/imf_norm.norm(dim=-1, keepdim=True)
                logits_txt =  (imf_norm @ classifier.T)
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
        
    ood_scores = np.array(ood_scores)
    metrics_exp['AUC'], metrics_exp['FPR95'] = cal_auc_fpr(ood_scores[gt_indicators['ID']], ood_scores[gt_indicators['OOD']])
    metrics_exp['ACC_HM'] = HM(metrics_exp['ACC_ID'], metrics_exp['ACC_OOD'])

    print(args.dataset, args.strong_OOD, tta_method, ood_detect)
    status_log = f"\n\nFinal metrics: Top-1 accuracy: {top1:.4f}; Top-5 accuracy: {top5:.4f}\n{metrics_exp}"
    print(status_log)
    log_file.write(status_log)

    # Save all metrics and scores
    json.dump(n_samples, open(f'{log_dir_path}/n_samples/{name}.json', 'w'))

    df_metrics = pd.DataFrame([metrics_exp])
    df_metrics.to_csv(f'{log_dir_path}/result_metrics/{name}.csv', index=False)  

    np.save(f'{log_dir_path}/ood_scores/{name}.npy', ood_scores)

    plt.hist(ood_scores[gt_indicators['ID']], bins=threshold_range, label='ID', alpha=0.5)
    plt.hist(ood_scores[gt_indicators['OOD']], bins=threshold_range, label='OOD', alpha=0.5)
    plt.savefig(f'{log_dir_path}/ood_scores/{name}.jpg')

    return metrics_exp