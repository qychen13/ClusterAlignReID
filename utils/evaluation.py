import time
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as functional
import scipy.io
import numpy as np

from .distance import compute_distance_matrix
from .rank import evaluate_rank
from .rerank import re_ranking


def test(feature_extractor, query_iterator, gallary_iterator, gpu_ids, store_fs=False, method='euclidean', flips=False, reranking=False):
    print('==> extracting query features...')
    qfets, qtargets = extract_features(
        feature_extractor, query_iterator, gpu_ids, is_test=True, flips=flips)
    print('==> extracting gallary features...')
    gfets, gtargets = extract_features(
        feature_extractor, gallary_iterator, gpu_ids, is_test=True, flips=flips)

    print('==> compute test metrics...')

    if store_fs:
        # used for external test
        print('==> save features and labels...')
        qnorm = torch.norm(qfets, p=2, dim=1, keepdim=True)
        qfets_n = qfets.div(qnorm.expand_as(qfets))
        gnorm = torch.norm(gfets, p=2, dim=1, keepdim=True)
        gfets_n = gfets.div(gnorm.expand_as(gfets))
        features = {'gallery_f': gfets_n.cpu().numpy(),
                    'gallery_label': gtargets['pid'].cpu().numpy(),
                    'gallery_cam': gtargets['camid'].cpu().numpy(),
                    'query_f': qfets_n.cpu().numpy(),
                    'query_label': qtargets['pid'].cpu().numpy(),
                    'query_cam': qtargets['camid'].cpu().numpy()}
        scipy.io.savemat('pytorch_result.mat', features)

    if method == 'external':
        if reranking:
            raise NotImplementedError
        result = evaluate_gpu(qfets, gfets, qtargets, gtargets, normalize=True)
    else:
        result = compute_test_metrics(
            qfets, gfets, qtargets, gtargets, metric=method, reranking=reranking)

    return dict(Top1=result['all_cmc'][0], Top5=result['all_cmc'][4], mAP=result['mAP'])


def test_external(result_file):
    result = scipy.io.loadmat(result_file)
    query_feature = torch.FloatTensor(result['query_f'])
    query_cam = torch.IntTensor(result['query_cam'][0])
    query_label = torch.IntTensor(result['query_label'][0])
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_cam = torch.IntTensor(result['gallery_cam'][0])
    gallery_label = torch.IntTensor(result['gallery_label'][0])

    qtargets = dict(pid=query_label, camid=query_cam)
    gtargets = dict(pid=gallery_label, camid=gallery_cam)

    good_index = (gtargets['pid'] != -1)
    gallery_feature = gallery_feature[good_index]
    gtargets = {key: gtargets[key][good_index] for key in gtargets}

    # result = compute_test_metrics(query_feature, gallery_feature, qtargets, gtargets, metric='cosine-non-normal')
    result = evaluate_gpu(query_feature, gallery_feature, qtargets, gtargets)

    return dict(Top1=result['all_cmc'][0], Top5=result['all_cmc'][4], mAP=result['mAP'])


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1, -1, -
                           1).long()  # N x C x H x W
    if img.is_cuda:
        inv_idx = inv_idx.cuda()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_features(feature_extractor, data_iterator, gpu_ids, is_test, flips=False):
    feature_extractor.eval()
    fets = []
    targets = defaultdict(list)
    if gpu_ids is not None:
        feature_extractor.cuda(gpu_ids[0])

    with torch.no_grad():
        for ipt, target in tqdm(data_iterator):
            if gpu_ids is not None:
                if len(gpu_ids) == 1:
                    ipt = ipt.cuda(gpu_ids[0], non_blocking=True)
                for key in target:
                    if isinstance(target[key], list):
                        continue
                    target[key] = target[key].cuda(
                        gpu_ids[0], non_blocking=True)

            if gpu_ids is not None:
                fet = nn.parallel.data_parallel(
                    feature_extractor, ipt, gpu_ids)
                fet1 = nn.parallel.data_parallel(
                    feature_extractor, fliplr(ipt), gpu_ids)
            else:
                fet = feature_extractor(ipt)
                fet1 = feature_extractor(fliplr(ipt))

            if isinstance(fet, dict):
                if 'features_test' in fet and is_test:
                    #print('==> use features_test')
                    key = 'features_test'
                else:
                    key = 'features'
                fet = fet[key]
                fet1 = fet1[key]
            if flips:
                fet += fet1
                fet /= 2

            fets.append(fet)
            for key in target:
                targets[key].append(target[key])
    fets = torch.cat(fets, 0)
    for key in targets:
        if isinstance(targets[key][0], list):
            temp = []
            for i in targets[key]:
                temp += i
            targets[key] = temp
        else:
            targets[key] = torch.cat(targets[key], 0)

    return fets, targets


def compute_test_metrics(qfets, gfets, qtargets, gtargets, metric='euclidean', reranking=False):
    distmat = compute_distance_matrix(qfets, gfets, metric=metric)
    if reranking:
        print('==> reranking the distance matrix at {} ...'.format(
            time.strftime('%c')))
        distmat = distmat.cpu()
        q_q_distmat = compute_distance_matrix(
            qfets, qfets, metric=metric).cpu()
        g_g_distmat = compute_distance_matrix(
            gfets, gfets, metric=metric).cpu()
        distmat = re_ranking(distmat, q_q_distmat, g_g_distmat)
        distmat = torch.from_numpy(distmat)
        print('==> done at {} ...'.format(time.strftime('%c')))

    q_pids = qtargets['pid']
    g_pids = gtargets['pid']
    q_camids = qtargets['camid']
    g_camids = gtargets['camid']

    return evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, use_cython=True)


################   copy from person-reid-baseline  ####################
def evaluate_gpu(qfets, gfets, qtargets, gtargets, normalize=False):
    # TODO: support reranking
    if normalize:
        qnorm = torch.norm(qfets, p=2, dim=1, keepdim=True)
        qfets = qfets.div(qnorm.expand_as(qfets))
        gnorm = torch.norm(gfets, p=2, dim=1, keepdim=True)
        gfets = gfets.div(gnorm.expand_as(gfets))

    if len(qfets.shape) != 2:
        qfets = qfets.view(qfets.shape[0], -1)
        gfets = gfets.view(gfets.shape[0], -1)

    query_feature = qfets
    query_cam = qtargets['camid'].cpu()
    query_label = qtargets['pid'].cpu()
    gallery_feature = gfets
    gallery_cam = gtargets['camid'].cpu()
    gallery_label = gtargets['pid'].cpu()

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    # print(query_label)
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(
            query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])

    CMC = CMC.float()
    CMC = CMC/len(query_label)

    return dict(all_cmc=CMC, mAP=ap/len(query_label))

# Evaluate


def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i] != 0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc
