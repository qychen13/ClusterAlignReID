import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from .evaluation import fliplr

def calculate_id_features(feature_extractor, training_iterator, gpu_ids, method='avg', flips=True):
    print('==> extracting training features...')
    with torch.no_grad():
        fets, targets, probs = extract_features_for_id(
            feature_extractor, training_iterator, gpu_ids, is_test=False, flips=flips)
        id_features = defaultdict(list)
        prob_dict = defaultdict(list)
        for i in range(fets.shape[0]):
            id_features[targets['pid'][i].item()].append(fets[i])
            prob_dict[targets['pid'][i].item()].append(probs[i])

        id_features = [torch.stack(id_features[i], dim=0) for i in range(len(id_features))]
        prob_dict = [torch.stack(prob_dict[i], dim=0) for i in range(len(prob_dict))]
        if method == 'avg':
            id_features = [id_fet.mean(dim=0) for id_fet in id_features]
        elif method == 'weight-prob':
            id_features = [(id_fet*weight.unsqueeze(1)).sum(dim=0)/weight.sum() for id_fet, weight in zip(id_features, prob_dict)]
        else:
            raise NotImplementedError

        id_features = torch.stack(id_features, dim=0)
        print('==> calculate ID features done...')

    return id_features

def update_id_features(fet, target, moumentum=0.9):
    if isinstance(fet, dict):
        fet = fet['features']
    with torch.no_grad():
        id_features = target['id_feature_dict']
        id_set = set(target['pid'][i] for i in range(target['pid'].shape[0]))
        for pid in id_set:
            id_feature_update = fet[target['pid']==pid]
            id_feature_update = fet[target['pid']==pid].mean(0)
            id_features[pid] = moumentum * id_features[pid] + (1 - moumentum) * id_feature_update
    return id_features

def extract_features_for_id(feature_extractor, data_iterator, gpu_ids, is_test, flips=False):
    feature_extractor.eval()
    fets = []
    probs = []
    targets = defaultdict(list)
    if gpu_ids is not None:
        feature_extractor.cuda(gpu_ids[0])

    with torch.no_grad():
        one_hot_class = None
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
                opt = nn.parallel.data_parallel(
                    feature_extractor, ipt, gpu_ids)
                opt1 = nn.parallel.data_parallel(
                    feature_extractor, fliplr(ipt), gpu_ids)
            else:
                opt = feature_extractor(ipt)
                opt1 = feature_extractor(fliplr(ipt))

            if 'features_test' in opt and is_test:
                #print('==> use features_test')
                key = 'features_test'
            else:
                key = 'features'

            if one_hot_class is None:
                if isinstance(opt['logits'], list):
                    logits = opt['logits'][0]
                else:
                    logits = opt['logits']
                num_classes = logits.shape[1]
                one_hot_class = torch.eye(num_classes, device=logits.device)

            fet = opt[key]
            fet1 = opt1[key]

            labels = target['pid']
            pos_mask = one_hot_class[labels].type(torch.bool)
            if isinstance(opt['logits'], list):
                prob = [F.softmax(opt['logits'][i], dim=1).masked_select(pos_mask) for i in range(len(opt['logits']))]
                prob = torch.stack(prob, 1)
                prob1 = [F.softmax(opt1['logits'][i], dim=1).masked_select(pos_mask) for i in range(len(opt['logits']))]
                prob1 = torch.stack(prob1, 1)
            else:
                prob = F.softmax(opt['logits'], dim=1).masked_select(pos_mask)
                prob1 = F.softmax(opt1['logits'], dim=1).masked_select(pos_mask)
            if flips:
                fet += fet1
                fet /= 2
                prob += prob1
                prob /=2

            fets.append(fet)
            probs.append(prob)
            for key in target:
                targets[key].append(target[key])

    fets = torch.cat(fets, 0)
    probs = torch.cat(probs)
    for key in targets:
        if isinstance(targets[key][0], list):
            temp = []
            for t in targets[key]:
                temp += t
            targets[key] = temp
        else:
            targets[key] = torch.cat(targets[key], 0)

    return fets, targets, probs