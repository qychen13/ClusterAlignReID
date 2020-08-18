import os
from tqdm import tqdm
import torch
import torch.nn as nn

from arguments import ArgumentsTest
from config import get_config
from datasets import construct_dataset
from models import construct_model
from utils.evaluation import test


def main():
    ############ arguments setup #############
    args = ArgumentsTest().parse_args()
    print('***********************Arguments***********************')
    print(args)

    ############ get configuration info #############
    config = get_config(args)
    print('***********************Configurations***********************')
    print(config)

    ########### get model setup ############
    model = construct_model(args, config)
    print('***********************Model************************')
    print(model)
    if args.gpu_ids is not None:
        model.cuda(args.gpu_ids[0])

    ########### restore the model weights #########
    if args.gpu_ids is None:
        checkpoint = torch.load(args.restore_file)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu_ids[0])
        checkpoint = torch.load(args.restore_file, map_location=loc)

    print('==> Resume checkpoint {}'.format(args.restore_file))
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if 'transfer' in args.version:
        checkpoint = {key: val for key, val in checkpoint.items() if 'classifier' not in key}
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    else:
        model.load_state_dict(checkpoint)

    ############ dataset setup #############
    query_iterator, gallery_iterator = construct_dataset(args, config)

    ############ start testing #############
    torch.backends.cudnn.benchmark = True
    model.eval()

    if 'neck-fs' in args.version or 'external' in args.model_name:
        feature_extractor = model
    else:
        feature_extractor = model.feature_extractor

    if 'store-fs' in args.version:
        store_fs = True
    else:
        store_fs = False

    if 'test-external' in args.version:
        test_method = 'external'
    elif 'test-org' in args.version:
        test_method = 'euclidean'
    elif 'test-bnneck' in args.version:
        test_method = 'euclidean-normal'
    else:
        test_method = 'cosine'

    if 'flips' in args.version:
        flips = True
    else:
        flips = False

    if 'reranking' in args.version:
        reranking = True
    else:
        reranking = False

    result = test(feature_extractor, query_iterator, gallery_iterator,
            args.gpu_ids, store_fs=store_fs, method=test_method, flips=flips, reranking=reranking)

    ############ print result #############
    print('*******************Test Results************************')
    for key in result:
        print('{}: {}'.format(key, result[key]))


if __name__ == '__main__':
    main()
