# stage 1
python trainval.py -b 64 -nw 4 -vb 256 -gs 0 -ds 'market1501' -ddir /media/DatasetSSD/Market-1501-v15.09.15 -model 'external-bnneck' -vs 'train-all_warm-up-v1_adam_triplet_label-smooth_erasing_test-bnneck-flips' -lr 3.5e-4 -logf 100 -cptf 10 -me 120 -lcd strong-baseline-market-bnneck-stage1
# stage 2
python trainval.py -b 64 -nw 4 -vb 256 -gs 0 -ds 'market1501' -ddir /media/DatasetSSD/Market-1501-v15.09.15 -model 'external-bnneckv1' -vs 'train-all_const-lr_new-optim_adam_id_erasing_test-bnneck-flips' -lr 3.5e-6 -logf 100 -cptf 10 -me 160 -lcd strong-baseline-market-bnneck-stage2 -rf checkpoints/strong-baseline-market-bnneck-stage1/e119t21990.pth.tar -re 119 -ri 21990