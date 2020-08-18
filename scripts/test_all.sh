# market1501 resnet50
python test.py -b 256 -gs 0 -nw 4 -ds 'market1501' -ddir /media/DatasetSSD/Market-1501-v15.09.15 -model 'external-bnneck' -vs 'train-all_flips' -rf pretrained_models/strong-baseline-market-bnneck-stage2/e159t30071.pth.tar
# market1501 resnet50
python test.py -b 256 -gs 0 -nw 4 -ds 'market1501' -ddir /media/DatasetSSD/Market-1501-v15.09.15 -model 'external-bnneck-ibn-a' -vs 'train-all_flips_duke-large-input' -rf pretrained_models/strong-baseline-market-bnneck-ibn-a-stage2/e159t30075.pth.tar
# dukemtm resnet50
python test.py -b 256 -gs 0 -nw 4 -ds 'dukemtm' -ddir /media/DatasetSSD/DukeMTMC-reID -model 'external-bnneck' -vs 'train-all_flips_duke-large-input' -rf pretrained_models/strong-baseline-duke-bnneck-stage2/e159t37434.pth.tar
# dukemtm resnet50-ibn-a
python test.py -b 256 -gs 0 -nw 4 -ds 'dukemtm' -ddir /media/DatasetSSD/DukeMTMC-reID -model 'external-bnneck-ibn-a' -vs 'train-all_flips_duke-large-input' -rf pretrained_models/strong-baseline-duke-bnneck-ibn-a-stage2/e139t32279.pth.tar