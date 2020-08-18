'''
Motivated by torchnet
'''
from tqdm import tqdm

import torch
import torch.backends.cudnn
import torch.nn as nn


class Engine(object):
    def __init__(self, gpu_ids, network, criterion, train_iterator, validate_iterator, optimizer):
        self.hooks = {}
        self.state = {
            'gpu_ids': gpu_ids,
            'network': network,
            'train_iterator': train_iterator,
            'validate_iterator': validate_iterator,
            'maxepoch': None,
            'epoch': None,
            'iteration': None,
            'train': True,
            'output': None,
            'loss': None,
            'optimizer': optimizer,
            'criterion': criterion,
        }

        # set cudnn bentchmark
        torch.backends.cudnn.benchmark = True

    def hook(self, name, state):
        self.hooks[name](state)

    def forward(self, ipt, target):
        state = self.state
        if state['gpu_ids'] is not None:
            output = nn.parallel.data_parallel(
                state['network'], ipt, state['gpu_ids'])
        else:
            output = state['network'](ipt)

        loss = state['criterion'](output, target)
        state['output'] = output
        state['loss'] = loss
        self.hook('on_end_forward', state)

        return loss

    def resume(self, maxepoch, epoch, iteration):
        state = self.state

        state['epoch'] = epoch
        state['iteration'] = iteration
        state['maxepoch'] = maxepoch
        state['train'] = True
        self.hook('on_start', state)

        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)
            state['network'].train() # TODO: may need extra attention for id only loss

            for sample in tqdm(state['train_iterator']):
                state['sample'] = sample
                self.hook('on_end_sample', state)
                ipt, target = state['sample'][0], state['sample'][1]

                loss = self.forward(ipt, target)

                state['optimizer'].zero_grad()
                if isinstance(loss, dict):
                    loss = loss['total_loss']
                loss.backward()
                state['optimizer'].step()
                self.hook('on_end_update', state)
                state['iteration'] += 1

                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None

            self.hook('on_end_epoch', state)
            self.validate()
            state['epoch'] += 1

        self.hook('on_end', state)
        return state

    def train(self, maxepoch):
        self.resume(maxepoch, 0, 0)

    def validate(self):
        if self.state['validate_iterator'] is None:
            return
        state = self.state
        state['train'] = False

        self.hook('on_start', state)
        state['network'].eval()

        with torch.no_grad():
            for sample in tqdm(state['validate_iterator']):
                state['sample'] = sample
                self.hook('on_end_sample', state)
                ipt, target = state['sample'][0], state['sample'][1]

                self.forward(ipt, target)

                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None

        self.hook('on_end', state)
        return state
