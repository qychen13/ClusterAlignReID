import torch
import torch.nn.functional as functional


class AverageValueMeter(object):
    # copy from tnt
    def __init__(self):
        self.reset()
        self.val = 0

    def add(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

    def value(self):
        return self.avg

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


class MetricBase(AverageValueMeter):
    def __init__(self):
        super(MetricBase, self).__init__()

    @staticmethod
    def caculate(target, logits, loss):
        raise NotImplementedError

    def __call__(self, target, logits, loss):
        n = target['pid'].shape[0]
        v = self.caculate(target, logits, loss)
        self.add(v, n)


class KeyLossMetric(MetricBase):
    def __init__(self, key):
        super(KeyLossMetric, self).__init__()
        self.key = key

    def caculate(self, target, logits, loss):
        if self.key == None:
            return loss
        return loss[self.key].item()


class ClassAccuracy(MetricBase):
    def __init__(self, top_k=1, index=None):
        super(ClassAccuracy, self).__init__()
        self.top_k = top_k
        self.index = index

    def caculate(self, target, logits, loss):
        target = target['pid']
        logits = logits['logits']
        if self.index is not None:
            logits = logits[self.index]
        if isinstance(logits, list):
            # normalization
            logits = functional.softmax(torch.stack(logits, dim=1), dim=2)
            logits = logits.mean(dim=1)

        batch_size = target.shape[0]

        _, pred = logits.topk(self.top_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:self.top_k].view(-1).float().sum(0, keepdim=True)
        res = correct_k.mul_(100.0 / batch_size)
        return res.item()
