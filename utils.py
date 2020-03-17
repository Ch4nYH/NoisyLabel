import argparse
import torch
def get_args():
    parser = argparse.ArgumentParser(description='RL')
    ##### 
    parser.add_argument(
        '--lr',
        type=float,
        default = 1e-3
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default = 'mnist'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default = ""
    )
    parser.add_argument(
        '--modeldir',
        type=str,
        default = "model"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default = 1000
    )
    parser.add_argument(
        '-j',
        '--num-workers',
        type = int,
        default = 8
    )
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument(
        '--epochs',
        type=int,
        default = 200
    )
    args = parser.parse_args()
    return args

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        #print(target)
        if (target.dim() > 1):
            target = torch.argmax(target, 1)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0].item()