import argparse
import torch
def get_args():
    parser = argparse.ArgumentParser(description='RL')
    ##### 
    parser.add_argument(
        '-l',
        '--lr',
        type=float,
        default = 1e-3
    )
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        default = 'mnist'
    )
    parser.add_argument(
        '-g',
        '--gpu',
        type=str,
        default = ""
    )
    parser.add_argument(
        '-m', 
        '--modeldir',
        type=str,
        default = "model"
    )
    parser.add_argument(
        '-b',
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
        '-s',
        '--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default = 100
    )

    parser.add_argument(
        '-p',
        '--percent',
        type=float,
        default = 0.5
    )
    
    parser.add_argument(
        '-c',
        '--components',
        type=str,
        nargs="+"
    )
    
    parser.add_argument(
        '--clamp',
        action="store_true"
    )

    parser.add_argument(
        '--gamma',
        type=float,
        default = 1.0
    )

    parser.add_argument(
        '--prefix',
        default="models",
        type=str
    )

    args = parser.parse_args()
    for i in args.components:
        assert i in ['all', 'fc', 'backbone']
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