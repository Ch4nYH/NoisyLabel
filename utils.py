import argparse
import torch
import numpy as np
def get_args():
    parser = argparse.ArgumentParser(description='RL')
    ##### 
    parser.add_argument('-l', '--lr', type=float, default = 1e-3 )
    parser.add_argument('-d', '--dataset', type=str, default = 'mnist' )
    parser.add_argument('-g', '--gpu', type=str, default = "" )
    parser.add_argument('-m', '--modeldir', type=str, default = "model" )
    parser.add_argument('-b', '--batch-size', type=int, default = 1000 )
    parser.add_argument('-j', '--num-workers', type = int, default = 8 )
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('-e', '--epochs', type=int, default = 100 )
    parser.add_argument('-p', '--percent', type=float, default = 0.5 )
    parser.add_argument('-c', '--components', type=str, nargs="+" )    
    parser.add_argument('-a', '--arch', type=str, default = "default")
    parser.add_argument('--clamp', action="store_true" )
    parser.add_argument('--gamma', type=float, default = 1.0 )
    parser.add_argument('--prefix', default="models", type=str )
    parser.add_argument('--num-classes', default=10, type=int)
    parser.add_argument('--val-batch-size', default = None, type = int)
    parser.add_argument('--with-kl', action="store_true")

    args = parser.parse_args()
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
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
    
    
def get_val_samples(iter_val_loader, val_loader):
    try:
        val_input, val_label, _ = next(iter_val_loader)
    except:
        iter_val_loader = iter(val_loader)
        val_input, val_label, _ = next(iter_val_loader)
    
    return  val_input, val_label, iter_val_loader


class WLogger(object):
    def __init__(self):
        self.w = []
        
    def update(self, w):
        if isinstance(w, torch.Tensor):
            w = w.detach().cpu().numpy().flatten()
        assert isinstance(w, np.ndarray)
        
        self.w.append(w)
    def cleanup(self):
        self.w = []
    def write(self, writer, name, epoch):
        w = np.concatenate(self.w, 0)
        writer.add_scalar('w_' + name, np.sum(w), epoch)
        writer.add_histogram('w_' + name, np.sum(w), epoch)
    def mask_write(self, writer, name, epoch, mask):
        w = np.concatenate(self.w, 0)
        writer.add_scalar('masked_w_' + name, np.sum(w[mask]) / np.sum(w), 0)
        
        
class ScalarLogger(object):
    def __init__(self, prefix):
        self.scalars = []
        self.prefix = prefix
    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.item()
        assert isinstance(scalar, float)
        
        self.scalars.append(scalar)
    def write(self, writer, name, epoch):
        avg_scalar = np.mean(np.array(self.scalars))
        writer.add_scalar(self.prefix + "_" + name, avg_scalar, epoch)
    
    def avg(self):
        return np.mean(np.array(self.scalars))