import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def parameters(self):
       for name, param in self.named_params(self):
            yield param
    
    def named_leaves(self):
        return []
    
    def named_submodules(self):
        return []
    
    def named_params(self, curr_module=None, memo=None, prefix=''):       
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
                    
        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p
    
    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                if grad is None:
                    print(name_t)
                    tmp = param_t
                else:
                    tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    if grad is None:
                        print(name)
                        tmp = param_t
                    else:
                        tmp = param_t - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self,curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)
            
    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())   
                
    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
       
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        if ignore.bias is not None: self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else: self.bias = None
    def forward(self, x):
        if self.bias is not None:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight)
    
    def named_leaves(self):
        if self.bias is not None:
            return [('weight', self.weight), ('bias', self.bias)]
        else:
            return [('weight', self.weight)]
    
class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)
        
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaConv3d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv3d(*args, **kwargs)
        
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x):
        return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)
        
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
       
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
class MetaBatchNorm3d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm3d(*args, **kwargs)
        
        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:           
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

        
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                        self.training or not self.track_running_stats, self.momentum, self.eps)
            
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)
        
        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:           
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

        
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                        self.training or not self.track_running_stats, self.momentum, self.eps)
            
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
    
class MetaBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm1d(*args, **kwargs)
        
        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats
       
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

        
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                        self.training or not self.track_running_stats, self.momentum, self.eps)
        
    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
        

class LeNet(MetaModule):
    def __init__(self, n_out):
        super(LeNet, self).__init__()
    
        layers = []
        layers.append(MetaConv2d(1, 6, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layers.append(MetaConv2d(6, 16, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        
        layers.append(MetaConv2d(16, 120, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        
        self.main = nn.Sequential(*layers)
        
        layers = []
        layers.append(MetaLinear(120, 84))
        layers.append(nn.ReLU(inplace=True))
        layers.append(MetaLinear(84, n_out))
        
        self.fc_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 120)
        return self.fc_layers(x).squeeze()
    
class MetaMLP(MetaModule):
    def __init__(self, dim):
        super(MetaMLP, self).__init__()
        layers = []
        self.l1 = MetaLinear(dim, 300, bias=False)
        self.bn1 = MetaBatchNorm1d(num_features = 300)
        self.l2 = MetaLinear(300, 300, bias=False)
        self.bn2 = MetaBatchNorm1d(num_features = 300)
        self.l3 = MetaLinear(300, 300, bias=False)
        self.bn3 = MetaBatchNorm1d(num_features = 300)
        self.l4 = MetaLinear(300, 300, bias=False)
        self.bn4 = MetaBatchNorm1d(num_features = 300)
        self.l5 = MetaLinear(300, 1)
    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.l4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = x.view(-1, 300)
        x = self.l5(x)
        return x
    
class Model(MetaModule):
    def __init__(self, num_classes = 10, input_channel = 3):
        super(Model, self).__init__()

        feature_layers = [
            MetaConv2d(input_channel, 128, 3, padding = 2),
            nn.LeakyReLU(negative_slope = 0.1, inplace = True),
            nn.BatchNorm2d(128),
            MetaConv2d(128, 128, 3, padding = 2),
            nn.LeakyReLU(negative_slope = 0.1, inplace = True),
            nn.BatchNorm2d(128),
            MetaConv2d(128, 128, 3, padding = 2),
            nn.LeakyReLU(negative_slope = 0.1, inplace = True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride = 2),
            nn.Dropout(p = 0.25),
            MetaConv2d(128, 256, 3, padding = 2),
            nn.LeakyReLU(negative_slope = 0.1, inplace = True),
            nn.BatchNorm2d(256),
            MetaConv2d(256, 256, 3, padding = 2),
            nn.LeakyReLU(negative_slope = 0.1, inplace = True),
            nn.BatchNorm2d(256),
            MetaConv2d(256, 256, 3, padding = 2),
            nn.LeakyReLU(negative_slope = 0.1, inplace = True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride = 2),
            nn.Dropout(p = 0.25),
            MetaConv2d(256, 512, 3, padding = 2),
            nn.LeakyReLU(negative_slope = 0.1, inplace = True),
            nn.BatchNorm2d(512),
            MetaConv2d(512, 256, 3, padding = 2),
            nn.LeakyReLU(negative_slope = 0.1, inplace = True),
            nn.BatchNorm2d(256),
            MetaConv2d(256, 128, 3, padding = 2),
            nn.LeakyReLU(negative_slope = 0.1, inplace = True),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1,1)),
        ]
        self.feature = nn.Sequential(*feature_layers)

        self.classifier = MetaLinear(128, num_classes, bias = False)

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(-1, 128)
        out = self.classifier(feature)
        return out