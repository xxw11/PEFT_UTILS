import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SSFLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super(SSFLinear, self).__init__(in_features=in_features, out_features=out_features)
        self.SSFscale, self.SSFshift = nn.Parameter(torch.ones(1,in_features)),nn.Parameter(torch.zeros(1,in_features))
        # nn.init.normal_(self.SSFscale, mean=1, std=.02)
        # nn.init.normal_(self.SSFshift, mean=0, std=.02)

    def forward(self, input):
        return F.linear(input*self.SSFscale + self.SSFshift, self.weight, self.bias)

    @staticmethod
    def from_linear(linear_module):
        new_linear = SSFLinear(linear_module.in_features, linear_module.out_features)
        new_linear.weight = linear_module.weight
        new_linear.bias = linear_module.bias
        return new_linear

class SSFModuleInjection:
    @staticmethod
    def make_scalable(linear_module):
        """Make a (linear) layer super scalable.
        :param linear_module: A Linear module
        :return: a suepr linear that can be trained to
        """
        new_linear = SSFLinear.from_linear(linear_module)
        return new_linear
    
def set_ssf(model):
    layers = []
    for name, l in model.named_modules():
        if isinstance(l, nn.Linear):
            tokens = name.strip().split('.')
            layer = model
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]
            layers.append([layer, tokens[-1]])
    for parent_layer, last_token in layers:
        if not 'head' in last_token:
            setattr(parent_layer, last_token, SSFModuleInjection.make_scalable(getattr(parent_layer, last_token)))

@torch.no_grad()
def save_ssf(save_path, model):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if any([x in n for x in ['SSF']]):
            trainable[n] = p.data
    torch.save(trainable, save_path)
    
def load_ssf(load_path, model):
    weights = torch.load(load_path)
    loaded = 0
    for n, p in model.named_parameters():
        if any([x in n for x in ['SSF']]):
            p.data = weights[n]
            loaded +=1
    print(f'successfully loaded {loaded} trained parameter tensors')
    return model