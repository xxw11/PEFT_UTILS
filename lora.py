import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class LoRALinear(torch.nn.Linear):
    def __init__(self, in_features, out_features,rank,lora_alpha=32,lora_dropout=0.01):
        super(LoRALinear, self).__init__(in_features=in_features, out_features=out_features)
        self.alpha = lora_alpha
        self.r = rank
        self.scaling = self.alpha / self.r
        self.lora_B, self.lora_A = nn.Parameter(torch.zeros(out_features, rank)), nn.Parameter(torch.zeros(rank, in_features))
        self.dropout = nn.Dropout(p=lora_dropout)


    def forward(self, input):
        result = F.linear(input, self.weight, self.bias)
        result += (self.dropout(input) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result
        
    @staticmethod
    def from_linear(linear_module,rank):
        new_linear = LoRALinear(linear_module.in_features, linear_module.out_features,rank)
        new_linear.weight = linear_module.weight
        new_linear.bias = linear_module.bias
        return new_linear

class LoRAModuleInjection:
    @staticmethod
    def make_scalable(module,rank):
        """Make a LoRA module
        """
        if isinstance(module, nn.Linear):
            new_linear = LoRALinear.from_linear(module,rank)
            return new_linear
        elif isinstance(module, nn.Linear):
            pass
        
def set_lora(model):
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
            setattr(parent_layer, last_token, LoRAModuleInjection.make_scalable(getattr(parent_layer, last_token)))

@torch.no_grad()
def save_lora(save_path, model):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if any([x in n for x in ['lora']]):
            trainable[n] = p.data
    torch.save(trainable, save_path)
    
def load_lora(load_path, model):
    weights = torch.load(load_path)
    loaded = 0
    for n, p in model.named_parameters():
        if any([x in n for x in ['lora']]):
            p.data = weights[n]
            loaded +=1
    print(f'successfully loaded {loaded} trained parameter tensors')
    return model

if __name__ == '__main__':
    input_tensor = torch.randn(1,1,20)
    linear = nn.Linear(20, 10, 4)
    out1 = linear(input_tensor)
    lora_linear = LoRALinear.from_linear(linear,4)
    out2 = lora_linear(input_tensor)
    print(out1)
    print(out2)
    # nn.init.normal_(repadpter_linear.adapter.conv_A.weight)
    # nn.init.normal_(repadpter_linear.adapter.conv_B.weight)
    # repadpter_linear.eval()
    # output_repadpter1 = repadpter_linear(input_tensor)
    # print("RepAdapterLinear 输出结果1:", output_repadpter1)
    # linear_module = RepAdapterLinear.to_linear(repadpter_linear)
    # output_repadpter2 = repadpter_linear(input_tensor)
    # print("RepAdapterLinear 输出结果2:", output_repadpter2)
    # output_linear = linear_module(input_tensor)
    # print("Linear            输出结果:", output_linear)